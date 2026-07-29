"""
export_poma_densify.py

Bathymetry-densification prototype (2026-07-29): exports the FULL set of
candidate Sentinel-1 masks in Poma's Period-B window, instead of the 10
WATER-LEVEL-stratified dates export_windowed_masks.py normally picks.

Why: the current DEM reconstruction only stacks ~10 waterline slices, so
between two consecutive calibrated levels the level-slice method interpolates
a single flat elevation across the whole exposed ring -- any real slope
feature in that gap (a terrace, a berm, a steep step) is invisible. Poma's
window (2025-12-15 to 2026-05-10) has 26 candidate S1 acquisitions on the
same orbit vs. the 10 already exported, and has NO declared gauge_bad_window
in this period, so almost every extra mask gets a REAL gauge-matched water
level (schwatke_bathymetry_3d.phase1's own source-priority chain), not a
curve-inferred one -- isolating the density question from the mixed-source
question. Reuses get_lake_poly / select_best_orbit / classify_image from
export_windowed_masks.py unchanged, so the extra masks are on the identical
grid/orbit/classifier as the 10 already on disk.

This script ONLY adds masks. It does not touch selected_mask_dates.json or
any file the manuscript pipeline reads, so the production Poma DEM/curve
numbers are unaffected until a separate step explicitly opts them in.

Run:
    python analysis/export_poma_densify.py --dry-run
    python analysis/export_poma_densify.py
"""
import sys, pathlib
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import ee

try:
    ee.Initialize(project='ee-ciceromartinsjr')
except Exception:
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project='ee-ciceromartinsjr')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from export_windowed_masks import (
    get_lake_poly, select_best_orbit, classify_image, candidate_dates_of,
    S1_GRD, JRC_GSW, JRC_OCC, WC, CFG, BANDS, NO_JRC_REFINE,
)

NAME = 'Poma'
LAT, LON, HYLAK_ID = 38.011037, 13.056135, 173610
WIN_START, WIN_END = '2025-12-15', '2026-05-10'
KNOWN_ORBIT = None
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')


def main(dry_run=False):
    lake_poly = get_lake_poly(LAT, LON, HYLAK_ID, jrc_refine=NAME not in NO_JRC_REFINE)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(WIN_START, WIN_END)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    if KNOWN_ORBIT is not None:
        orbit_num, pass_dir = KNOWN_ORBIT, 'ASCENDING'
    else:
        orbit_num, pass_dir = select_best_orbit(s1_raw, aoi)
    best_raw = s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num)) \
                     .filter(ee.Filter.eq('orbitProperties_pass', pass_dir))

    n = best_raw.size().getInfo()
    print(f'{NAME}: window {WIN_START}..{WIN_END}  orbit={orbit_num} ({pass_dir})  n={n}')
    if n == 0:
        print('  NO images in window on this orbit -- aborting'); return

    candidate_strs = candidate_dates_of(best_raw)
    print(f'  {len(candidate_strs)} candidate dates on this orbit (full set, no WL stratification):')
    print(f'  {candidate_strs}')

    new_dates = [d for d in candidate_strs if not (MASK_DIR / f'mask_{NAME}_{d}.tif').exists()]
    print(f'  {len(candidate_strs) - len(new_dates)} already on disk, {len(new_dates)} need export')
    print(f'  New dates: {new_dates}')

    if dry_run or len(new_dates) == 0:
        print('  (dry-run or nothing to export -- stopping before task submission)')
        return

    s1_composite = (S1_GRD
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filterBounds(train_clip)
        .filter(ee.Filter.calendarRange(CFG['train_year'], CFG['train_year'], 'year'))
        .select(BANDS).mosaic()
        .focal_mean(30, 'circle', 'meters').clip(train_clip))

    water_strict = (JRC_OCC.gte(CFG['jrc_occ_thresh']).selfMask()
        .sample(region=lake_poly, scale=30, numPixels=500, seed=42, geometries=True)
        .map(lambda f: f.set('landcover', 1)))
    water_fallback = (JRC_OCC.gte(CFG['jrc_occ_fallback']).selfMask()
        .sample(region=lake_poly, scale=30, numPixels=500, seed=42, geometries=True)
        .map(lambda f: f.set('landcover', 1)))
    water_samples = ee.FeatureCollection(
        ee.Algorithms.If(water_strict.size().gt(10), water_strict, water_fallback))

    land_ring = lake_poly.buffer(CFG['land_ring_outer_m']).difference(lake_poly.buffer(CFG['land_ring_inner_m']))
    land_mask = (WC.neq(80).And(WC.neq(90)).And(WC.neq(95)).And(JRC_OCC.unmask(0).eq(0)).selfMask())
    land_samples = (land_mask
        .sample(region=land_ring, scale=30, numPixels=500, seed=42, geometries=True)
        .map(lambda f: f.set('landcover', 2)))

    trained = s1_composite.select(BANDS).sampleRegions(
        collection=water_samples.merge(land_samples), properties=['landcover'], scale=30,
    ).filter(ee.Filter.inList('landcover', [1, 2])).filter(ee.Filter.notNull(BANDS))
    svm = ee.Classifier.libsvm(kernelType='RBF', cost=1, gamma=0.01).train(
        features=trained, classProperty='landcover', inputProperties=BANDS)

    best_col = best_raw.map(lambda img: img.select(BANDS)
        .focal_mean(30, 'circle', 'meters')
        .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                              'relativeOrbitNumber_start', 'angle']))

    submitted = 0
    for date_str in new_dates:
        d0 = ee.Date(date_str)
        img = ee.Image(best_col.filterDate(d0, d0.advance(1, 'day')).first())
        cleaned = classify_image(img, svm, lake_poly)
        mask_out = cleaned.unmask(0).toByte()
        task = ee.batch.Export.image.toDrive(
            image=mask_out,
            description=f'mask_{NAME}_{date_str}_densify',
            folder=CFG['drive_folder'],
            fileNamePrefix=f'mask_{NAME}_{date_str}',
            region=lake_poly.buffer(300),
            scale=10, crs='EPSG:32633', maxPixels=1e9,
        )
        task.start()
        submitted += 1
    print(f'  Submitted {submitted} export tasks -> Drive/{CFG["drive_folder"]}/')


if __name__ == '__main__':
    main(dry_run='--dry-run' in sys.argv)
