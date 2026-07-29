"""
export_densify.py (2026-07-29)

Generalized version of export_poma_densify.py / export_rosamarina_densify.py:
exports the FULL set of candidate Sentinel-1 masks in a reservoir's Period-B
window (all 9 reservoirs supported), instead of the 10 WATER-LEVEL-stratified
dates export_windowed_masks.py normally picks. Reuses get_lake_poly /
select_best_orbit / classify_image from export_windowed_masks.py so the
extra masks are on the identical grid/orbit/classifier as the originals.

Ancipa is revisit-limited (candidates == dates already used, per
export_windowed_masks.py's own docstring) -- running it here is harmless
(will just report 0 new dates) but adds nothing.

Run:
    python analysis/export_densify.py Castello --dry-run
    python analysis/export_densify.py Castello
    python analysis/export_densify.py --all --dry-run
"""
import argparse, sys, pathlib

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
    S1_GRD, JRC_OCC, WC, CFG, BANDS, NO_JRC_REFINE, RESERVOIRS,
)

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
BY_NAME = {r[0]: r for r in RESERVOIRS}


def process(name, dry_run=False):
    if name not in BY_NAME:
        print(f'{name}: not found in export_windowed_masks.RESERVOIRS'); return 0
    _, lat, lon, hylak_id, win_start, win_end, known_orbit = BY_NAME[name]

    lake_poly = get_lake_poly(lat, lon, hylak_id, jrc_refine=name not in NO_JRC_REFINE)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(win_start, win_end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    if known_orbit is not None:
        orbit_num, pass_dir = known_orbit, 'ASCENDING'
    else:
        orbit_num, pass_dir = select_best_orbit(s1_raw, aoi)
    best_raw = s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num)) \
                     .filter(ee.Filter.eq('orbitProperties_pass', pass_dir))

    n = best_raw.size().getInfo()
    print(f'{name}: window {win_start}..{win_end}  orbit={orbit_num} ({pass_dir})  n={n}')
    if n == 0:
        print(f'  {name}: NO images in window on this orbit -- skipping'); return 0

    candidate_strs = candidate_dates_of(best_raw)
    new_dates = [d for d in candidate_strs if not (MASK_DIR / f'mask_{name}_{d}.tif').exists()]
    print(f'  {name}: {len(candidate_strs)} candidates, '
          f'{len(candidate_strs) - len(new_dates)} already on disk, {len(new_dates)} need export')
    print(f'  New dates: {new_dates}')

    if dry_run or len(new_dates) == 0:
        return 0

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
            description=f'mask_{name}_{date_str}_densify',
            folder=CFG['drive_folder'],
            fileNamePrefix=f'mask_{name}_{date_str}',
            region=lake_poly.buffer(300),
            scale=10, crs='EPSG:32633', maxPixels=1e9,
        )
        task.start()
        submitted += 1
    print(f'  {name}: submitted {submitted} export tasks')
    return submitted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reservoir', nargs='?', help='Reservoir name, or omit with --all')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    names = [r[0] for r in RESERVOIRS] if args.all else [args.reservoir]
    total = 0
    for name in names:
        total += process(name, dry_run=args.dry_run)
    print(f'\nTotal tasks submitted: {total}')
