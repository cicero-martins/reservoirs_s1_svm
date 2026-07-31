"""
export_ancipa_orbit124.py (2026-07-31)

Ancipa-specific multi-orbit densification. export_windowed_masks.py's
known_orbit path hardcodes pass_dir='ASCENDING', so it can only ever use
orbit 117 (the one Ancipa has always used). A one-off investigation this
session found Ancipa actually has 3 available orbits in its reconstruction
window (44 ASCENDING, 117 ASCENDING, 124 DESCENDING, ~9 scenes each, no
date collisions between them), and that orbit 124 resolves a real
near-dam radar-shadow gap in orbit 117's masks: at 2025-01-24 (WL rising
934.1->941.2 m from the previous production date) orbit 117's classified
area actually DROPS 58.27->42.41 ha, while orbit 124's classification one
day later (2025-01-25) shows water reaching the dam pixel and an area
(69.02 ha) that fits the rising trend correctly. Orbit 44 was checked and
rejected: same 100% AOI coverage but a much shallower incidence angle
(33.9 deg vs 117's 44.5 and 124's 39.2), producing severe layover across
most of the reservoir, not just near the dam -- not usable.

This exports all 9 orbit-124 DESCENDING dates as mask_Ancipa_{date}.tif,
same naming convention as orbit 117's masks, same grid (scale=10,
EPSG:32633, region=lake_poly.buffer(300)) -- densify_reservoir.py Ancipa
picks these up automatically since it just globs mask_Ancipa_*.tif in the
window, no other code changes needed.

Run:
    python analysis/export_ancipa_orbit124.py --dry-run
    python analysis/export_ancipa_orbit124.py
"""
import argparse, pathlib, sys

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
    get_lake_poly, classify_image, candidate_dates_of,
    S1_GRD, JRC_OCC, WC, CFG, BANDS, NO_JRC_REFINE, RESERVOIRS,
)

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
BY_NAME = {r[0]: r for r in RESERVOIRS}

NAME = 'Ancipa'
ORBIT_NUM, PASS_DIR = 124, 'DESCENDING'


def main(dry_run=False):
    _, lat, lon, hylak_id, win_start, win_end, _ = BY_NAME[NAME]

    lake_poly = get_lake_poly(lat, lon, hylak_id, jrc_refine=NAME not in NO_JRC_REFINE)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(win_start, win_end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    best_raw = (s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', ORBIT_NUM))
                      .filter(ee.Filter.eq('orbitProperties_pass', PASS_DIR)))

    n = best_raw.size().getInfo()
    print(f'{NAME}: window {win_start}..{win_end}  orbit={ORBIT_NUM} ({PASS_DIR})  n={n}')
    if n == 0:
        print(f'  {NAME}: NO images on orbit {ORBIT_NUM} -- skipping'); return 0

    candidate_strs = candidate_dates_of(best_raw)
    new_dates = [d for d in candidate_strs if not (MASK_DIR / f'mask_{NAME}_{d}.tif').exists()]
    print(f'  {NAME}: {len(candidate_strs)} candidates, '
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
            description=f'mask_{NAME}_{date_str}_orbit124',
            folder=CFG['drive_folder'],
            fileNamePrefix=f'mask_{NAME}_{date_str}',
            region=lake_poly.buffer(300),
            scale=10, crs='EPSG:32633', maxPixels=1e9,
        )
        task.start()
        submitted += 1
    print(f'  {NAME}: submitted {submitted} export tasks -> Drive/{CFG["drive_folder"]}/')
    return submitted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    main(dry_run=args.dry_run)
