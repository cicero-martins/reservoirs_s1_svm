"""
resubmit_failed_masks.py

Re-submits just the specific (reservoir, date) export tasks that failed with
"Image.select: input... may not be null" out of the original
export_extended_masks.py batch (56/120 tasks: Arancio 16, Castello 20, Olivo 20).

Diagnosis: re-querying the exact same collection/date interactively (see
project session notes) shows the image DOES exist with the correct VV/VH bands
today -- the failure does not reproduce on a fresh evaluation. This points to a
transient issue at the time those specific tasks actually executed on the EE
backend (not a systematic bug in the orbit/date-selection logic, which was
already fixed once for the "Image.select null" class of failure -- see
restratify_to_orbit_dates in export_extended_masks.py). A plain resubmit of the
same dates is therefore the correct fix to try first.

Run:
    python analysis/resubmit_failed_masks.py            # submit
    python analysis/resubmit_failed_masks.py --dry-run   # resolve AOIs only
"""
import sys

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

from export_extended_masks import (
    CFG, BANDS, S1_GRD, get_lake_poly, select_best_orbit, classify_image,
)

FAILED_DATES = {
    'Arancio': ['2014-12-24', '2015-03-30', '2015-05-17', '2015-07-28', '2015-09-14',
                '2015-10-20', '2016-04-05', '2022-05-16', '2022-09-25', '2023-04-05',
                '2024-01-30', '2024-03-30', '2024-05-05', '2024-12-31', '2025-02-17',
                '2025-11-26'],
    'Castello': ['2015-03-31', '2015-06-23', '2015-09-03', '2016-05-24', '2016-06-29',
                 '2016-07-23', '2016-09-09', '2016-09-21', '2016-10-27', '2016-12-08',
                 '2022-01-17', '2022-08-09', '2022-11-13', '2023-03-01', '2024-05-18',
                 '2024-05-30', '2024-10-09', '2025-01-01', '2025-01-25', '2025-02-18'],
    'Olivo': ['2015-04-11', '2015-06-22', '2015-08-09', '2015-09-26', '2016-02-05',
              '2016-08-03', '2016-09-20', '2016-10-08', '2016-10-14', '2016-10-26',
              '2022-05-16', '2022-10-19', '2023-02-16', '2023-12-25', '2024-03-06',
              '2024-04-23', '2024-08-21', '2024-09-02', '2024-09-26', '2024-12-07'],
}

RESERVOIRS = {
    'Arancio': (37.634491, 13.065184),
    'Castello': (37.582494, 13.420304),
    'Olivo': (37.405048, 14.286604),
}


def resubmit(name, lat, lon, dates, dry_run=False):
    lake_poly = get_lake_poly(lat, lon)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    s1_composite = (S1_GRD
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filterBounds(train_clip)
        .filter(ee.Filter.calendarRange(CFG['train_year'], CFG['train_year'], 'year'))
        .select(BANDS).mosaic()
        .focal_mean(30, 'circle', 'meters').clip(train_clip))

    from export_extended_masks import JRC_OCC, WC
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

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(CFG['s1_start'], CFG['s1_end'])
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    best_raw = select_best_orbit(s1_raw, aoi)
    best_col = best_raw.map(lambda img: img.select(BANDS)
        .focal_mean(30, 'circle', 'meters')
        .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                              'relativeOrbitNumber_start', 'angle']))

    if dry_run:
        for date_str in dates:
            d0 = ee.Date(date_str)
            n = best_col.filterDate(d0, d0.advance(1, 'day')).size().getInfo()
            print(f'  {name} {date_str}: {n} image(s) on best orbit')
        return 0

    submitted = 0
    for date_str in dates:
        d0 = ee.Date(date_str)
        img = ee.Image(best_col.filterDate(d0, d0.advance(1, 'day')).first())
        cleaned = classify_image(img, svm, lake_poly)
        mask_out = cleaned.unmask(0).toByte()
        task = ee.batch.Export.image.toDrive(
            image=mask_out,
            description=f'mask_{name}_{date_str}',
            folder=CFG['drive_folder'],
            fileNamePrefix=f'mask_{name}_{date_str}',
            region=lake_poly.buffer(300),
            scale=10, crs='EPSG:32633', maxPixels=1e9,
        )
        task.start()
        submitted += 1
    print(f'  {name}: resubmitted {submitted} tasks')
    return submitted


if __name__ == '__main__':
    dry = '--dry-run' in sys.argv
    total = 0
    for name, (lat, lon) in RESERVOIRS.items():
        total += resubmit(name, lat, lon, FAILED_DATES[name], dry_run=dry)
    if not dry:
        print(f'\nTotal resubmitted: {total}')
        print('Monitor at https://code.earthengine.google.com/tasks')
