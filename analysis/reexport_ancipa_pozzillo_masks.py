"""
reexport_ancipa_pozzillo_masks.py

Re-exports Ancipa and Pozzillo mask GeoTIFFs on the validated, angle-based best
orbit (117 for both -- confirmed higher mean incidence angle AND more images than
the pixel-count-selected orbit 124 originally used by exportSicilyMasks.js; see
project_export_pipeline memory for why angle beats pixel-count: near-range/low-
angle backscatter is less specular, weaker water/land contrast, more prone to
wind-driven misclassification dips -- empirically 73/564 (13%) suspected dips for
Ancipa and 41/564 (7%) for Pozzillo on the old orbit, vs 1/564 (0.2%) for
Rosamarina, whose pixel-count-chosen orbit happened to already have the higher
angle).

Because there is no pre-existing area series on orbit 117 to stratify dates from,
this script:
  1. Resolves the AOI via the HydroLAKES_sicily4 asset (Hylak_id), matching
     exportSicilyMasks.js's getLakePoly for non-null hylak_id.
  2. Selects the best orbit by incidence angle (see export_extended_masks.py's
     select_best_orbit, reused here).
  3. Computes a QUICK Otsu-threshold area series on that orbit (cheap, only used
     to rank percentile dates -- not the final classification).
  4. Percentile-stratifies 10 dates per period (A: 2014-2016, B: 2022-2026) from
     that quick series.
  5. Trains the SAME dual-pol SVM as exportSicilyMasks.js and exports final masks
     for the selected dates -- methodologically identical to the original 5-
     reservoir pipeline, only the orbit differs.

Output: GeoTIFF per date in Google Drive folder 'GEE_SicilyMasks' (same folder;
existing mask_Ancipa_*/mask_Pozzillo_*.tif on orbit 124 are left in place locally
-- the new orbit-117 files use the same date-based naming, so the caller should
replace the local raw_data/GEE_SicilyMasks/ copies with the new downloads once
verified, not merge the two orbits).

Run:
    python analysis/reexport_ancipa_pozzillo_masks.py --dry-run
    python analysis/reexport_ancipa_pozzillo_masks.py
"""
import sys

import numpy as np

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

CFG = dict(
    s1_start='2014-10-01', s1_end='2026-06-30',
    jrc_occ_thresh=95, jrc_occ_fallback=80,
    train_year=2023, clean_scale_m=30, max_pixels=1e9,
    land_ring_inner_m=500, land_ring_outer_m=2000,
    drive_folder='GEE_SicilyMasks',
    otsu_hist_buffer_m=500, otsu_hist_buckets=256,
)
BANDS = ['VV', 'VH']
S1_GRD = ee.ImageCollection('COPERNICUS/S1_GRD')
JRC_GSW = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
JRC_OCC = JRC_GSW.select('occurrence')
WC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
HYDROLAKES_SIC = ee.FeatureCollection('projects/ee-ciceromartinsjr/assets/HydroLAKES_sicily4')

# [name, hylak_id]
RESERVOIRS = [('Ancipa', 1369046), ('Pozzillo', 173729)]
PERIODS = {'A': ('2014-10-01', '2016-12-31'), 'B': ('2022-01-01', '2026-06-30')}
N_PER_PERIOD = 10


def get_lake_poly(hylak_id):
    return HYDROLAKES_SIC.filter(ee.Filter.eq('Hylak_id', hylak_id)).first().geometry()


def select_best_orbit(col, aoi):
    """Angle-based orbit selection -- identical logic to export_extended_masks.py."""
    coverage_strict_pct = 90
    min_coverage_pct = 50

    def with_stats(img):
        mean_angle = img.select('angle').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100,
            maxPixels=1e7, bestEffort=True, tileScale=4).getNumber('angle')
        pct = (img.select('VV').mask().multiply(ee.Image.pixelArea())
               .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi,
                             scale=100, maxPixels=1e7, bestEffort=True, tileScale=4)
               .getNumber('VV').divide(aoi.area(1)).multiply(100))
        return img.set({
            '_relOrbit': img.getNumber('relativeOrbitNumber_start'),
            '_pass': img.getString('orbitProperties_pass'),
            'angle_mean': mean_angle, 'percentCovered': pct,
        })

    with_stats_col = col.map(with_stats).filter(ee.Filter.notNull(['angle_mean']))
    orbits = with_stats_col.aggregate_array('_relOrbit').distinct()

    def score(o):
        sub = with_stats_col.filter(ee.Filter.eq('_relOrbit', o))
        covered = sub.filter(ee.Filter.gte('percentCovered', coverage_strict_pct))
        return ee.Feature(None, {
            'orbit': o, 'pass': ee.String(sub.first().get('_pass')),
            'angle_mean': sub.aggregate_mean('angle_mean'),
            'n': sub.size(), 'n_covered': covered.size(),
        })

    scored = ee.FeatureCollection(orbits.map(score))
    qualified = scored.filter(ee.Filter.gte('n', 20)).filter(ee.Filter.gt('n_covered', 0))
    candidates = ee.FeatureCollection(ee.Algorithms.If(
        qualified.size().gt(0), qualified,
        scored.filter(ee.Filter.gt('n_covered', 0))))
    candidates = ee.FeatureCollection(ee.Algorithms.If(candidates.size().gt(0), candidates, scored))
    best = ee.Feature(candidates.sort('angle_mean', False).first())

    best_col = with_stats_col.filter(ee.Filter.And(
        ee.Filter.eq('_relOrbit', best.getNumber('orbit')),
        ee.Filter.eq('_pass', best.getString('pass')),
    ))
    strict = best_col.filter(ee.Filter.gte('percentCovered', coverage_strict_pct))
    return ee.ImageCollection(ee.Algorithms.If(
        strict.size().gt(0), strict,
        best_col.filter(ee.Filter.gte('percentCovered', min_coverage_pct))))


def otsu(histogram):
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    total_sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = total_sum.divide(total)
    indices = ee.List.sequence(1, size)

    def bss(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        b_count = total.subtract(a_count)
        b_mean = total_sum.subtract(a_count.multiply(a_mean)).divide(b_count)
        return a_count.multiply(a_mean.subtract(mean).pow(2)).add(
            b_count.multiply(b_mean.subtract(mean).pow(2)))

    bss_vals = indices.map(bss)
    return means.sort(ee.Array(bss_vals)).get([-1])


def quick_otsu_area_series(best_col, lake_poly, scale=30, batch_size=60):
    """Cheap VV-Otsu area estimate per image, ONLY for percentile date-selection --
    coarser scale (default 30 m, vs the 10 m final export) AND batched .getInfo()
    calls (default 60 images/request). Pozzillo -- the largest core reservoir --
    hit 'User memory limit exceeded' materializing all ~570 images' Otsu results in
    ONE request, even at scale=30; the per-request compute graph (histogram + otsu
    + area, chained per image via .map()) is what exceeds the limit, not per-image
    pixel count, so batching the client-side .getInfo() round-trips (not just
    coarsening resolution) is the actual fix. Ancipa (n=571) happened to fit in one
    request; Pozzillo did not -- batching is safe/cheap for both either way."""
    def area_feat(img):
        hist = img.select('VV').reduceRegion(
            reducer=ee.Reducer.histogram(CFG['otsu_hist_buckets']),
            geometry=lake_poly.buffer(CFG['otsu_hist_buffer_m']), scale=scale,
            maxPixels=1e9, bestEffort=True, tileScale=8).get('VV')
        threshold = ee.Number(otsu(hist))
        water = img.select('VV').lt(threshold).rename('Water').clip(lake_poly.buffer(100))
        area_m2 = water.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=lake_poly.buffer(100),
            scale=scale, maxPixels=1e8, bestEffort=True, tileScale=8).get('Water')
        return ee.Feature(None, {
            'date': img.date().format('YYYY-MM-dd'),
            'area_ha': ee.Number(area_m2).divide(1e4),
        })

    raw_list = best_col.toList(100000)          # cheap: no per-image computation yet
    n = raw_list.size().getInfo()
    results = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = ee.ImageCollection(ee.List(raw_list.slice(start, end)))
        batch_fc = (batch_imgs.map(area_feat)
                    .filter(ee.Filter.notNull(['area_ha'])))
        info = batch_fc.reduceColumns(ee.Reducer.toList(2), ['date', 'area_ha']).get('list').getInfo()
        results += [(r[0], r[1]) for r in info]
        print(f'    ...batch {start}-{end}/{n} OK')
    return results


def stratify_dates(series):
    import pandas as pd
    df = pd.DataFrame(series, columns=['date', 'area_ha'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    picked = []
    for start, end in PERIODS.values():
        sub = df[(df['date'] >= start) & (df['date'] <= end)]
        if len(sub) == 0:
            continue
        pcts = np.linspace(5, 95, min(N_PER_PERIOD, len(sub)))
        targets = np.percentile(sub['area_ha'], pcts)
        used = set()
        for target in targets:
            residual = (sub['area_ha'] - target).abs().copy()
            for u in used:
                residual[sub['date'] == u] = np.inf
            idx = residual.idxmin()
            if residual[idx] == np.inf:
                continue
            d = sub.loc[idx, 'date']
            used.add(d)
            picked.append(d.strftime('%Y-%m-%d'))
    return sorted(set(picked))


def classify_image(img, svm, lake_poly):
    aoi = lake_poly.buffer(100)
    water = img.select(BANDS).classify(svm).eq(1).clip(aoi).rename('Water')
    mask = water.unmask(0).clip(aoi)
    dist = mask.fastDistanceTransform(30).clip(aoi)
    filled = dist.lte(0.5).updateMask(dist.lte(0.5)).where(mask, 1).rename('WaterFilled')

    polys = filled.reduceToVectors(
        geometryType='polygon', reducer=ee.Reducer.countEvery(),
        scale=CFG['clean_scale_m'], maxPixels=CFG['max_pixels'],
        bestEffort=True, tileScale=4,
    )
    intersecting = polys.filterBounds(lake_poly)

    def largest(fc):
        return ee.FeatureCollection([fc.map(
            lambda f: f.set('_area', f.geometry().area(maxError=1))
        ).sort('_area', False).first()])

    kept_polys = ee.FeatureCollection(ee.Algorithms.If(
        intersecting.size().gt(0), largest(intersecting), largest(polys)))
    kept_mask = ee.Image(0).paint(featureCollection=kept_polys, color=1)
    return filled.updateMask(kept_mask).rename('WaterCleaned')


# Both reservoirs' best orbit was already independently confirmed (angle-based
# select_best_orbit, run separately for each): Ancipa n=571, Pozzillo n=572, both
# orbit 117 / ASCENDING. Filtering directly to it (a plain, cheap ee.Filter.eq, no
# reduceRegion) instead of re-running select_best_orbit skips the expensive
# per-image angle/coverage scoring entirely -- which is what was hitting "User
# memory limit exceeded" for Pozzillo (AOI ~706 ha vs Ancipa's ~126 ha, over the
# same ~1708-image raw collection).
CONFIRMED_ORBIT = 117
CONFIRMED_PASS = 'ASCENDING'


def process_reservoir(name, hylak_id, dry_run=False):
    lake_poly = get_lake_poly(hylak_id)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(CFG['s1_start'], CFG['s1_end'])
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    best_raw = s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', CONFIRMED_ORBIT)) \
                     .filter(ee.Filter.eq('orbitProperties_pass', CONFIRMED_PASS))
    diag = {'n': best_raw.size().getInfo(), 'orbit': CONFIRMED_ORBIT}
    print(f"{name}: best orbit = {diag['orbit']}  n={diag['n']}")

    print(f'  {name}: computing quick Otsu area series for date stratification...')
    series = quick_otsu_area_series(best_raw, lake_poly)
    dates = stratify_dates(series)
    print(f'  {name}: {len(series)} images scanned, {len(dates)} dates selected')

    if dry_run:
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
    for date_str in dates:
        d0 = ee.Date(date_str)
        img = ee.Image(best_col.filterDate(d0, d0.advance(1, 'day')).first())
        cleaned = classify_image(img, svm, lake_poly)
        mask_out = cleaned.unmask(0).toByte()
        task = ee.batch.Export.image.toDrive(
            image=mask_out,
            description=f'mask_{name}_{date_str}_orbit117',
            folder=CFG['drive_folder'],
            fileNamePrefix=f'mask_{name}_{date_str}',
            region=lake_poly.buffer(300),
            scale=10, crs='EPSG:32633', maxPixels=1e9,
        )
        task.start()
        submitted += 1
    print(f"  {name}: submitted {submitted} export tasks (orbit {diag['orbit']})")
    return submitted


if __name__ == '__main__':
    dry = '--dry-run' in sys.argv
    total = 0
    for name, hylak_id in RESERVOIRS:
        total += process_reservoir(name, hylak_id, dry_run=dry)
    if not dry:
        print(f'\nTotal tasks submitted: {total}  ->  Drive/{CFG["drive_folder"]}/')
