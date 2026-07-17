"""
export_extended_masks.py

Exports binary water-mask GeoTIFFs for the 3 extended Sicilian reservoirs
(Arancio, Castello, Olivo). Dates are stratified by area percentile (5th-95th,
10 per period x 2 periods) FROM WITHIN the specific best-orbit collection this
script locks onto for SVM masking (see restratify_to_orbit_dates) -- an earlier
version stratified from exportSicilyExtended.js's area series instead, which
uses a different orbit-selection/gap-filling scheme, causing 56/60 export tasks
to fail with "Image.select: input... may not be null" (no image on the chosen
orbit for ~most of those dates).

Python port of analysis/exportSicilyMasks.js — IDENTICAL classification pipeline
(same SVM training, same pixel-coverage orbit selection, same polygon-cleaning)
so the new reservoirs' masks are methodologically consistent with the 5 core
reservoirs'. Submitted headlessly via the Earth Engine Python API (no manual
Code Editor step).

Output: GeoTIFF per date in Google Drive folder 'GEE_SicilyMasks' (same folder as
the 5 core reservoirs — filenames are namespaced by reservoir name already).
  CRS: EPSG:32633, scale 10 m, file naming mask_{Reservoir}_{YYYY-MM-DD}

Run:
    python analysis/export_extended_masks.py            # submit all tasks
    python analysis/export_extended_masks.py --dry-run   # resolve AOIs/orbits only, no export
"""
import sys

try:
    import truststore
    truststore.inject_into_ssl()   # UniPa corporate TLS MITM breaks bundled CA stores
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
)
BANDS = ['VV', 'VH']
S1_GRD = ee.ImageCollection('COPERNICUS/S1_GRD')
JRC_GSW = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
JRC_OCC = JRC_GSW.select('occurrence')
WC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')

# [name, lat, lon] -- dam coordinates from sicilia_dighe_anagrafica.csv, same as
# exportSicilyExtended.js (coord-fallback: not in the HydroLAKES_sicily4 asset).
RESERVOIRS = [
    ('Arancio', 37.634491, 13.065184),
    ('Castello', 37.582494, 13.420304),
    ('Olivo', 37.405048, 14.286604),
]


def get_lake_poly(lat, lon):
    jrc_max = JRC_GSW.select('max_extent').eq(1).selfMask()
    search_area = ee.Geometry.Point([lon, lat]).buffer(3000)
    water_vecs = jrc_max.reduceToVectors(
        geometry=search_area, scale=30, maxPixels=1e9, bestEffort=True,
        geometryType='polygon', eightConnected=True, tileScale=4,
    )
    return water_vecs.sort('count', False).first().geometry()


def select_best_orbit(col, aoi):
    """Rank orbits by mean INCIDENCE ANGLE (not pixel count) -- the validated fix from
    exportGlobalPilotV2.js/app v226 (see project_export_pipeline memory): a
    pixel-coverage-based choice can select an orbit prone to episodic wind-driven
    misclassification (near-range/low-angle backscatter is less specular, weaker
    water/land contrast), concretely observed at Harlan County (spurious 5000ha ->
    8-2300ha dips on the pixel-count orbit; stable on the angle-selected one). Must
    run on RAW imagery (keeps the 'angle' band) -- caller applies focal_mean/band
    selection AFTER this, not before, so orbit selection isn't biased by preprocessing."""
    coverage_strict_pct = 90
    min_coverage_pct = 50

    def with_stats(img):
        mean_angle = img.select('angle').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100,
            maxPixels=1e7, bestEffort=True).getNumber('angle')
        pct = (img.select('VV').mask().multiply(ee.Image.pixelArea())
               .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi,
                             scale=100, maxPixels=1e7, bestEffort=True)
               .getNumber('VV').divide(aoi.area(1)).multiply(100))
        return img.set({
            '_relOrbit': img.getNumber('relativeOrbitNumber_start'),
            '_pass': img.getString('orbitProperties_pass'),
            'angle_mean': mean_angle,
            'percentCovered': pct,
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


def restratify_to_orbit_dates(name, orbit_dates_str):
    """The originally selected dates came from exportSicilyExtended.js's area series,
    which uses a DIFFERENT orbit-selection/gap-filling scheme than this script's
    single-best-orbit SVM export -- many of those dates have no image on the specific
    orbit this script locks onto (cause of the 56/60 'Image.select null' failures).
    Re-run the same percentile stratification restricted to dates the chosen orbit
    actually has."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(
        f'raw_data/exportSicilyExtended/GEE_SicilyExtended_VVotsu/SAR_area_{name}.csv',
        parse_dates=['date'])
    df = df[['date', 'area_ha']].dropna().sort_values('date')
    avail = set(orbit_dates_str)
    df = df[df['date'].dt.strftime('%Y-%m-%d').isin(avail)]
    periods = {'A': ('2014-10-01', '2016-12-31'), 'B': ('2022-01-01', '2026-06-30')}
    picked = []
    for start, end in periods.values():
        sub = df[(df['date'] >= start) & (df['date'] <= end)]
        if len(sub) == 0:
            continue
        pcts = np.linspace(5, 95, min(10, len(sub)))
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


def process_reservoir(name, lat, lon, dry_run=False):
    lake_poly = get_lake_poly(lat, lon)
    aoi = lake_poly.buffer(100)
    train_clip = aoi.buffer(CFG['land_ring_outer_m'])

    area_m2 = lake_poly.area(1)
    perim_m = lake_poly.perimeter(1)
    ap_m = ee.Number(area_m2).divide(perim_m)
    print(f'{name}: resolved AOI, A/P = {ap_m.getInfo():.1f} m, area = '
          f'{ee.Number(area_m2).divide(1e4).getInfo():.1f} ha')

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

    # Orbit selection runs on RAW imagery (angle band intact); preprocessing
    # (focal_mean + drop to VV/VH) is applied AFTER, to the already-orbit-filtered
    # collection -- order matters (see select_best_orbit docstring).
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

    orbit_dates_str = sorted(set(
        ee.Date(t).format('YYYY-MM-dd').getInfo()
        for t in best_col.aggregate_array('system:time_start').getInfo()
    ))
    dates = restratify_to_orbit_dates(name, orbit_dates_str)

    if dry_run:
        print(f'  {name}: dry-run OK, {len(orbit_dates_str)} images on best orbit, '
              f'{len(dates)} dates re-stratified and queued')
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
    print(f'  {name}: submitted {submitted} export tasks')
    return submitted


if __name__ == '__main__':
    dry = '--dry-run' in sys.argv
    total = 0
    for name, lat, lon in RESERVOIRS:
        total += process_reservoir(name, lat, lon, dry_run=dry)
    if not dry:
        print(f'\nTotal tasks submitted: {total}  ->  Drive/{CFG["drive_folder"]}/')
        print('Monitor at https://code.earthengine.google.com/tasks')
