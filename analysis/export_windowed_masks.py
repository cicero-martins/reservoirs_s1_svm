"""
export_windowed_masks.py

Re-exports Period-B SAR masks for a reservoir restricted to a specific, short
date WINDOW instead of the full 2022-2026 span, then stratifies ~10 dates by
WATER LEVEL WITHIN that window.

Why: a per-reservoir audit of the actual gauge/SWOT time series (2026-07-21)
found that most Sicilian reservoirs went through a single, sharp, near-total
drawdown-to-refill (or refill) event concentrated in a few months of 2025-2026
(a shared regional hydrological event), while the existing Period-B masks were
percentile-stratified across the full 4-year window and mostly missed it. A
dense set of masks from inside one short, low-noise event gives a materially
better hypsometric fit than a sparse set spread across 4 years (no risk of
between-date sedimentation/vegetation drift, single consistent orbit/season).

Date-selection history (2026-07-21, second pass): the first version of this
script stratified by AREA percentile within the window. A follow-up "steps and
plateaus" artifact in the 3D DEMs traced back to that choice: the area-elevation
curve is strongly non-linear for these valley reservoirs, so area-percentile
sampling oversamples the wide/fast-area-change part and undersamples the flat
part -- precisely where the level-slicing DEM reconstruction assigns one
uniform elevation to the whole ring exposed between two consecutive masks,
producing a giant flat step. Fixed by stratifying on WATER LEVEL instead
(gauge, or SWOT inside a known gauge-bad window -- the same source-priority
logic as schwatke_bathymetry_3d.phase1()), with targets evenly spaced across
the value RANGE (np.linspace over [5th,95th] percentile WL, not percentiles of
the candidate distribution -- percentile-of-distribution targets reproduce
whatever clustering already exists among candidate images, e.g. oversampling
a slow/flat stretch and leaving a fast transition just as under-sampled).
Ancipa was found to be revisit-limited (candidates == dates already used, no
selection can help) and is intentionally excluded from re-stratification.

AOI resolution: HydroLAKES Hylak_id if given, else coordinate-fallback (same
as export_extended_masks.py / reexport_ancipa_pozzillo_masks.py). Orbit: reuse
select_best_orbit (angle-based) unless a KNOWN_ORBIT is supplied (skip the
expensive rescoring once an orbit is already validated -- see
reexport_ancipa_pozzillo_masks.py / project_export_pipeline memory for why).

Run:
    python analysis/export_windowed_masks.py --dry-run
    python analysis/export_windowed_masks.py
"""
import sys, pathlib
import numpy as np
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
import schwatke_bathymetry_3d as m  # gauge/SWOT loaders + interp_wl, no ee dependency

SWOT_DIR = pathlib.Path('validation_data/SWOT')

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

# name -> (lat, lon, hylak_id_or_None, window_start, window_end, known_orbit_or_None)
# Windows from the 2026-07-21 gauge/SWOT audit: local min -> local max of the
# 2025-2026 shared drawdown/refill event (Ancipa uses its earlier, deeper
# Nov-2024/Feb-2025 event instead -- see project memory).
RESERVOIRS = [
    ('Poma',       38.011037, 13.056135, 173610, '2025-12-15', '2026-05-10', None),
    ('Rosamarina', 37.960336, 13.654665, 173633, '2025-09-15', '2026-05-15', None),
    ('Castello',   37.582494, 13.420304, None, '2025-09-15', '2026-04-25', None),
    ('Olivo',      37.405048, 14.286604, None, '2025-09-25', '2026-04-05', None),
    ('Arancio',    37.634491, 13.065184, None, '2025-09-01', '2026-04-15', None),
    ('Ancipa',     37.887,    14.565,    1369046, '2024-11-10', '2025-03-01', 117),
    ('Nicoletti',  37.604822, 14.346314, None, '2025-10-15', '2026-03-20', None),
    ('Pozzillo',   37.674037, 14.610613, 173729, '2025-10-01', '2026-03-25', 117),
    # Garcia: the gauge reads a corrupted, noisy dry-lakebed floor (~176.0-176.5 m)
    # from 2025-08-06 to 2026-02-03 regardless of true level (confirmed against SWOT,
    # which shows a real decline to ~172 m over the same window), then recovers and
    # tracks a real, smooth 176.5->189.8 m refill from 2026-02-04 onward. Span the
    # FULL combined window: SAR-area stratification doesn't need a WL source at
    # export time, only at pairing time (schwatke_bathymetry_3d.py uses SWOT for
    # dates before 2026-02-04, gauge after) -- capturing the deeper SWOT-only levels
    # is the point, not something to avoid.
    ('Garcia',     37.799,    13.119,    None, '2025-08-06', '2026-05-31', None),
]
N_DATES = 10

# Reservoirs whose Period-A reference masks were built from the RAW HydroLAKES
# polygon directly (no JRC max_extent refinement). Found 2026-07-21 while
# re-exporting Pozzillo's WL-restratified dates: get_lake_poly's JRC-refined
# search (needed to fix Poma/Rosamarina's undersized AOI, see module docstring)
# gives Pozzillo a deterministic but DIFFERENT, larger polygon (646.5 ha, extends
# ~1.1 km further west) than its established (297, 543) reference grid, which
# matches raw HydroLAKES (557.7 ha) exactly. There is no universal rule here --
# each reservoir's original reference must be matched empirically, and Pozzillo
# (like Ancipa) turns out to need the unrefined polygon.
NO_JRC_REFINE = {'Pozzillo', 'Ancipa'}


def get_lake_poly(lat, lon, hylak_id=None, jrc_refine=True):
    """Mirrors exportSicilyMasks.js getLakePoly exactly: the Hylak_id polygon
    (when given) is only a SEARCH AREA (buffered 2000m), not the AOI itself --
    the actual AOI is the largest JRC max_extent water polygon inside it. Using
    the raw HydroLAKES geometry directly (as an earlier version of this function
    did) silently shrank Poma/Rosamarina's AOI to ~56% of their true extent,
    producing masks on a grid incompatible with their Period-A reference files.

    jrc_refine=False returns the raw HydroLAKES geometry unrefined -- needed for
    reservoirs in NO_JRC_REFINE (Pozzillo, Ancipa) whose established Period-A
    reference grid matches the raw polygon, not the JRC-refined one (found
    2026-07-21: for Pozzillo the two are NOT equivalent, unlike Poma/Rosamarina)."""
    if hylak_id is not None and not jrc_refine:
        return HYDROLAKES_SIC.filter(ee.Filter.eq('Hylak_id', hylak_id)).first().geometry()
    jrc_max = JRC_GSW.select('max_extent').eq(1).selfMask()
    if hylak_id is not None:
        hydro_geom = HYDROLAKES_SIC.filter(ee.Filter.eq('Hylak_id', hylak_id)).first().geometry()
        search_area = hydro_geom.buffer(2000)
    else:
        search_area = ee.Geometry.Point([lon, lat]).buffer(3000)
    water_vecs = jrc_max.reduceToVectors(
        geometry=search_area, scale=30, maxPixels=1e9, bestEffort=True,
        geometryType='polygon', eightConnected=True, tileScale=4,
    )
    return water_vecs.sort('count', False).first().geometry()


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
    qualified = scored.filter(ee.Filter.gte('n', 10)).filter(ee.Filter.gt('n_covered', 0))
    candidates = ee.FeatureCollection(ee.Algorithms.If(
        qualified.size().gt(0), qualified,
        scored.filter(ee.Filter.gt('n_covered', 0))))
    candidates = ee.FeatureCollection(ee.Algorithms.If(candidates.size().gt(0), candidates, scored))
    best = ee.Feature(candidates.sort('angle_mean', False).first())
    orbit_num = best.getNumber('orbit').getInfo()
    pass_dir = best.getString('pass').getInfo()
    return orbit_num, pass_dir


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


def candidate_dates_of(best_col, batch_size=200):
    """List of acquisition dates (YYYY-MM-DD) in an already orbit-filtered
    collection -- cheap (.aggregate_array), no per-image reduceRegion needed."""
    n = best_col.size().getInfo()
    ts = []
    for start in range(0, n, batch_size):
        batch = ee.ImageCollection(best_col.toList(batch_size, start))
        ts.extend(batch.aggregate_array('system:time_start').getInfo())
    return sorted(set(pd.Timestamp(t, unit='ms').strftime('%Y-%m-%d') for t in ts))


def wl_for_date(cfg, dt, gauge, swot, bad_windows):
    """Same source-priority as schwatke_bathymetry_3d.phase1(): gauge unless a
    known-bad window applies, else SWOT, with a gauge fallback if SWOT is also
    missing for that date."""
    in_bad = any(lo <= dt <= hi for lo, hi in bad_windows)
    if len(gauge) > 0 and not in_bad:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            return val
    if len(swot) > 0:
        val = m.interp_wl(swot, dt, m.MAX_DT)
        if not np.isnan(val):
            return val
    if len(gauge) > 0:
        val = m.interp_wl(gauge, dt, m.MAX_DT)
        if not np.isnan(val):
            return val
    return np.nan


def wl_series_for(name):
    cfg = m.CONFIGS[name]
    try:
        gauge = m.load_gauge(cfg)
    except Exception:
        gauge = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    swot_f = SWOT_DIR / f'{name}_swot.csv'
    swot = m.load_swot(swot_f) if swot_f.exists() else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    gauge_bad = cfg.get('gauge_bad_window')
    bad_windows = []
    if gauge_bad:
        raw = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
        bad_windows = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in raw]
    return gauge, swot, bad_windows


def stratify_dates_by_wl(name, candidate_date_strs, n=N_DATES):
    """Pick n dates evenly spaced in WATER-LEVEL VALUE (linspace over the
    [5th,95th] percentile range, NOT percentiles of the candidate distribution
    -- see module docstring for why that distinction matters)."""
    gauge, swot, bad_windows = wl_series_for(name)
    cand = []
    for d in candidate_date_strs:
        dt = pd.Timestamp(d)
        wl = wl_for_date(m.CONFIGS[name], dt, gauge, swot, bad_windows)
        if not np.isnan(wl):
            cand.append((d, wl))
    if len(cand) == 0:
        return []
    wls = np.array([w for _, w in cand])
    wl_lo, wl_hi = np.percentile(wls, [5, 95])
    targets = np.linspace(wl_lo, wl_hi, min(n, len(cand)))
    used = set(); picked = []
    for target in targets:
        best_d, best_diff = None, np.inf
        for d, w in cand:
            if d in used:
                continue
            diff = abs(w - target)
            if diff < best_diff:
                best_diff, best_d = diff, d
        if best_d is not None:
            used.add(best_d)
            picked.append(best_d)
    return sorted(picked)


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


def process_reservoir(name, lat, lon, hylak_id, win_start, win_end, known_orbit, dry_run=False):
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
        best_raw = s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num)) \
                         .filter(ee.Filter.eq('orbitProperties_pass', pass_dir))
    else:
        orbit_num, pass_dir = select_best_orbit(s1_raw, aoi)
        best_raw = s1_raw.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit_num)) \
                         .filter(ee.Filter.eq('orbitProperties_pass', pass_dir))

    n = best_raw.size().getInfo()
    print(f'{name}: window {win_start}..{win_end}  orbit={orbit_num} ({pass_dir})  n={n}')
    if n == 0:
        print(f'  {name}: NO images in window on any orbit -- skipping'); return 0

    print(f'  {name}: listing candidate dates + stratifying by water level...')
    candidate_strs = candidate_dates_of(best_raw)
    dates = stratify_dates_by_wl(name, candidate_strs)
    print(f'  {name}: {len(candidate_strs)} candidate images, {len(dates)} dates selected: {dates}')

    mask_dir = pathlib.Path('raw_data/GEE_SicilyMasks')
    new_dates = [d for d in dates if not (mask_dir / f'mask_{name}_{d}.tif').exists()]
    print(f'  {name}: {len(dates) - len(new_dates)} already on disk, {len(new_dates)} need export')

    if dry_run:
        return 0
    if len(new_dates) == 0:
        print(f'  {name}: nothing new to export')
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
            description=f'mask_{name}_{date_str}_windowed',
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
    for name, lat, lon, hylak_id, ws, we, orbit in RESERVOIRS:
        total += process_reservoir(name, lat, lon, hylak_id, ws, we, orbit, dry_run=dry)
    if not dry:
        print(f'\nTotal tasks submitted: {total}  ->  Drive/{CFG["drive_folder"]}/')
