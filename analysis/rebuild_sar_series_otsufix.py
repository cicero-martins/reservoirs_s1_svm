"""
rebuild_sar_series_otsufix.py (2026-08-03)

Regenerates the continuous SAR water-area series for the 4 "Fase-3 extended"
reservoirs (Olivo, Nicoletti, Castello, Arancio) over the FULL Sentinel-1 era,
replacing the series produced by analysis/exportSicilyExtended.js.

WHY (root cause, established 2026-08-03 against two user-supplied reference
CSVs exported from the GEE UI app with VV-Otsu explicitly selected):

  gee_reservoir_monitor_app_v226.js -- the published Paper-1 method -- clips
  each scene to aoi = lakePoly.buffer(100) inside preprocessS1() BEFORE
  computeOtsuWater() asks for a histogram over lakePoly.buffer(500). The
  histogram therefore only ever sees the 100 m collar: reservoir-dominated,
  cleanly bimodal, stable scene to scene.

  exportSicilyExtended.js never clips before classifying, so its Otsu
  histogram really does span buffer(500) -- ~5x more land, whose backscatter
  moves with soil moisture and wind. The between-class-variance optimum then
  wanders (measured: -16.8 dB on clean dates vs -12.3 dB on spike dates for
  Olivo), and since water = VV < T, the shoreline moves with it. That single
  difference produces BOTH the systematic area overestimate and the day-to-day
  spikes.

  Measured against the reference CSVs (11 dates, 2023-05..2026-04, same orbit,
  everything else identical):
      no pre-clip, hist @10 m (exportSicilyExtended)  bias +3.02  RMSE 3.73
      pre-clip,    hist @30 m (v226, this script)     bias +1.25  RMSE 1.96
      no pre-clip, hist @30 m                         bias +7.35  RMSE 9.81
  Residual error in the v226 row is expected: the reference also uses an
  angle-bin collection (several relative orbits -- visible as identical
  value-pairs one day apart) plus +-6-day mosaic compositing, both replicated
  here but not in that isolation test.

SCOPE: only these 4 reservoirs' area series use VV-Otsu. The production
watermasks behind every DEM (export_windowed_masks.py) and the core-5 series
(exportGlobalPilotV2.js) both classify with the SVM and are untouched.

Deliberately NOT applied: v226's 4-pass outlier rejection + LOWESS smoothing.
The core-5 series are raw per-scene areas; smoothing 4 of 9 would make the set
non-comparable. Fixing the histogram support removes the instability at its
source instead of masking it.

Faithful port of v226: angle-bin orbit selection (PrioritizeDescendingAngleBins),
preprocessS1 (focal_mean 30 m -> clip aoi), fillCoverageGaps(+-6 d),
computeOtsuWater (hist over buffer(500) @ 30 m, 256 buckets), morphological
gap-fill, centroid-inside-lakePoly cleaning with largest-polygon fallback,
area from WaterCleaned at 10 m.

Run:  python analysis/rebuild_sar_series_otsufix.py [Reservoir]
"""
import argparse
import pathlib
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

OUT_DIR = pathlib.Path('raw_data/exportSicilyExtended/GEE_SicilyExtended_VVotsu')
S1_GRD = ee.ImageCollection('COPERNICUS/S1_GRD')
JRC_MAX = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent').eq(1).selfMask()
BANDS = ['VV', 'VH']

CFG = dict(
    s1_start='2014-10-01', s1_end='2026-08-01',
    clean_scale_m=30, sar_scale_m=10, max_pixels=int(1e9),
    otsu_hist_buffer_m=500, otsu_hist_buckets=256, otsu_hist_scale_m=30,
    composite_window_days=6, coverage_strict_pct=90, min_coverage_pct=50,
    smoothing_radius_m=30, angle_bin_deg=3,
)

# Dam coordinates, identical to exportSicilyExtended.js's PILOT_RESERVOIRS, so
# getLakePoly resolves the same JRC polygon and ap_m stays comparable.
RESERVOIRS = {
    'Olivo':     (37.405048, 14.286604),
    'Nicoletti': (37.604822, 14.346314),
    'Castello':  (37.582494, 13.420304),
    'Arancio':   (37.634491, 13.065184),
}

CHUNK_DATES = 30   # dates per getInfo call; reduceToVectors is the cost driver


def get_lake_poly(lat, lon):
    """Coord fallback from exportSicilyExtended.js/v226: largest JRC max_extent
    polygon within 20 km of the dam point, preferring one within 10 km."""
    pt = ee.Geometry.Point([lon, lat])
    vecs = JRC_MAX.reduceToVectors(
        geometry=pt.buffer(20000), scale=30, maxPixels=int(1e8), bestEffort=True,
        reducer=ee.Reducer.countEvery(), geometryType='polygon',
        eightConnected=True, tileScale=4)
    with_dist = vecs.map(lambda f: f.set('_dist', f.geometry().centroid(1).distance(pt, 1)))
    nearby = with_dist.filter(ee.Filter.lte('_dist', 10000))
    return ee.Geometry(ee.Algorithms.If(
        nearby.size().gt(0),
        nearby.sort('count', False).first().geometry(),
        with_dist.sort('count', False).first().geometry()))


def scene_stats(s1_raw, aoi, aoi_area):
    """Per-scene mean incidence angle + AOI coverage %, the two quantities
    v226's selectBestOrbit ranks on. Reduced at 100 m: the angle band is a
    smooth across-track ramp, so the mean is insensitive to scale, and this
    keeps the whole-archive pass affordable."""
    def stat(img):
        angle = img.select('angle').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100,
            maxPixels=int(1e7), bestEffort=True).get('angle')
        cov = (img.select('VV').mask().multiply(ee.Image.pixelArea())
               .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=100,
                             maxPixels=int(1e7), bestEffort=True)
               .getNumber('VV').divide(aoi_area).multiply(100))
        return ee.Feature(None, {
            'idx': img.get('system:index'),
            'time': img.get('system:time_start'),
            'angle_mean': angle,
            'coverage': cov,   # not 'cov' -- collides with DataFrame.cov()
            'relOrbit': img.get('relativeOrbitNumber_start'),
            'pass': img.get('orbitProperties_pass'),
        })
    return s1_raw.map(stat)


def select_best_orbit(df):
    """Client-side replica of v226's PrioritizeDescendingAngleBins: walk unique
    mean angles high -> low, take the first 3-degree bin holding at least one
    scene with >= 90% AOI coverage, and keep that bin's well-covered scenes.
    Falls back to any scene above min_coverage_pct, then to everything."""
    df = df.dropna(subset=['angle_mean']).copy()
    angles = sorted(df.angle_mean.unique(), reverse=True)
    for cur in angles:
        nxt = cur - CFG['angle_bin_deg']
        binned = df[(df.angle_mean >= nxt) & (df.angle_mean < cur)]
        valid = binned[binned['coverage'] >= CFG['coverage_strict_pct']]
        if len(valid):
            return valid.sort_values('time').reset_index(drop=True), f'angle bin [{nxt:.2f}, {cur:.2f})'
    fb = df[df['coverage'] >= CFG['min_coverage_pct']]
    if len(fb):
        return fb.sort_values('time').reset_index(drop=True), 'fallback: coverage >= 50%'
    return df.sort_values('time').reset_index(drop=True), 'fallback: all scenes'


def preprocess(img, aoi):
    """v226 preprocessS1 -- speckle filter THEN clip to aoi. The clip is what
    later confines the Otsu histogram to the 100 m collar (see module docstring)."""
    return (img.select(BANDS)
            .focal_mean(CFG['smoothing_radius_m'], 'circle', 'meters')
            .clip(aoi)
            .copyProperties(img, ['system:time_start', 'orbitProperties_pass',
                                  'relativeOrbitNumber_start']))


def otsu(histogram):
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    tsum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = tsum.divide(total)

    def bss(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        b_count = total.subtract(a_count)
        b_mean = tsum.subtract(a_count.multiply(a_mean)).divide(b_count)
        return (a_count.multiply(a_mean.subtract(mean).pow(2))
                .add(b_count.multiply(b_mean.subtract(mean).pow(2))))

    vals = ee.List.sequence(1, size).map(bss)
    return means.sort(ee.Array(vals)).get([-1])


def classify_and_area(img, lake_poly, aoi):
    """v226 computeOtsuWater + gap-fill + centroid-inside cleaning + area."""
    hist = img.select('VV').reduceRegion(
        reducer=ee.Reducer.histogram(CFG['otsu_hist_buckets']),
        geometry=lake_poly.buffer(CFG['otsu_hist_buffer_m']),
        scale=CFG['otsu_hist_scale_m'], maxPixels=int(1e8),
        bestEffort=True).get('VV')
    threshold = ee.Number(otsu(hist))
    water = img.select('VV').lt(threshold).rename('Water').clip(aoi)

    mask = water.unmask(0).clip(aoi)
    dist = mask.fastDistanceTransform(30).clip(aoi)
    filled = dist.lte(0.5).updateMask(dist.lte(0.5)).where(mask, 1).rename('WaterFilled')

    polys = filled.reduceToVectors(
        geometryType='polygon', reducer=ee.Reducer.countEvery(),
        scale=CFG['clean_scale_m'], maxPixels=CFG['max_pixels'],
        bestEffort=True, tileScale=4)
    with_flag = polys.map(lambda f: f.set(
        '_inside', lake_poly.contains(f.geometry().centroid(1), ee.ErrorMargin(1))))
    inside = with_flag.filter(ee.Filter.eq('_inside', 1))
    largest = ee.FeatureCollection([polys.map(
        lambda f: f.set('_area', f.geometry().area(1))).sort('_area', False).first()])
    kept = ee.FeatureCollection(ee.Algorithms.If(inside.size().gt(0), inside, largest))

    kept_mask = ee.Image().paint(featureCollection=kept, color=1).rename('KeptRegionMask')
    cleaned = filled.updateMask(kept_mask).rename('WaterCleaned')
    area_m2 = cleaned.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=CFG['sar_scale_m'],
        maxPixels=CFG['max_pixels'], bestEffort=True).get('WaterCleaned')

    merged = kept.union(1).geometry()
    dyn_ap = ee.Number(area_m2).divide(ee.Number(merged.perimeter(1)).max(1))
    return ee.Feature(None, {
        'area_m2': area_m2,
        'area_ha': ee.Number(area_m2).divide(1e4),
        'ap_m_dynamic': dyn_ap,
        'otsu_db': threshold,
    })


def composite_for_date(s1_proc, date_str, window_days):
    """v226 fillCoverageGaps: mosaic every scene within +-window_days of this
    date. Sorted newest-first so mosaic() lays the oldest on top, exactly as
    the JS does."""
    t = ee.Date(date_str)
    window = s1_proc.filterDate(t.advance(-window_days, 'day'),
                                t.advance(window_days + 1, 'day'))
    return window.sort('system:time_start', False).mosaic().set('system:time_start', t.millis())


def rebuild(name):
    lat, lon = RESERVOIRS[name]
    print(f'\n=== {name} ===')
    lake_poly = get_lake_poly(lat, lon)
    aoi = lake_poly.buffer(100)
    aoi_area = aoi.area(1)
    ap_m = float(lake_poly.area(1).divide(lake_poly.perimeter(1)).getInfo())
    print(f'  lakePoly {float(lake_poly.area(1).getInfo())/1e4:.1f} ha   A/P {ap_m:.1f} m')

    s1_raw = (S1_GRD.filterBounds(aoi).filterDate(CFG['s1_start'], CFG['s1_end'])
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.eq('resolution_meters', 10))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    # Scene stats, fetched a year at a time so no single getInfo has to carry
    # the whole archive.
    frames = []
    for yr in range(2014, 2027):
        sub = s1_raw.filterDate(f'{yr}-01-01', f'{yr + 1}-01-01')
        n = sub.size().getInfo()
        if n == 0:
            continue
        feats = scene_stats(sub, aoi, aoi_area).getInfo()['features']
        frames.append(pd.DataFrame([f['properties'] for f in feats]))
        print(f'  {yr}: {n} scenes', end='\r')
    stats_df = pd.concat(frames, ignore_index=True)
    print(f'  {len(stats_df)} scenes total, all orbits            ')

    sel, how = select_best_orbit(stats_df)
    sel['date'] = pd.to_datetime(sel['time'], unit='ms').dt.strftime('%Y-%m-%d')
    orbits = sorted(sel.relOrbit.unique())
    passes = sorted(sel['pass'].unique())
    print(f'  orbit selection: {how} -> {len(sel)} scenes, relOrbit={orbits}, pass={passes}')

    sel_col = s1_raw.filter(ee.Filter.inList('system:index', sel.idx.tolist()))
    s1_proc = sel_col.map(lambda img: preprocess(img, aoi))

    dates = sorted(sel.date.unique())
    print(f'  {len(dates)} distinct dates -> classifying in chunks of {CHUNK_DATES}')

    rows = []
    for i in range(0, len(dates), CHUNK_DATES):
        chunk = dates[i:i + CHUNK_DATES]
        imgs = [composite_for_date(s1_proc, d, CFG['composite_window_days']) for d in chunk]
        col = ee.ImageCollection.fromImages(imgs)
        fc = col.map(lambda img: classify_and_area(img, lake_poly, aoi))
        try:
            feats = fc.getInfo()['features']
        except Exception as e:
            print(f'    chunk {i//CHUNK_DATES + 1} FAILED: {e}')
            continue
        for d, f in zip(chunk, feats):
            p = f['properties']
            if p.get('area_m2') is None or p['area_m2'] <= 0:
                continue
            rows.append({'date': d, 'area_m2': p['area_m2'], 'area_ha': p['area_ha'],
                         'ap_m': ap_m, 'ap_m_dynamic': p.get('ap_m_dynamic'),
                         'otsu_db': p.get('otsu_db')})
        print(f'    {min(i + CHUNK_DATES, len(dates))}/{len(dates)} dates', end='\r')

    out = pd.DataFrame(rows)
    meta = sel.drop_duplicates('date').set_index('date')
    out['relOrbit'] = out.date.map(meta.relOrbit)
    out['passDirection'] = out.date.map(meta['pass'])
    out = out[['date', 'area_m2', 'area_ha', 'relOrbit', 'passDirection',
               'ap_m', 'ap_m_dynamic', 'otsu_db']]
    fp = OUT_DIR / f'SAR_area_{name}.csv'
    out.to_csv(fp, index=False)
    print(f'  wrote {len(out)} rows -> {fp}                 ')
    print(f'  area {out.area_ha.min():.1f}-{out.area_ha.max():.1f} ha, '
          f'otsu {out.otsu_db.min():.2f}..{out.otsu_db.max():.2f} dB')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reservoir', nargs='?', default=None)
    args = ap.parse_args()
    for n in ([args.reservoir] if args.reservoir else list(RESERVOIRS)):
        rebuild(n)
