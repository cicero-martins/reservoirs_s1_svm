"""
tool/bathymetry.py — data layer for the Paper-2 bathymetry explorer (Streamlit MVP).

Loads the already-reconstructed Period-A/B satellite DEMs and the reference curves
(design + updated survey) for all 9 Sicilian reservoirs, and derives AEV
curves, capacity-change numbers and A-vs-B change maps. Pure functions, no UI —
imported by tool/app.py. Reuses the same logic as analysis/consolidate_bathymetry.py
so the tool and the paper report identical numbers.
"""

import pathlib, glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.signal import savgol_filter

REPO = pathlib.Path(__file__).resolve().parent.parent
DEM_DIR      = REPO / 'analysis' / 'schwatke_output'
PLANET_DIR   = DEM_DIR / 'planet'
TERRAIN_DIR  = DEM_DIR / 'terrain'
CURVE_BUNDLE = REPO / 'tool' / 'data' / 'curves'                                    # bundled (deploy)
CURVE_EXT    = pathlib.Path('C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi')  # local fallback
NEWCURVE_EXT = pathlib.Path('C:/Users/Unipa/Documents/GEE/Data/NewCurves')
UPDATED   = REPO / 'validation_data' / 'updated_curves'
PIXEL_HA  = 0.01

# Fase-3 extended reservoirs: each official survey curve carries its own volume
# column directly -- (glob pattern, sheet, [(quota_col, vol_col), ...]) per block.
# Identical spec to consolidate_bathymetry.py::EXT_CURVE_SPEC (kept in sync).
# Blocks are (quota_col, vol_col, area_col) triples -- the source spreadsheets
# DO carry a direct area column next to volume (found 2026-07-31; the previous
# 2-tuple spec silently dropped it, leaving area_interp=None for all 4 of
# these reservoirs even though real area data was sitting right there).
# Castello's sheet ('Quota_V_S') has one complete table at columns (5,6,7)
# (3001 rows, uniform 0.01 m steps, 267.2-297.2 m, no gaps/duplicates -- fully
# spans its ~277-297 m operating range) plus ~56 small, separate few-row
# snippets elsewhere on the same sheet of unclear provenance; only the one
# complete block is used, matching what the original (pre-area-fix) spec
# already selected.
EXT_CURVE_SPEC = {
    'arancio_2022':      ('ARANCIO*', 'BASE', [(0, 1, 2)]),
    'castello_updated':  ('CASTELLO*', 'Quota_V_S', [(5, 6, 7)]),
    'nicoletti_updated': ('NICOLETTI*', 'Dati Aree-Volumi', [(0, 1, 2)]),
    'olivo_2021':        ('OLIVO*', 'Tabella centimetrica 2021',
                          [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15)]),
}

def _curve_xls(name):
    b = CURVE_BUNDLE / f'{name}.xls'
    return b if b.exists() else CURVE_EXT / f'{name}.xls'

# design=(quota_col, area_col, vol_col, area_unit); ap from JRC max_extent polygon.
RESERVOIRS = {
    'Ancipa':     dict(design=(2, 3, 4, 'km2'), ap=90.5,  updated=None,             notes='Nebrodi; narrow, low A/P → least reliable'),
    'Garcia':     dict(design=(2, 3, 4, 'km2'), ap=167.7, updated='garcia_survey',  notes='Dec-2025 echosounder field survey'),
    'Rosamarina': dict(design=(2, 3, 5, 'ha'),  ap=187.4, updated='rosamarina_2025',notes='2025 official bathymetric survey'),
    'Poma':       dict(design=(2, 4, 5, 'ha'),  ap=190.1, updated='poma_new',       notes='updated centimetric survey curve'),
    'Pozzillo':   dict(design=(2, 4, 5, 'ha'),  ap=240.5, updated=None,             notes='largest; V→h historically unreliable'),
    # Fase-3 extended set: same design-curve source as consolidate_bathymetry.py,
    # 'updated' curves carry their own volume column directly (NewCurves/ files).
    'Arancio':    dict(design=(2, 3, 4, 'km2'), ap=182.2, updated='arancio_2022',   notes='updated official survey curve'),
    'Castello':   dict(design=(2, 3, 4, 'km2'), ap=126.7, updated='castello_updated', notes='updated official survey curve'),
    'Olivo':      dict(design=(2, 4, 5, 'ha'),  ap=50.7,  updated='olivo_2021',     notes='narrowest basin; updated survey curve'),
    'Nicoletti':  dict(design=(2, 4, 5, 'ha'),  ap=119.7, updated='nicoletti_updated', notes='updated official survey curve'),
}


# ── DEM ────────────────────────────────────────────────────────────────────────
def dem_file(name, period):
    if period == 'Planet':
        return PLANET_DIR / f'dem_{name}_Planet.tif'
    return DEM_DIR / f'dem_{name}_{period}.tif'

def has_period(name, period):
    return dem_file(name, period).exists()


def load_dem(name, period):
    """Return dict(arr, transform, bounds, floor, top, mask) or None if absent."""
    fp = dem_file(name, period)
    if not fp.exists():
        return None
    with rasterio.open(fp) as s:
        arr = s.read(1).astype(np.float64)
        tf, bounds = s.transform, s.bounds
    mask = np.isfinite(arr)
    if not mask.any():
        return None
    return dict(arr=arr, transform=tf, bounds=bounds, mask=mask,
                pixel_ha=abs(tf.a) * abs(tf.e) / 1e4,
                floor=float(arr[mask].min()), top=float(arr[mask].max()))


# ── Topo-bathymetry: reconstructed basin merged into the real surrounding terrain ──
def has_terrain(name):
    return (TERRAIN_DIR / f'terrain_{name}.tif').exists()


def topobathy(name, period):
    """Merge the reconstructed bathymetry into the real GLO-30 surrounding terrain for
    the 3D view: our bathymetry below the max shoreline, real terrain above it. The two
    are joined at the shoreline by a single vertical offset (terrain median at the rim
    aligned to the reconstructed max water level), so no geoid/datum conversion is
    needed. The period DEM is reprojected onto the terrain grid, so it works for the
    10 m SAR periods and the 3 m PlanetScope one alike. Returns dict(arr, bounds, maxwl,
    floor) on the buffered terrain grid, or None if the terrain tile is missing.
    Display-only — never used for AEV or the download."""
    d = load_dem(name, period)
    tfp = TERRAIN_DIR / f'terrain_{name}.tif'
    if d is None or not tfp.exists():
        return None
    with rasterio.open(tfp) as s:
        T = s.read(1).astype(np.float64)
        Ttf, Tbounds = s.transform, s.bounds
    if not np.isfinite(T).all():                       # gap-free terrain (no ragged edge)
        fin = np.isfinite(T); _, idx = distance_transform_edt(~fin, return_indices=True)
        T = T[tuple(idx)]
    # reproject the DEM onto the terrain grid (handles SAR 10 m and Planet 3 m)
    Dg = np.full(T.shape, np.nan)
    reproject(d['arr'], Dg, src_transform=d['transform'], src_crs='EPSG:32633',
              dst_transform=Ttf, dst_crs='EPSG:32633',
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    finD = np.isfinite(Dg)
    if finD.sum() < 20:
        return None                                    # DEM outside the terrain buffer
    maxwl = float(np.nanmax(Dg))
    rim = finD & ~binary_erosion(finD)
    offset = float(np.nanmedian(T[rim])) - maxwl       # align terrain to the shoreline
    Ta = np.maximum(T - offset, maxwl)                 # terrain sits at/above the max shoreline
    merged = np.where(finD, Dg, Ta)
    return dict(arr=merged, bounds=Tbounds, maxwl=maxwl, floor=float(np.nanmin(Dg)))


def vertical_range(name):
    """Common elevation window across the reconstructions (A/B/Planet) so the 3D views
    share one z-scale and are directly comparable, plus the terrain top. Returns
    (basin_lo, basin_hi, terrain_hi) or None."""
    los, his = [], []
    for p in ('A', 'B', 'Planet'):
        d = load_dem(name, p)
        if d is not None:
            los.append(d['floor']); his.append(d['top'])
    if not los:
        return None
    lo, hi = min(los), max(his)
    terr_hi = hi
    tfp = TERRAIN_DIR / f'terrain_{name}.tif'
    if tfp.exists():
        with rasterio.open(tfp) as s:
            terr_hi = float(np.nanmax(s.read(1)))
    return lo, hi, terr_hi


def vol_exact(arr, mask, floor, top, pixel_ha=PIXEL_HA):
    """Volume above floor (Mm3), exact per-pixel water-column sum. aev()'s 0.5 m-
    step trapezoidal integration underestimates volume by 1.7-11.7% for these
    sparse (~10-mask) level-slice DEMs, worse for the steppiest ones (found
    2026-07-24) -- use this for any reported total-volume scalar; aev()'s curve
    stays fine for plotting the intermediate area/volume-vs-elevation shape."""
    col = np.clip(top - arr, 0.0, top - floor)
    return float(np.sum(col[mask]) * pixel_ha * 1e4 / 1e6)


def aev(arr, mask, levels, pixel_ha=PIXEL_HA):
    """area(h)=pixels below h (ha); volume above first level via trapezoid (Mm3)."""
    areas = np.array([np.sum((arr < h) & mask) * pixel_ha for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols


# ── Reference curves ───────────────────────────────────────────────────────────
def design_curve(name):
    """Return (area_ha_interp, vol_Mm3_interp) or None if the external file is missing."""
    cfg = RESERVOIRS[name]
    qc, ac, vc, unit = cfg['design']
    fp = _curve_xls(name)
    if not fp.exists():
        return None
    df = pd.read_excel(fp, sheet_name=0, header=None, engine='xlrd')[[qc, ac, vc]]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df.columns = ['quota', 'area', 'vol']
    df = df[df.quota > 80].sort_values('quota').reset_index(drop=True)
    area_ha = df.area * (100.0 if unit == 'km2' else 1.0)
    return (interp1d(df.quota, area_ha, bounds_error=False, fill_value='extrapolate'),
            interp1d(df.quota, df.vol, bounds_error=False, fill_value='extrapolate'))


def deepzone_split(name, period='B'):
    """Split the design curve's total volume into the SAR-observable band [floor, top]
    and the deep zone below the floor, using the design curve's own low-elevation
    branch as the deep-zone anchor (every core reservoir's curve is tabulated down to
    at or near zero volume, i.e. its own lowest surveyed point is effectively the
    pre-impoundment valley floor -- no separate terrain source is needed). Returns
    dict(floor, deep_min, band_pct, deepzone_pct) or None if the design curve or DEM
    is unavailable. Estimate, not a measurement: only the band is SAR-observed."""
    dc = design_curve(name)
    d = load_dem(name, period)
    if dc is None or d is None:
        return None
    _, vol_i = dc
    v_floor = float(vol_i(d['floor']))
    v_top = float(vol_i(d['top']))
    if v_top <= 0:
        return None
    band_pct = 100 * (v_top - v_floor) / v_top
    return dict(floor=d['floor'], deep_min=float(vol_i.x.min()),
                band_pct=band_pct, deepzone_pct=100 - band_pct)


def updated_curve(name):
    """Return (area_ha_interp_or_None, vol_Mm3_interp) for the updated survey, or None."""
    kind = RESERVOIRS[name]['updated']
    if kind == 'poma_new':
        hits = sorted(glob.glob(str(CURVE_BUNDLE / 'POMA*.XLS'))) or \
               sorted(glob.glob(str(NEWCURVE_EXT / 'POMA*.XLS')), key=len)
        if not hits:
            return None
        u = pd.read_excel(hits[0], sheet_name='foglio1', header=None, engine='xlrd')[[0, 1]]
        u.columns = ['quota', 'vol_m3']
        u = u.apply(pd.to_numeric, errors='coerce').dropna().sort_values('quota').reset_index(drop=True)
        # POMA_new.XLS has no area column at all (verified 2026-07-31 -- neither
        # 'foglio1' nor 'Tabella centimetrica' carries one), so area must come
        # from differentiating volume. A genuine data-entry glitch at
        # 177.00-177.99 m (100 consecutive rows frozen at exactly 15,750,000 m3,
        # then an instant +1,750,000 m3 jump at 178.00 m -- a spreadsheet
        # fill-down error, not a real flat-then-cliff reservoir shape) turns
        # into a spurious 0-then-450 ha spike in np.gradient's area(h) right at
        # that point; repair it by linearly redistributing the jump across the
        # frozen span before differentiating (the only such >=20-step flat run
        # in the whole 2886-row table, confirmed 2026-07-31). The rest of the
        # table is genuinely tabulated at a coarse, piecewise-near-constant
        # area resolution (not further rounding noise), so a light smoothing
        # pass only takes the edge off the resulting staircase's sharp corners
        # rather than fabricating false precision.
        vol = u.vol_m3.values.astype(float).copy()
        q = u.quota.values
        d = np.diff(vol)
        i = 0
        while i < len(d):
            if d[i] == 0:
                j = i
                while j < len(d) and d[j] == 0:
                    j += 1
                # d[i]..d[j-1] == 0 means vol is frozen across indices [i, j];
                # the jump happens at d[j] (vol[j] -> vol[j+1]), so the repaired
                # ramp must run through j+1 (the point AFTER the jump), not j
                # (still the frozen value) -- interpolating to vol[j] would
                # leave the frozen run untouched and the cliff exactly where it was.
                if j - i >= 20 and j + 1 < len(vol):
                    vol[i:j + 2] = np.linspace(vol[i], vol[j + 1], j - i + 2)
                i = j
            else:
                i += 1
        # A light window (41, ~0.4 m) barely touched the underlying tabulation's
        # own ~2 m-wide steps -- still visibly a staircase. The steps are a real
        # tabulation-resolution artifact (no independent area measurement backs
        # each 1 cm row), not a genuine sub-metre feature, so a much wider window
        # (601, ~6 m) is the right scale to average them into a smooth, still
        # monotonically-increasing curve without distorting the real large-scale
        # shape (checked against 1001: both agree closely, 601 chosen as the
        # tighter of the two well-behaved options).
        vol_smooth = savgol_filter(vol, window_length=601, polyorder=3)
        area = np.gradient(vol_smooth, q) / 1e4
        return (interp1d(q, area, bounds_error=False, fill_value='extrapolate'),
                interp1d(q, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    if kind == 'rosamarina_2025':
        fp = UPDATED / 'rosamarina_2025.csv'
        if not fp.exists():
            return None
        u = pd.read_csv(fp)
        return (interp1d(u.quota_m, u.area_m2 / 1e4, bounds_error=False, fill_value='extrapolate'),
                interp1d(u.quota_m, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    if kind == 'garcia_survey':
        # Curated official quota-area-volume table (validation_data/updated_curves/
        # garcia_2026.csv, from raw_data/Garcia_updated_GC.xlsx, outlier-curated by
        # the water authority), NOT our own re-gridded echosounder raster. Found
        # 2026-08-03: the raster (survey_dem_Garcia.tif) grids sonar points below the
        # survey's own waterline (~175.8 m) and separately-measured shore/terrain
        # points above it, with a 50 m distance filter from the nearest survey point;
        # the terrain points are far sparser than the sonar transects, so area/volume
        # above 175.8 m are increasingly under-represented as elevation rises away
        # from the shoreline (up to 16% area / 6.5% volume low at full pool, 190 m,
        # versus this curated table) -- an artifact of our own filtering, not the
        # reconstruction. The raster is still used for the pixel-level bias/RMSE
        # comparison in Section~sec:res_garcia (a genuinely 2D need this 1D table
        # can't replace), but this curated table is the correct AEV reference
        # everywhere else (Table tab:capacity, the hypsometry comparison, the
        # volume-timeseries validation).
        fp = UPDATED / 'garcia_2026.csv'
        if not fp.exists():
            return None
        u = pd.read_csv(fp)
        return (interp1d(u.quota_m, u.area_m2 / 1e4, bounds_error=False, fill_value='extrapolate'),
                interp1d(u.quota_m, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    if kind in EXT_CURVE_SPEC:
        pat, sheet, blocks = EXT_CURVE_SPEC[kind]
        hits = [h for h in glob.glob(str(NEWCURVE_EXT / pat))
                if h.lower().endswith(('.xls', '.xlsx'))]
        if not hits:
            return None
        raw = pd.read_excel(hits[0], sheet_name=sheet, header=None, engine='openpyxl')
        parts = []
        for qc, vc, ac in blocks:
            p = raw[[qc, vc, ac]].apply(pd.to_numeric, errors='coerce').dropna()
            p.columns = ['quota', 'vol_m3', 'area_m2']
            parts.append(p)
        u = pd.concat(parts).drop_duplicates(subset='quota')
        u = u[(u.quota > 50) & (u.quota < 1000)].sort_values('quota')
        return (interp1d(u.quota, u.area_m2 / 1e4, bounds_error=False, fill_value='extrapolate'),
                interp1d(u.quota, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    return None


# ── Change map (A vs B = sedimentation proxy) ──────────────────────────────────
def change_map(name):
    """DEM_B - DEM_A over the co-observed range, or None if Period A absent."""
    A = load_dem(name, 'A')
    B = load_dem(name, 'B')
    if A is None or B is None or A['arr'].shape != B['arr'].shape:
        return None
    both = A['mask'] & B['mask']
    lo = max(A['floor'], B['floor'])
    zone = both & (A['arr'] >= lo) & (B['arr'] >= lo)   # restrict to co-observed elevations
    diff = np.where(zone, B['arr'] - A['arr'], np.nan)
    return dict(diff=diff, zone=zone, lo=lo, bounds=B['bounds'])


# ── Consolidated capacity-change numbers (band-relative + total) ───────────────
def capacity_change(name, period='B'):
    """Return dict of band/total capacity change vs design (see consolidate_bathymetry.py).
    period='B' is the production gauge+SWOT-fallback DEM; period='B_swotonly' is the
    full-remote-sensing (FRS) DEM built by build_frs_dem.py -- same metrics, computed
    against the same design/updated curves, for the gauge+SWOT-vs-FRS comparison."""
    B = load_dem(name, period)
    if B is None:
        return None
    dc = design_curve(name)
    out = dict(floor=B['floor'], top=B['top'],
               vol_dem_rel=vol_exact(B['arr'], B['mask'], B['floor'], B['top'], B['pixel_ha']))
    if dc is None:
        return out
    _, des_vol = dc
    vdes_rel = float(des_vol(B['top']) - des_vol(B['floor']))
    out['vol_design_rel'] = vdes_rel
    out['sar_band_pct'] = (out['vol_dem_rel'] - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan
    uc = updated_curve(name)
    if uc is not None and RESERVOIRS[name]['updated'] in (
            'poma_new', 'rosamarina_2025', 'garcia_survey', *EXT_CURVE_SPEC.keys()):
        _, upd_vol = uc
        vupd_rel = float(upd_vol(B['top']) - upd_vol(B['floor']))
        out['truth_band_pct']  = (vupd_rel - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan
        out['truth_total_pct'] = (float(upd_vol(B['top'])) - float(des_vol(B['top']))) / float(des_vol(B['top'])) * 100
    return out
