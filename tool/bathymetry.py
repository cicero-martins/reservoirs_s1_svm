"""
tool/bathymetry.py — data layer for the Paper-2 bathymetry explorer (Streamlit MVP).

Loads the already-reconstructed Period-A/B satellite DEMs and the reference curves
(design + updated survey) for the 5 core Sicilian reservoirs, and derives AEV
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

REPO = pathlib.Path(__file__).resolve().parent.parent
DEM_DIR      = REPO / 'analysis' / 'schwatke_output'
PLANET_DIR   = DEM_DIR / 'planet'
TERRAIN_DIR  = DEM_DIR / 'terrain'
CURVE_BUNDLE = REPO / 'tool' / 'data' / 'curves'                                    # bundled (deploy)
CURVE_EXT    = pathlib.Path('C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi')  # local fallback
NEWCURVE_EXT = pathlib.Path('C:/Users/Unipa/Documents/GEE/Data/NewCurves')
UPDATED   = REPO / 'validation_data' / 'updated_curves'
PIXEL_HA  = 0.01

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
        u = u.apply(pd.to_numeric, errors='coerce').dropna().sort_values('quota')
        area = np.gradient(u.vol_m3.values, u.quota.values) / 1e4
        return (interp1d(u.quota, area, bounds_error=False, fill_value='extrapolate'),
                interp1d(u.quota, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    if kind == 'rosamarina_2025':
        fp = UPDATED / 'rosamarina_2025.csv'
        if not fp.exists():
            return None
        u = pd.read_csv(fp)
        return (interp1d(u.quota_m, u.area_m2 / 1e4, bounds_error=False, fill_value='extrapolate'),
                interp1d(u.quota_m, u.vol_m3 / 1e6, bounds_error=False, fill_value='extrapolate'))
    if kind == 'garcia_survey':
        # AEV straight from the gridded echosounder survey raster (full sonar+terrain).
        fp = DEM_DIR / 'garcia_survey' / 'survey_dem_Garcia.tif'
        if not fp.exists():
            return None
        with rasterio.open(fp) as s:
            g = s.read(1).astype(np.float64)
        m = np.isfinite(g)
        lv = np.arange(g[m].min(), g[m].max() + 1e-6, 0.5)
        ar, vo = aev(g, m, lv)
        # volume here is relative to the survey's own floor; caller aligns by floor.
        return (interp1d(lv, ar, bounds_error=False, fill_value='extrapolate'),
                interp1d(lv, vo, bounds_error=False, fill_value=np.nan))
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
def capacity_change(name):
    """Return dict of band/total capacity change vs design (see consolidate_bathymetry.py)."""
    B = load_dem(name, 'B')
    if B is None:
        return None
    dc = design_curve(name)
    levels = np.arange(B['floor'], B['top'] + 1e-6, 0.5)
    _, v_dem = aev(B['arr'], B['mask'], levels, B['pixel_ha'])
    out = dict(floor=B['floor'], top=B['top'], vol_dem_rel=float(v_dem[-1]))
    if dc is None:
        return out
    _, des_vol = dc
    vdes_rel = float(des_vol(B['top']) - des_vol(B['floor']))
    out['vol_design_rel'] = vdes_rel
    out['sar_band_pct'] = (out['vol_dem_rel'] - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan
    uc = updated_curve(name)
    if uc is not None and RESERVOIRS[name]['updated'] in ('poma_new', 'rosamarina_2025'):
        _, upd_vol = uc
        vupd_rel = float(upd_vol(B['top']) - upd_vol(B['floor']))
        out['truth_band_pct']  = (vupd_rel - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan
        out['truth_total_pct'] = (float(upd_vol(B['top'])) - float(des_vol(B['top']))) / float(des_vol(B['top'])) * 100
    return out
