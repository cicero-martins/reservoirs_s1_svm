"""
schwatke_bathymetry_3d.py

Bathymetric DEM reconstruction for the 4 Sicilian reservoirs using GEE-exported
water masks and in-situ water level data (gauge + DAHITI).

WORKFLOW
--------
Phase 1 — WL pairing (runs immediately, no GeoTIFFs needed):
  Match each selected mask date with the nearest in-situ WL observation.
  Output: analysis/schwatke_output/mask_wl_pairs_{reservoir}.csv

Phase 2 — DEM reconstruction (runs after GeoTIFFs downloaded to GEE_SicilyMasks/):
  Stack binary masks sorted by water level. For each consecutive mask pair
  (WL_lo, WL_hi), pixels that APPEAR at WL_hi but not WL_lo are assigned
  elevation ≈ (WL_lo + WL_hi) / 2. Interpolate gaps. Export DEM GeoTIFF.
  Output: analysis/schwatke_output/dem_{reservoir}_{period}.tif

Phase 3 — Visualisation:
  2D depth map (matplotlib contourf) and 3D surface (plotly interactive HTML).
  For Period A (2014-2016): gauge WL is unavailable → use area-rank as proxy for
  relative elevation (rank 1 = smallest area = deepest zone visible). The Period A
  map enables SPATIAL change detection against Period B by comparing which pixels
  were submerged at equivalent area percentiles in each epoch.

WL SOURCES
----------
  Poma       gauge 51527  (2022–present)
  Rosamarina gauge 50016  (2022–present)
  Pozzillo   gauge 58946  (2022–present)
  Ancipa     gauge 88601  (Oct 2024–present)

  For dates outside gauge coverage (all Period A + Ancipa pre-Oct-2024):
  The power-law model A = a * (h - h0)^b is calibrated on Period B pairs,
  then INVERTED to estimate h from observed area_ha. This gives relative
  elevation sufficient for 3D geometry but NOT for absolute comparisons.

USAGE
-----
    # Phase 1 (run now to see pairing quality):
    python analysis/schwatke_bathymetry_3d.py --phase 1

    # Phase 2 + 3 (run after downloading GeoTIFFs from Google Drive):
    python analysis/schwatke_bathymetry_3d.py --phase 2
    python analysis/schwatke_bathymetry_3d.py --phase 3

    # All phases:
    python analysis/schwatke_bathymetry_3d.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import binary_fill_holes, binary_closing, gaussian_filter

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO             = Path('.')
MASK_DIR         = REPO / 'raw_data' / 'GEE_SicilyMasks'
OUT_DIR          = REPO / 'analysis' / 'schwatke_output'
GAUGE_DIR        = OUT_DIR / 'gauge_downloads'
DATES_JSON       = REPO / 'analysis' / 'selected_mask_dates.json'
SAR_DIR          = REPO / 'raw_data' / 'GEE_GlobalPilotV2a'
CURVE_DIR        = Path('C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi')
ODS_MONTHLY_PATH = REPO / 'raw_data' / 'opendatasicilia' / 'sicilia_dighe_volumi.csv'
OUT_DIR.mkdir(exist_ok=True)

MAX_DT = 10  # days: max allowed gap for gauge-to-SAR matching

# ── Reservoir configs ─────────────────────────────────────────────────────────
DAHITI_DIR = REPO / 'validation_data' / 'DAHITI'

CONFIGS = {
    'Poma': {
        'gauge_csv':    GAUGE_DIR / 'poma_wl.csv',
        'gauge_min':    170.0,
        'sar_csv':      SAR_DIR / 'SAR_area_Poma.csv',
        'h0_bound_lo':  155.0,
        'dahiti_csv':   DAHITI_DIR / '42134_Poma_wl.csv',
        'boletin_cfg':  {
            'cod':        'dig-18',
            'curve_xls':  CURVE_DIR / 'Poma.xls',
            'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),
            'bias_corr':  -0.90,  # V->h overestimates by +0.9 m (sedimentation drift)
        },
    },
    'Rosamarina': {
        'gauge_csv':    GAUGE_DIR / 'rosamarina_wl.csv',
        'gauge_min':    130.0,
        'sar_csv':      SAR_DIR / 'SAR_area_Rosamarina.csv',
        'h0_bound_lo':  95.0,
        'dahiti_csv':   DAHITI_DIR / '42122_Rosamarina_wl.csv',
        'swot_csv':     REPO / 'validation_data' / 'SWOT' / 'Rosamarina_swot.csv',
        # 2026-07-21: the gauge has TWO separate multi-month stuck episodes, each at a
        # different flat value (a recurring sensor fault, not a one-off): 2024-01-24 to
        # 2025-04-02 (~145.71 m, the originally-known stuck period) and 2025-07-24 to
        # 2026-01-29 (~145.13 m, found when the new windowed Period-B masks landed
        # mostly inside it). SWOT is used in both.
        'gauge_bad_window': [('2024-01-24', '2025-04-02'), ('2025-07-24', '2026-01-29')],
        'boletin_cfg':  {
            'cod':        'dig-22',
            'curve_xls':  CURVE_DIR / 'Rosamarina.xls',
            'curve_cols': ('quota', 'area_ha', 'area_km2', 'vol_Mm3'),
            'bias_corr':  +1.65,  # V->h underestimates by -1.65 m
        },
    },
    'Pozzillo': {
        'gauge_csv':    GAUGE_DIR / 'pozzillo_wl.csv',
        'gauge_min':    330.0,
        'sar_csv':      SAR_DIR / 'SAR_area_Pozzillo.csv',
        'h0_bound_lo':  310.0,
        # boletin_cfg added 2026-07-21: the model-inversion fallback, fit on only 9
        # B-period pairs all <=356 m, was being asked to extrapolate Period-A's large-area
        # dates (360-590 ha) and inverting to 377-392 m -- ABOVE the dam's own design-curve
        # max of 366.5 m (790 ha), a physically impossible level. The earlier "V->h
        # unreliable at low volumes (R2=-0.24)" finding concerned LOW volumes specifically;
        # these large-area dates sit well inside the design curve's own tabulated range
        # (interpolation, not extrapolation), so boletin is a safer source for them than
        # the wildly-extrapolated model. bias_corr unvalidated (no independent check yet).
        'boletin_cfg':  {
            'cod':        'dig-19',
            'curve_xls':  CURVE_DIR / 'Pozzillo.xls',
            'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),
            'bias_corr':  0.0,
        },
    },
    'Ancipa': {
        'gauge_csv':    GAUGE_DIR / 'ancipa_livello_secca.csv',
        'gauge_min':    880.0,
        'sar_csv':      SAR_DIR / 'SAR_area_Ancipa.csv',
        'h0_bound_lo':  860.0,
        # GEE centroid-inside filter selects upstream patches instead of dam body;
        # anchor-based post-processing fixes this in Phase 2.
        'fix_components': True,
        'boletin_cfg':  {
            'cod':        'dig-01',
            'curve_xls':  CURVE_DIR / 'Ancipa.xls',
            'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),  # no area_ha column
            'bias_corr':  -0.93,  # limited validation (9 months only)
        },
    },
    # --- Extended set (Fase 3, 2026-07): full DEM reconstruction added for the 3
    # reservoirs that previously only had scalar hypsometry (schwatke_extended.py).
    # h0_bound_lo/gauge_min set conservatively ~15m below each design curve's own
    # zero-area quota (Arancio 157m, Castello 250m, Olivo 408m -- see
    # C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/{name}.xls); bias_corr left
    # at 0.0 (unvalidated -- no V->h-vs-gauge check done yet for these 3, unlike the
    # original 5). sar_csv points at the VV-Otsu area series from
    # exportSicilyExtended.js (different folder than the other 5's GlobalPilotV2a,
    # but same date/area_ha schema). Masks use SVM (exportSicilyMasks.js pipeline,
    # via analysis/export_extended_masks.py) for consistency with the 5 core
    # reservoirs -- NOT the VV-Otsu masks that produced this sar_csv, which is used
    # here only for Period-A SAR<->gauge model-fit pairing, same role as the other 5.
    'Arancio': {
        'gauge_csv':    GAUGE_DIR / 'arancio_wl.csv',
        'gauge_min':    160.0,
        'sar_csv':      REPO / 'raw_data' / 'exportSicilyExtended' / 'GEE_SicilyExtended_VVotsu' / 'SAR_area_Arancio.csv',
        'h0_bound_lo':  145.0,
        'boletin_cfg':  {
            'cod':        'dig-02',
            'curve_xls':  CURVE_DIR / 'Arancio.xls',
            'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),
            'bias_corr':  0.0,  # unvalidated
        },
    },
    'Castello': {
        'gauge_csv':    GAUGE_DIR / 'castello_wl.csv',
        'gauge_min':    260.0,
        'sar_csv':      REPO / 'raw_data' / 'exportSicilyExtended' / 'GEE_SicilyExtended_VVotsu' / 'SAR_area_Castello.csv',
        'h0_bound_lo':  245.0,
        'boletin_cfg':  {
            'cod':        'dig-03',
            'curve_xls':  CURVE_DIR / 'Castello.xls',
            'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),
            'bias_corr':  0.0,  # unvalidated
        },
    },
    'Olivo': {
        'gauge_csv':    GAUGE_DIR / 'olivo_wl.csv',
        'gauge_min':    415.0,
        'sar_csv':      REPO / 'raw_data' / 'exportSicilyExtended' / 'GEE_SicilyExtended_VVotsu' / 'SAR_area_Olivo.csv',
        'h0_bound_lo':  400.0,
        'boletin_cfg':  {
            'cod':        'dig-15',
            'curve_xls':  CURVE_DIR / 'Olivo.xls',
            'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),
            'bias_corr':  0.0,  # unvalidated
        },
    },
    'Nicoletti': {
        'gauge_csv':    GAUGE_DIR / 'nicoletti_wl.csv',
        'gauge_min':    370.0,
        'sar_csv':      REPO / 'raw_data' / 'exportSicilyExtended' / 'GEE_SicilyExtended_VVotsu' / 'SAR_area_Nicoletti.csv',
        'h0_bound_lo':  355.0,
        'boletin_cfg':  {
            'cod':        'dig-13',
            'curve_xls':  CURVE_DIR / 'Nicoletti.xls',
            'curve_cols': ('quota', 'area_km2', 'area_ha', 'vol_Mm3'),
            'bias_corr':  0.0,  # unvalidated
        },
    },
    'Garcia': {
        'gauge_csv':    GAUGE_DIR / 'garcia_idrometro_radar.csv',
        'gauge_min':    170.0,
        'sar_csv':      SAR_DIR / 'SAR_area_Garcia.csv',
        'h0_bound_lo':  155.0,
        'swot_csv':     REPO / 'validation_data' / 'SWOT' / 'Garcia_swot.csv',
        # 2026-07-21: the gauge reads a corrupted, noisy dry-lakebed floor
        # (~176.0-176.5 m regardless of true level) from 2025-08-06 to 2026-02-03 --
        # confirmed against SWOT, which shows a real decline to ~172 m over the same
        # window; the gauge then recovers and tracks a real 176.5->189.8 m refill from
        # 2026-02-04 on. Dates in this window use SWOT instead of the gauge.
        'gauge_bad_window': ('2025-08-06', '2026-02-03'),
        # V->h validation vs AEGIS gauge: n=44, bias=-0.10 m, RMSE=1.04 m, R2=0.954
        # Bias essentially zero — no significant sedimentation signal in design curve.
        'boletin_cfg':  {
            'cod':        'dig-09',
            'curve_xls':  CURVE_DIR / 'Garcia.xls',
            'curve_cols': ('quota', 'area_km2', 'vol_Mm3'),  # 3 data cols, no area_ha
            'bias_corr':  +0.10,
        },
    },
}

PERIOD_LABELS = {
    'A':         '2014-2016 (model WL)',
    'A_boletin': '2014-2016 (boletin V->h)',
    'B':         '2022-2026 (gauge)',
    'B_dahiti':  '2022-2026 (DAHITI)',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gauge(cfg: dict, flat_tol: float = 0.005, flat_min_days: int = 5) -> pd.Series:
    """Load gauge CSV → daily mean water level Series (date index).

    Removes 'stuck sensor' periods: runs of flat_min_days or more consecutive
    days where the daily mean changes by less than flat_tol metres.
    These arise from sensor failure or AEGIS data quality issues.
    """
    df = pd.read_csv(cfg['gauge_csv'])
    df.columns = [c.strip() for c in df.columns]
    time_col = 'time' if 'time' in df.columns else df.columns[0]
    wl_col   = 'wl_m'  if 'wl_m'  in df.columns else df.columns[1]
    df['_dt'] = pd.to_datetime(df[time_col], errors='coerce')
    df['_wl'] = pd.to_numeric(df[wl_col], errors='coerce')
    df = df.dropna(subset=['_dt', '_wl'])
    df = df[df['_wl'] > cfg['gauge_min']]
    daily = df.set_index('_dt')['_wl'].resample('D').mean().dropna()

    # Detect and remove flat (stuck sensor) periods
    run_len = pd.Series(0, index=daily.index, dtype=int)
    count = 0
    for i, chg in enumerate(daily.diff().abs()):
        count = count + 1 if (not np.isnan(chg) and chg < flat_tol) else 0
        run_len.iloc[i] = count
    stuck = run_len >= flat_min_days
    n_removed = int(stuck.sum())
    if n_removed > 0:
        daily = daily[~stuck]

    return daily.rename('wl_m')


def load_dahiti(path: Path) -> pd.Series:
    """Load DAHITI CSV → daily mean WSE Series (date index)."""
    df = pd.read_csv(path)
    df['_dt'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['_wl'] = pd.to_numeric(df['wse'], errors='coerce')
    df = df.dropna(subset=['_dt', '_wl'])
    daily = df.set_index('_dt')['_wl'].resample('D').mean().dropna()
    return daily.rename('wl_dahiti')


def interp_wl(series: pd.Series, dt: pd.Timestamp, max_gap_days: float) -> float:
    """Water level at dt, linearly interpolated by time between the nearest real
    observation before and the nearest after dt (each within max_gap_days),
    rather than snapped to whichever single observation happens to be closest.
    Matters most for sparse sources like SWOT (~10-21 day revisit): nearest-
    neighbour can be many days off dt, while the two real bracketing passes
    let us estimate the level actually reached in between. Falls back to
    whichever single side is available if only one exists in range."""
    if len(series) == 0:
        return np.nan
    before = series.loc[series.index <= dt]
    after  = series.loc[series.index > dt]
    have_b = len(before) > 0 and (dt - before.index[-1]).days <= max_gap_days
    have_a = len(after)  > 0 and (after.index[0] - dt).days <= max_gap_days
    if have_b and have_a:
        t0, t1 = before.index[-1], after.index[0]
        v0, v1 = float(before.iloc[-1]), float(after.iloc[0])
        frac = (dt - t0) / (t1 - t0)
        return v0 + (v1 - v0) * frac
    if have_b:
        return float(before.iloc[-1])
    if have_a:
        return float(after.iloc[0])
    return np.nan


def load_swot(path: Path) -> pd.Series:
    """Load SWOT LakeSP CSV -> daily mean WSE Series (date index), quality-screened."""
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df[df['quality_f'].isin([0, 1])]
    df['_dt'] = df['datetime'].dt.tz_localize(None)
    df['_wl'] = pd.to_numeric(df['wse'], errors='coerce')
    df = df.dropna(subset=['_dt', '_wl'])
    daily = df.set_index('_dt')['_wl'].resample('D').mean().dropna()
    return daily.rename('wl_swot')


def build_anchor_mask(res: str) -> np.ndarray | None:
    """Build a binary anchor mask = pixels present in >= 50% of all reservoir GeoTIFFs.

    Used to select the correct connected component when the classifier picks
    up disconnected water bodies (e.g. upstream areas instead of the dam zone).
    Returns None if fewer than 3 masks exist.
    """
    try:
        import rasterio
    except ImportError:
        return None

    tifs = sorted(MASK_DIR.glob(f'mask_{res}_*.tif'))
    if len(tifs) < 3:
        return None

    count = None
    for tif in tifs:
        with rasterio.open(tif) as src:
            arr = src.read(1)
        water = (arr == 1).astype(np.int16)
        count = water if count is None else count + water

    threshold = len(tifs) * 0.5
    return count >= threshold


def select_main_component(arr: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep the connected water component with the most overlap with anchor.

    Falls back to the largest component if no component overlaps anchor.
    """
    from scipy.ndimage import label as nd_label
    water = (arr == 1)
    labeled, n = nd_label(water)
    if n <= 1:
        return arr  # nothing to do

    best_comp, best_overlap, best_size = 0, 0, 0
    for comp in range(1, n + 1):
        comp_mask = labeled == comp
        overlap   = int(np.sum(comp_mask & anchor))
        size      = int(np.sum(comp_mask))
        if overlap > best_overlap or (overlap == best_overlap and size > best_size):
            best_overlap = overlap
            best_comp    = comp
            best_size    = size

    if best_comp == 0:
        # No anchor overlap — fall back to largest component
        sizes = [np.sum(labeled == c) for c in range(1, n + 1)]
        best_comp = int(np.argmax(sizes)) + 1

    result = arr.copy()
    result[labeled != best_comp] = 0
    return result


def load_boletin_wl(cfg: dict) -> pd.Series:
    """Monthly boletin volume -> h via official design V-h curve, linearly interpolated to daily.

    Returns an empty Series if boletin_cfg is not present or files are missing.
    """
    bcfg = cfg.get('boletin_cfg')
    if bcfg is None or not ODS_MONTHLY_PATH.exists():
        return pd.Series(dtype=float, name='wl_boletin')

    xls = bcfg['curve_xls']
    if not xls.exists():
        return pd.Series(dtype=float, name='wl_boletin')

    # Build V->h interpolator from official design curve
    cols = list(bcfg['curve_cols'])
    col_idx = list(range(2, 2 + len(cols)))
    df_c = pd.read_excel(xls, sheet_name=0, header=None)
    data = df_c[col_idx].apply(pd.to_numeric, errors='coerce').dropna()
    data.columns = cols
    data = data.sort_values('quota').reset_index(drop=True)
    v2h = interp1d(data['vol_Mm3'], data['quota'],
                   kind='linear', bounds_error=False, fill_value='extrapolate')

    # Monthly volumes -> WL
    monthly = pd.read_csv(ODS_MONTHLY_PATH, parse_dates=['data'])
    mon = monthly[monthly['cod'] == bcfg['cod']][['data', 'volume']].copy()
    mon = mon.sort_values('data').set_index('data')
    mon['h_Vh'] = v2h(mon['volume'].values) + bcfg.get('bias_corr', 0.0)

    # Linearly interpolate to daily (reservoir levels change slowly)
    daily_idx = pd.date_range(mon.index.min(), mon.index.max(), freq='D')
    daily = mon['h_Vh'].reindex(daily_idx).interpolate(method='time')
    return daily.rename('wl_boletin')


import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _dem_recon import build_dem as _recon_build   # shared bathtub reconstruction

def build_dem_from_arrays(masks_raw: list, wls: list) -> np.ndarray:
    """Reconstruct the bathymetric DEM from (mask, WL) pairs (SAR grid = 10 m).

    Delegates to the shared reconstruction in _dem_recon.build_dem: a persistence
    footprint (keeps disconnected in-reservoir pools such as a near-dam pool, and
    rejects external water), a max-WL rim, per-mask despeckling and masked smoothing —
    replacing the earlier union + outside-min-fill that produced a spiky 'mountain
    range' DEM with a dipped border.
    """
    return _recon_build(list(masks_raw), list(wls), pixel_m=10.0)


def power_law(h, a, h0, b):
    return a * (h - h0) ** b


def fit_hyps_model(pairs: pd.DataFrame, h0_lo: float):
    """Fit A = a*(h-h0)^b to SAR-gauge pairs. Returns (a, h0, b) or None."""
    valid = pairs.dropna(subset=['area_ha', 'wl_m'])
    valid = valid[(valid['area_ha'] > 1) & (valid['wl_m'] > h0_lo + 0.5)]
    if len(valid) < 6:
        return None
    try:
        h0_init = valid['wl_m'].min() - 1.0
        p0 = [1.0, max(h0_lo, h0_init), 1.5]
        bounds = ([0, h0_lo, 0.5], [1e6, valid['wl_m'].min() - 0.01, 5.0])
        popt, _ = curve_fit(
            power_law,
            valid['wl_m'].values,
            valid['area_ha'].values,
            p0=p0, bounds=bounds, maxfev=10000,
        )
        return popt  # (a, h0, b)
    except Exception:
        return None


def invert_power_law(area_ha: float, a: float, h0: float, b: float) -> float:
    """Invert A = a*(h-h0)^b → h = h0 + (A/a)^(1/b)."""
    if area_ha <= 0 or a <= 0:
        return np.nan
    return h0 + (area_ha / a) ** (1.0 / b)


# ── Phase 1: WL pairing ───────────────────────────────────────────────────────

def phase1():
    print('\n=== Phase 1: WL pairing ===\n')

    dates_json = json.loads(DATES_JSON.read_text())
    sar_daily  = {}
    for res in CONFIGS:
        df = pd.read_csv(CONFIGS[res]['sar_csv'], parse_dates=['date'])
        df = df[['date', 'area_ha']].dropna().set_index('date').resample('D').mean()
        sar_daily[res] = df['area_ha'].rename('area_ha')

    for res, cfg in CONFIGS.items():
        print(f'--- {res} ---')

        # Load gauge (may be empty / only recent)
        try:
            gauge = load_gauge(cfg)
            print(f'  Gauge: {gauge.index.min().date()} to {gauge.index.max().date()}'
                  f'  ({len(gauge)} daily obs)  WL {gauge.min():.1f}–{gauge.max():.1f} m')
        except Exception as e:
            gauge = pd.Series(dtype=float, name='wl_m')
            print(f'  Gauge: UNAVAILABLE ({e})')

        # Load DAHITI if configured
        dahiti = pd.Series(dtype=float, name='wl_dahiti')
        if 'dahiti_csv' in cfg and cfg['dahiti_csv'].exists():
            try:
                dahiti = load_dahiti(cfg['dahiti_csv'])
                print(f'  DAHITI: {dahiti.index.min().date()} to {dahiti.index.max().date()}'
                      f'  ({len(dahiti)} daily obs)  WL {dahiti.min():.1f}-{dahiti.max():.1f} m')
            except Exception as e:
                print(f'  DAHITI: UNAVAILABLE ({e})')

        # Load SWOT if configured (used to replace the gauge inside a known bad window)
        swot = pd.Series(dtype=float, name='wl_swot')
        if 'swot_csv' in cfg and cfg['swot_csv'].exists():
            try:
                swot = load_swot(cfg['swot_csv'])
                print(f'  SWOT: {swot.index.min().date()} to {swot.index.max().date()}'
                      f'  ({len(swot)} obs)  WL {swot.min():.1f}-{swot.max():.1f} m')
            except Exception as e:
                print(f'  SWOT: UNAVAILABLE ({e})')
        # gauge_bad_window: one (start, end) pair, or a list of them for reservoirs
        # with more than one stuck episode (e.g. Rosamarina: two separate multi-month
        # stuck runs at two different flat values -- a recurring sensor fault, not a
        # one-off).
        gauge_bad = cfg.get('gauge_bad_window')
        bad_windows = []
        if gauge_bad:
            raw_windows = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
            for lo, hi in raw_windows:
                bad_windows.append((pd.Timestamp(lo), pd.Timestamp(hi)))
                print(f'  Gauge known-bad window: {lo} to {hi} (using SWOT there)')

        # Load boletin V->h (Period A independent WL source)
        try:
            boletin = load_boletin_wl(cfg)
            if len(boletin) > 0:
                bcfg = cfg['boletin_cfg']
                corr = bcfg.get('bias_corr', 0.0)
                print(f'  Boletin V->h: {boletin.index.min().date()} to {boletin.index.max().date()}'
                      f'  WL {boletin.min():.1f}-{boletin.max():.1f} m  (bias_corr={corr:+.2f} m)')
        except Exception as e:
            boletin = pd.Series(dtype=float, name='wl_boletin')
            print(f'  Boletin V->h: UNAVAILABLE ({e})')

        # Fit power-law model on B-period SAR+gauge pairs for inversion
        # Only possible for reservoirs where period B overlaps with gauge
        sar = sar_daily[res]
        pairs_B = match_sar_gauge(sar, gauge, dates_json[res].get('B', []))
        model   = fit_hyps_model(pairs_B, cfg['h0_bound_lo'])
        if model is not None:
            a, h0, b = model
            print(f'  Model A = {a:.3f} * (h - {h0:.2f})^{b:.3f}  '
                  f'[fitted on {len(pairs_B.dropna())} B-period pairs]')
        else:
            print('  Model: could not fit (insufficient B-period pairs with WL)')
            a, h0, b = None, None, None

        # Pair all selected dates with WL (gauge or model-inferred)
        rows = []
        for period in ('A', 'B'):
            period_dates = dates_json[res].get(period, [])
            for entry in period_dates:
                date_str = entry['date']
                area_ha  = entry['area_ha']
                dt       = pd.Timestamp(date_str)

                # Try gauge match within ±MAX_DT days (always first priority) -- unless
                # this date falls inside a known gauge-bad window (e.g. Garcia's
                # dry-lakebed floor reading), in which case skip the gauge entirely and
                # try SWOT first instead.
                wl_m   = np.nan
                source = 'none'
                in_bad_window = any(lo <= dt <= hi for lo, hi in bad_windows)
                if len(gauge) > 0 and not in_bad_window:
                    val = interp_wl(gauge, dt, MAX_DT)
                    if not np.isnan(val):
                        wl_m   = val
                        source = 'gauge'

                if np.isnan(wl_m) and in_bad_window and len(swot) > 0:
                    val = interp_wl(swot, dt, MAX_DT)
                    if not np.isnan(val):
                        wl_m   = val
                        source = 'swot'

                # Try boletin V->h before model inversion (independent source). Used for
                # both periods: Period A always lacks a gauge, and a reservoir's gauge can
                # also start mid-way through Period B (e.g. Nicoletti, gauge from 2023-11),
                # leaving earlier B-period dates with neither a gauge reading nor enough
                # gauge-matched pairs to fit the power-law model at all.
                if np.isnan(wl_m) and len(boletin) > 0:
                    win_b = boletin.loc[
                        (boletin.index >= dt - pd.Timedelta(days=15)) &
                        (boletin.index <= dt + pd.Timedelta(days=15))
                    ]
                    if len(win_b) > 0:
                        wl_m   = float(win_b.iloc[
                            int(np.argmin(np.abs((win_b.index - dt).total_seconds().values)))
                        ])
                        source = 'boletin'

                # Fall back to model inversion (Period B without gauge, or Pozzillo Period A)
                if np.isnan(wl_m) and a is not None:
                    wl_m   = invert_power_law(area_ha, a, h0, b)
                    source = 'model'

                # DAHITI match within ±MAX_DT days
                wl_dahiti_val = np.nan
                if len(dahiti) > 0:
                    win_d = dahiti.loc[
                        (dahiti.index >= dt - pd.Timedelta(days=MAX_DT)) &
                        (dahiti.index <= dt + pd.Timedelta(days=MAX_DT))
                    ]
                    if len(win_d) > 0:
                        del_d   = np.abs((win_d.index - dt).total_seconds().values)
                        wl_dahiti_val = float(win_d.iloc[int(np.argmin(del_d))])

                # Boletin V->h WL (store independently for DEM comparison)
                wl_boletin_val = np.nan
                if len(boletin) > 0:
                    win_b2 = boletin.loc[
                        (boletin.index >= dt - pd.Timedelta(days=15)) &
                        (boletin.index <= dt + pd.Timedelta(days=15))
                    ]
                    if len(win_b2) > 0:
                        wl_boletin_val = float(win_b2.iloc[
                            int(np.argmin(np.abs((win_b2.index - dt).total_seconds().values)))
                        ])

                rows.append({
                    'reservoir':    res,
                    'period':       period,
                    'date':         date_str,
                    'area_ha':      area_ha,
                    'wl_m':         wl_m,
                    'wl_source':    source,
                    'wl_dahiti':    wl_dahiti_val,
                    'wl_boletin':   wl_boletin_val,
                })

        out_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f'mask_wl_pairs_{res}.csv'
        out_df.to_csv(out_path, index=False, float_format='%.4f')

        n_gauge  = (out_df['wl_source'] == 'gauge').sum()
        n_boletin= (out_df['wl_source'] == 'boletin').sum()
        n_swot   = (out_df['wl_source'] == 'swot').sum()
        n_model  = (out_df['wl_source'] == 'model').sum()
        n_none   = (out_df['wl_source'] == 'none').sum()
        print(f'  Pairs saved → {out_path.name}'
              f'  (gauge={n_gauge}, boletin={n_boletin}, swot={n_swot}, model={n_model}, none={n_none})')
        if len(out_df.dropna(subset=['wl_m'])) > 0:
            print(f'  WL range (all): '
                  f'{out_df["wl_m"].min():.2f}–{out_df["wl_m"].max():.2f} m')


def match_sar_gauge(sar: pd.Series, gauge: pd.Series,
                    period_entries: list) -> pd.DataFrame:
    """Return DataFrame with area_ha + wl_m for dates in period_entries."""
    rows = []
    for entry in period_entries:
        dt      = pd.Timestamp(entry['date'])
        area_ha = entry['area_ha']
        wl_m    = np.nan
        if len(gauge) > 0:
            window = gauge.loc[
                (gauge.index >= dt - pd.Timedelta(days=MAX_DT)) &
                (gauge.index <= dt + pd.Timedelta(days=MAX_DT))
            ]
            if len(window) > 0:
                deltas  = np.abs((window.index - dt).total_seconds().values)
                closest = int(np.argmin(deltas))
                wl_m    = window.iloc[closest]
        rows.append({'area_ha': area_ha, 'wl_m': wl_m})
    return pd.DataFrame(rows)


# ── Phase 2: DEM reconstruction ───────────────────────────────────────────────

def phase2():
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS
    except ImportError:
        print('ERROR: rasterio is required for Phase 2.  pip install rasterio')
        return

    print('\n=== Phase 2: DEM reconstruction ===\n')

    if not MASK_DIR.exists():
        print(f'ERROR: mask directory not found: {MASK_DIR}')
        print('Download GeoTIFFs from Google Drive folder GEE_SicilyMasks first.')
        return

    for res in CONFIGS:
        pairs_file = OUT_DIR / f'mask_wl_pairs_{res}.csv'
        if not pairs_file.exists():
            print(f'{res}: run Phase 1 first to generate {pairs_file.name}')
            continue

        pairs = pd.read_csv(pairs_file, parse_dates=['date'])
        pairs = pairs.sort_values('wl_m').reset_index(drop=True)

        for period in ('A', 'B'):
            label    = PERIOD_LABELS[period]
            sub      = pairs[pairs['period'] == period].copy()
            sub      = sub.dropna(subset=['wl_m']).sort_values('wl_m')

            # For period A where WL source is 'model', note this in output
            has_abs  = (sub['wl_source'] == 'gauge').any()
            print(f'\n{res} [{label}]: {len(sub)} masks, '
                  f'WL {"absolute" if has_abs else "model-inferred"}')

            if len(sub) < 3:
                print(f'  Skipping: fewer than 3 valid masks with WL.')
                continue

            # Load all GeoTIFFs for this period (raw arrays + all WL columns)
            raw_arrays, wls_gauge, wls_dahiti, wls_boletin, meta = [], [], [], [], None
            has_dahiti  = 'wl_dahiti'  in sub.columns
            has_boletin = 'wl_boletin' in sub.columns
            for _, row in sub.iterrows():
                date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
                tif_path = MASK_DIR / f'mask_{res}_{date_str}.tif'
                if not tif_path.exists():
                    print(f'  Missing: {tif_path.name}')
                    continue
                with rasterio.open(tif_path) as src:
                    arr = src.read(1).astype(np.float32)
                    if meta is None:
                        meta = src.meta.copy()
                raw_arrays.append(arr)
                wls_gauge.append(row['wl_m'])
                wls_dahiti.append(row['wl_dahiti']  if has_dahiti  else np.nan)
                wls_boletin.append(row['wl_boletin'] if has_boletin else np.nan)

            if len(raw_arrays) < 3:
                print(f'  Skipping: only {len(raw_arrays)} GeoTIFFs found on disk.')
                continue

            # Anchor-based component selection: keep only the connected water body
            # that overlaps the always-submerged core (≥50% of all masks).
            # Fixes GEE centroid-inside filter failures for elongated valley reservoirs.
            cfg = CONFIGS[res]
            # Component fix (select_main_component) is now handled inside the shared
            # reconstruction: _dem_recon.build_dem uses a persistence footprint that
            # rejects external water AND keeps disconnected in-reservoir pools, so the
            # old per-mask largest-component step (which dropped Ancipa's near-dam pool)
            # is no longer applied.

            def _save_dem(arrays, wls_vals, suffix):
                valid = [(a, w) for a, w in zip(arrays, wls_vals)
                         if not np.isnan(w)]
                if len(valid) < 3:
                    return
                valid.sort(key=lambda x: x[1])
                arrs, wls_s = zip(*valid)
                dem_out = build_dem_from_arrays(list(arrs), list(wls_s))
                depth   = np.nanmin(dem_out) - max(wls_s)
                out_meta = meta.copy()
                out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
                out_tif = OUT_DIR / f'dem_{res}_{suffix}.tif'
                with rasterio.open(out_tif, 'w', **out_meta) as dst:
                    dst.write(dem_out[np.newaxis, :, :])
                print(f'  DEM saved: {out_tif.name}  '
                      f'WL range {min(wls_s):.1f}–{max(wls_s):.1f} m  '
                      f'depth range {depth:.1f}–0 m')

            _save_dem(raw_arrays, wls_gauge, period)

            # Boletin V->h DEM for period A (independent WL baseline)
            if period == 'A' and has_boletin:
                n_bol = sum(1 for w in wls_boletin if not np.isnan(w))
                if n_bol >= 3:
                    print(f'  Building boletin V->h DEM ({n_bol} dates)...')
                    _save_dem(raw_arrays, wls_boletin, 'A_boletin')
                else:
                    print(f'  Boletin DEM skipped: only {n_bol} dates matched (need >=3)')

            # DAHITI-based DEM for period B (satellite-only validation)
            if period == 'B' and has_dahiti:
                n_dahiti = sum(1 for w in wls_dahiti if not np.isnan(w))
                if n_dahiti >= 3:
                    print(f'  Building DAHITI DEM ({n_dahiti} dates matched)...')
                    _save_dem(raw_arrays, wls_dahiti, 'B_dahiti')
                else:
                    print(f'  DAHITI DEM skipped: only {n_dahiti} dates matched '
                          f'(need ≥3)')


# ── Phase 3: Visualisation ────────────────────────────────────────────────────

def phase3():
    try:
        import rasterio
        import plotly.graph_objects as go
    except ImportError:
        print('ERROR: rasterio and plotly are required for Phase 3.')
        print('  pip install rasterio plotly')
        return

    print('\n=== Phase 3: Visualisation ===\n')

    for res in CONFIGS:
        dem_files = list(OUT_DIR.glob(f'dem_{res}_*.tif'))
        if not dem_files:
            print(f'{res}: no DEM files found — run Phase 2 first.')
            continue

        period_data = {}
        for f in sorted(dem_files):
            # stem examples: dem_Poma_A, dem_Poma_B, dem_Poma_B_dahiti
            prefix = f'dem_{res}_'
            period = f.stem[len(prefix):]   # 'A', 'B', or 'B_dahiti'
            with rasterio.open(f) as src:
                dem   = src.read(1).astype(np.float64)
                dem[dem == src.nodata] = np.nan
                if hasattr(src, 'transform'):
                    cols  = np.arange(src.width)
                    rows_ = np.arange(src.height)
                    xs    = src.transform.c + cols  * src.transform.a
                    ys    = src.transform.f + rows_ * src.transform.e
                else:
                    xs = np.arange(dem.shape[1])
                    ys = np.arange(dem.shape[0])
            period_data[period] = {'dem': dem, 'xs': xs, 'ys': ys}

        # ── 2D depth map ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            1, len(period_data),
            figsize=(6 * len(period_data), 5),
            squeeze=False,
        )
        for col_i, (period, d) in enumerate(sorted(period_data.items())):
            ax    = axes[0][col_i]
            label = PERIOD_LABELS.get(period, period)
            dem   = d['dem']
            wl_max = np.nanmax(dem)
            depth  = dem - wl_max

            im = ax.imshow(
                depth, cmap='Blues_r',
                vmin=np.nanmin(depth), vmax=0,
                origin='upper',
            )
            plt.colorbar(im, ax=ax, label='Depth (m)')
            ax.set_title(f'{res} — {label}', fontsize=11)
            ax.set_xlabel('Column (px)')
            ax.set_ylabel('Row (px)')

        fig.suptitle(f'{res}: Bathymetric depth map', fontsize=13)
        fig.tight_layout()
        out_fig = OUT_DIR / f'bathymetry_{res}_2D.png'
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f'  2D figure: {out_fig.name}')

        # ── 3D interactive (plotly) ───────────────────────────────────────────
        for period, d in sorted(period_data.items()):
            label = PERIOD_LABELS.get(period, period)
            dem   = d['dem']
            xs    = d['xs']
            ys    = d['ys']

            # Downsample for plotly (max ~200×200 grid)
            step  = max(1, max(dem.shape) // 200)
            dem_s = dem[::step, ::step]
            xs_s  = xs[::step]
            ys_s  = ys[::step]

            # Compute aspect ratio from actual data extents (20× vertical exag.)
            EXAG    = 20
            x_range = float(abs(xs_s[-1] - xs_s[0]))
            y_range = float(abs(ys_s[0]  - ys_s[-1]))
            z_range = float(np.nanmax(dem_s) - np.nanmin(dem_s)) if not np.all(np.isnan(dem_s)) else 1.0
            horiz   = max(x_range, y_range, 1.0)
            z_ratio = max((z_range * EXAG) / horiz, 0.01)
            y_ratio = y_range / horiz

            fig_3d = go.Figure(go.Surface(
                z=dem_s,
                x=xs_s,
                y=ys_s,
                colorscale='Blues_r',
                showscale=True,
                colorbar=dict(title='WL (m a.s.l.)'),
            ))
            fig_3d.update_layout(
                title=f'{res} — {label} bathymetry (3D)',
                scene=dict(
                    xaxis_title='Easting (m)',
                    yaxis_title='Northing (m)',
                    zaxis_title='Elevation (m a.s.l.)',
                    aspectmode='manual',
                    aspectratio=dict(x=1.0, y=y_ratio, z=z_ratio),
                ),
            )
            out_html = OUT_DIR / f'bathymetry_{res}_{period}_3D.html'
            fig_3d.write_html(str(out_html))
            print(f'  3D figure: {out_html.name}')

        # ── Sedimentation difference map (A vs B) ────────────────────────────
        if 'A' in period_data and 'B' in period_data:
            dem_A = period_data['A']['dem']
            dem_B = period_data['B']['dem']

            if dem_A.shape != dem_B.shape:
                print(f'  Skipping diff map: shapes differ '
                      f'{dem_A.shape} vs {dem_B.shape}')
            else:
                diff      = dem_B - dem_A
                mask_both = ~np.isnan(dem_A) & ~np.isnan(dem_B)

                # Restrict comparison to the overlapping WL range.
                # Always-wet pixels get assigned the period's floor WL, so
                # diff = floor_B − floor_A ≠ real bathymetric change.
                # Valid comparison: both periods observed the pixel transitioning
                # from water to land, i.e. pixel elevation > max(floor_A, floor_B).
                floor_A = float(np.nanmin(dem_A[~np.isnan(dem_A)]))
                floor_B = float(np.nanmin(dem_B[~np.isnan(dem_B)]))
                wl_overlap_lo = max(floor_A, floor_B)
                overlap_mask = (mask_both
                                & (dem_A > wl_overlap_lo)
                                & (dem_B > wl_overlap_lo))

                fig, ax = plt.subplots(figsize=(7, 6))

                # Grey: area covered by both DEMs but outside overlap range
                grey = np.where(mask_both & ~overlap_mask, 1.0, np.nan)
                ax.imshow(grey, cmap='Greys', vmin=0, vmax=2,
                          origin='upper', alpha=0.4)

                vmax = (np.nanpercentile(np.abs(diff[overlap_mask]), 95)
                        if np.any(overlap_mask) else 1.0)
                im = ax.imshow(
                    np.where(overlap_mask, diff, np.nan),
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper',
                )
                plt.colorbar(im, ax=ax, label='Elevation change (m)  [+= raised]')
                ax.set_title(
                    f'{res}: elevation change {PERIOD_LABELS["A"]} vs {PERIOD_LABELS["B"]}\n'
                    f'Valid zone: WL > {wl_overlap_lo:.1f} m'
                    f'  (grey = outside overlapping WL range)\n'
                    '(+) = bathymetric surface raised = sedimentation',
                    fontsize=9,
                )
                out_diff = OUT_DIR / f'bathymetry_{res}_change.png'
                fig.tight_layout()
                fig.savefig(out_diff, dpi=150)
                plt.close(fig)

                net_change = (np.nanmean(diff[overlap_mask])
                              if np.any(overlap_mask) else np.nan)
                n_valid = int(np.sum(overlap_mask))
                n_excl  = int(np.sum(mask_both & ~overlap_mask))
                print(f'  Change map: {out_diff.name}  '
                      f'mean delta (overlap zone) = {net_change:+.2f} m  '
                      f'valid px: {n_valid}, excluded px: {n_excl}')

        # ── Gauge vs DAHITI comparison ────────────────────────────────────────
        if 'B' in period_data and 'B_dahiti' in period_data:
            dem_g = period_data['B']['dem']
            dem_d = period_data['B_dahiti']['dem']

            if dem_g.shape != dem_d.shape:
                print(f'  Skipping gauge/DAHITI comparison: shape mismatch')
            else:
                mask_both_gd = ~np.isnan(dem_g) & ~np.isnan(dem_d)
                bias = dem_d - dem_g   # DAHITI − gauge

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                wl_max_g = np.nanmax(dem_g)
                wl_max_d = np.nanmax(dem_d)
                depth_g  = np.where(mask_both_gd, dem_g - wl_max_g, np.nan)
                depth_d  = np.where(mask_both_gd, dem_d - wl_max_d, np.nan)
                d_min    = min(np.nanmin(depth_g), np.nanmin(depth_d))

                im0 = axes[0].imshow(depth_g, cmap='Blues_r',
                                     vmin=d_min, vmax=0, origin='upper')
                plt.colorbar(im0, ax=axes[0], label='Depth (m)')
                axes[0].set_title(f'{res} — gauge WL', fontsize=10)

                im1 = axes[1].imshow(depth_d, cmap='Blues_r',
                                     vmin=d_min, vmax=0, origin='upper')
                plt.colorbar(im1, ax=axes[1], label='Depth (m)')
                axes[1].set_title(f'{res} — DAHITI WL', fontsize=10)

                vmax_b = (np.nanpercentile(np.abs(bias[mask_both_gd]), 95)
                          if np.any(mask_both_gd) else 1.0)
                im2 = axes[2].imshow(np.where(mask_both_gd, bias, np.nan),
                                     cmap='RdBu_r', vmin=-vmax_b, vmax=vmax_b,
                                     origin='upper')
                plt.colorbar(im2, ax=axes[2], label='WL bias DAHITI−gauge (m)')
                axes[2].set_title('Bias (DAHITI − gauge)', fontsize=10)

                fig.suptitle(
                    f'{res}: satellite-only (DAHITI) vs in-situ (gauge) bathymetry',
                    fontsize=12,
                )
                fig.tight_layout()
                out_comp = OUT_DIR / f'bathymetry_{res}_dahiti_vs_gauge.png'
                fig.savefig(out_comp, dpi=150)
                plt.close(fig)

                rmse = (np.sqrt(np.nanmean(bias[mask_both_gd] ** 2))
                        if np.any(mask_both_gd) else np.nan)
                mean_b = (np.nanmean(bias[mask_both_gd])
                          if np.any(mask_both_gd) else np.nan)
                print(f'  DAHITI vs gauge: {out_comp.name}  '
                      f'RMSE = {rmse:.2f} m  mean bias = {mean_b:+.2f} m')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only this phase (default: all)')
    args = parser.parse_args()

    if args.phase is None or args.phase == 1:
        phase1()
    if args.phase is None or args.phase == 2:
        phase2()
    if args.phase is None or args.phase == 3:
        phase3()


if __name__ == '__main__':
    main()
