"""
planet_bathymetry.py  (cross-sensor validation — PlanetScope optical bathymetry)

Reconstruct reservoir bathymetry from PlanetScope 3 m NDWI water masks using the
SAME waterline-stacking method as the SAR DEM, then cross-validate against the
Sentinel-1 reconstruction (and the survey curves). Because the method is agnostic
to the mask source, agreement between the optical and SAR DEMs — whose error modes
are orthogonal — is strong evidence neither is dominated by sensor-specific error.
Crucially, this gives Ancipa and Pozzillo (no modern field survey) an independent
near-truth.

Inputs : raw_data/GEE_SicilyPlanetMasks/mask_Planet_<Site>_<YYYY-MM-DD>.tif (3 m, EPSG:32633)
         water level from the in-situ gauge (preferred) or SWOT.
Outputs: analysis/schwatke_output/planet/ dem_<Site>_Planet.tif, planet_vs_sar.csv,
         planet_vs_sar.png
"""

import sys, glob, re, pathlib, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))          # analysis/ for _dem_recon
sys.path.insert(0, str((pathlib.Path(__file__).resolve().parent.parent / 'tool')))
import bathymetry as bt          # load_dem (SAR), aev, design_curve, updated_curve
from _dem_recon import build_dem  # shared waterline-stacking reconstruction (bathtub)
import schwatke_bathymetry_3d as m3d  # CONFIGS[...]['swot_bias_corr'] -- single source of truth

REPO   = pathlib.Path('.')
MASKS  = REPO / 'raw_data' / 'GEE_SicilyPlanetMasks'
GAUGE  = REPO / 'analysis' / 'schwatke_output' / 'gauge_downloads'
SWOT   = REPO / 'validation_data' / 'SWOT'
OUT    = REPO / 'analysis' / 'schwatke_output' / 'planet'
OUT.mkdir(parents=True, exist_ok=True)
MAX_DT = 7   # days for mask<->WL pairing

SITES = {
    'Ancipa':     dict(gauge='ancipa_livello_secca.csv', ap=90.5),
    # force_swot=True added 2026-07-24: Pozzillo's gauge under-represents its true
    # water-level range throughout the record (independently confirmed via SAR area
    # correlating more strongly with SWOT than with the gauge, see
    # schwatke_bathymetry_3d.py CONFIGS['Pozzillo'] and Results sec:res_inputvalid) --
    # the old IQR-based stuck-sensor detection in choose_wl() never triggers for
    # Pozzillo since its gauge isn't frozen, just compressed, so this needs an
    # explicit override rather than relying on that heuristic.
    'Pozzillo':   dict(gauge='pozzillo_wl.csv',          ap=240.5, force_swot=True),
    'Rosamarina': dict(gauge='rosamarina_wl.csv',        ap=187.4),
    'Poma':       dict(gauge='poma_wl.csv',              ap=190.1),
}


# ── Water level (gauge preferred, SWOT fallback) ──────────────────────────────
def _wl(df, dcol, wcol):
    df = df[[dcol, wcol]].copy(); df.columns = ['date', 'wl']
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    return df.dropna().groupby('date', as_index=False).wl.mean().sort_values('date')

def _remove_global(s, threshold=2.0):
    mean, sd = s.mean(), s.std()
    return s[np.abs(s - mean) <= threshold * sd]

def _remove_local(s, window=5, threshold=1.5):
    arr, idx = s.values.copy(), s.index.tolist()
    keep, half = [], window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        mean, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - mean) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]

def load_wl(site, cfg):
    g = pd.read_csv(GAUGE / cfg['gauge']); g.columns = [c.strip().lower() for c in g.columns]
    gauge = _wl(g, next(c for c in g.columns if 'time' in c or 'date' in c),
                   next(c for c in g.columns if 'wl' in c or 'quota' in c or 'value' in c))
    sp = SWOT / f'{site}_swot.csv'
    if not sp.exists():
        return gauge, None
    swot = _wl(pd.read_csv(sp), 'datetime', 'wse')
    # Paper 1's outlier-cleaning pipeline (same as schwatke_bathymetry_3d.load_swot)
    s = swot.set_index('date')['wl']
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 10, 1.5)
    # gauge-vs-SWOT datum offset (see wl_gauge_swot_validation.py / CONFIGS comments
    # in schwatke_bathymetry_3d.py) -- otherwise a per-site constant reference-frame
    # mismatch gets scored as pixel-elevation "error" against the gauge-calibrated SAR DEM.
    corr = m3d.CONFIGS.get(site, {}).get('swot_bias_corr', 0.0)
    if corr:
        s = s + corr
    swot = s.rename('wl').rename_axis('date').reset_index()
    return gauge, swot

def pair_series(dates, df):
    """Nearest WL (within MAX_DT) for each mask date; None where no WL is close."""
    if df is None or not len(df):
        return [None] * len(dates)
    out = []
    for d in dates:
        dt = (df.date - d).abs(); i = dt.idxmin()
        out.append(float(df.wl[i]) if dt[i] <= pd.Timedelta(f'{MAX_DT}D') else None)
    return out

def _iqr(vals):
    """Robust vertical-signal measure: interquartile range. Unlike the full span it is not
    inflated by a couple of far-apart outlier dates (e.g. Rosamarina's 2022/2023 masks make
    the flat 2024--25 gauge *look* like an 18 m span), and not deflated by SWOT spikes."""
    v = [x for x in vals if x is not None]
    return float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) >= 4 else -1.0

def choose_wl(dates, gauge, swot, force_swot=False):
    """Pick a SINGLE water-level source per site (no datum mixing). Prefer the in-situ gauge,
    but fall back to SWOT when the gauge is degenerate over the mask window — flat to
    < 0.5 m IQR, i.e. a stuck/dead sensor. Rosamarina's gauge reads 145.72 m to the cm for
    ~10 months in 2024--25 (IQR ~0.01 m) while SWOT and the optical masks both show ~17 m of
    drawdown, so SWOT recovers the hypsometry the dead gauge lost; every well-behaved gauge
    (Poma, Ancipa) is kept, avoiding a needless SWOT datum shift. `force_swot` overrides this
    for a reservoir whose gauge is known-compressed rather than stuck (Pozzillo), since the
    IQR heuristic only catches a frozen gauge, not a merely narrow-but-varying one."""
    g, s = pair_series(dates, gauge), pair_series(dates, swot)
    use_swot = force_swot or (_iqr(g) < 0.5 and _iqr(s) > 1.0)
    return (s, 'SWOT') if use_swot else (g, 'gauge')


# ── Load + align masks of a site onto one grid ────────────────────────────────
def load_site_masks(site):
    fs = sorted(glob.glob(str(MASKS / f'mask_Planet_{site}_*.tif')))
    dates = [re.search(r'_(\d{4}-\d{2}-\d{2})', f).group(1) for f in fs]
    with rasterio.open(fs[0]) as s0:
        ref_tf, ref_crs, H, W = s0.transform, s0.crs, s0.height, s0.width
    arrs = []
    for f in fs:
        with rasterio.open(f) as s:
            if (s.height, s.width) == (H, W) and s.transform == ref_tf:
                arrs.append(s.read(1))
            else:                       # reproject onto the reference grid
                dst = np.zeros((H, W), np.uint8)
                reproject(s.read(1), dst, src_transform=s.transform, src_crs=s.crs,
                          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
                arrs.append(dst)
    return arrs, [pd.Timestamp(d) for d in dates], ref_tf, ref_crs


# DEM reconstruction (bathtub level-slicing) is in _dem_recon.build_dem — shared with the SAR pipeline.


def aev(elev, mask, levels, pixel_ha):
    areas = np.array([np.sum((elev < h) & mask) * pixel_ha for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols


def vol_exact(elev, mask, lo, hi, pixel_ha):
    """Volume in [lo,hi] (Mm3), exact per-pixel water-column sum -- the 0.5 m-step
    trapezoidal aev() above underestimates volume by up to ~12% for these sparse
    (~10-mask) level-slice DEMs (see consolidate_bathymetry.py::vol_exact for the
    full diagnosis); use this exact sum for any reported total-volume scalar."""
    col = np.clip(hi - elev, 0.0, hi - lo)
    return float(np.sum(col[mask]) * pixel_ha * 1e4 / 1e6)


rows, panels = [], {}
for site, cfg in SITES.items():
    arrs, dates, tf, crs = load_site_masks(site)
    pix_m = abs(tf.a); pix_ha = pix_m * pix_m / 1e4
    gauge, swot = load_wl(site, cfg)
    chosen, wl_src = choose_wl(dates, gauge, swot, force_swot=cfg.get('force_swot', False))
    wls, keep, srcs = [], [], []
    for a, w in zip(arrs, chosen):
        if w is not None:
            wls.append(w); keep.append(a); srcs.append(wl_src)
    if len(keep) < 4:
        print(f'{site}: only {len(keep)} masks with WL, skipping'); continue

    dem = build_dem(keep, wls, pix_m)
    meta = dict(driver='GTiff', height=dem.shape[0], width=dem.shape[1], count=1,
                dtype='float32', crs=crs, transform=tf, nodata=np.nan)
    with rasterio.open(OUT / f'dem_{site}_Planet.tif', 'w', **meta) as dst:
        dst.write(dem, 1)

    pmask = np.isfinite(dem)
    pfloor, ptop = float(dem[pmask].min()), float(dem[pmask].max())

    # ── compare to SAR DEM_B: reproject Planet DEM onto the SAR grid ──────────
    sar = bt.load_dem(site, 'B')
    rec = dict(reservoir=site, ap_m=cfg['ap'], n_masks=len(keep),
               wl_gauge=srcs.count('gauge'), wl_swot=srcs.count('SWOT'),
               planet_floor=round(pfloor, 1), planet_max=round(ptop, 1))
    if sar is not None:
        sar_dem, sar_tf, sar_crs = sar['arr'], sar['transform'], None
        with rasterio.open(bt.dem_file(site, 'B')) as s:
            sar_crs = s.crs
        pl_on_sar = np.full(sar_dem.shape, np.nan, np.float32)
        reproject(dem, pl_on_sar, src_transform=tf, src_crs=crs,
                  dst_transform=sar_tf, dst_crs=sar_crs, resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        lo = max(pfloor, sar['floor']) + 1.0; hi = min(ptop, sar['top'])
        both = np.isfinite(pl_on_sar) & np.isfinite(sar_dem) & (sar_dem >= lo) & (sar_dem <= hi)
        if both.sum() > 50:
            diff = pl_on_sar[both] - sar_dem[both]   # Planet - SAR
            rec['n_px'] = int(both.sum())
            rec['planet_minus_sar_bias_m'] = round(float(diff.mean()), 2)
            rec['planet_vs_sar_rmse_m'] = round(float(np.sqrt(np.mean(diff**2))), 2)
        # AEV over common band
        levels = np.arange(lo, hi + 1e-6, 0.5)
        a_pl, v_pl = aev(dem, pmask, levels, pix_ha)
        a_sar, v_sar = aev(sar_dem, sar['mask'], levels, bt.PIXEL_HA)
        rec['planet_vol_Mm3'] = round(vol_exact(dem, pmask, lo, hi, pix_ha), 2)
        rec['sar_vol_Mm3'] = round(vol_exact(sar_dem, sar['mask'], lo, hi, bt.PIXEL_HA), 2)
        panels[site] = dict(levels=levels, a_pl=a_pl, a_sar=a_sar, v_pl=v_pl, v_sar=v_sar,
                            ap=cfg['ap'], lo=lo, hi=hi)
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'planet_vs_sar.csv', index=False)
pd.set_option('display.width', 200, 'display.max_columns', 30)
print(df.to_string(index=False))
if 'planet_vs_sar_rmse_m' in df:
    v = df.dropna(subset=['planet_vs_sar_rmse_m'])
    print(f"\nPlanetScope vs SAR DEM (observable band): mean pixel RMSE "
          f"{v.planet_vs_sar_rmse_m.mean():.2f} m, mean bias {v.planet_minus_sar_bias_m.mean():+.2f} m; "
          f"band volume Planet vs SAR: "
          f"{', '.join(f'{r.reservoir} {r.planet_vol_Mm3}/{r.sar_vol_Mm3}' for _,r in v.iterrows())} Mm³.")
print(f"\nSaved: {OUT/'planet_vs_sar.csv'}")

# ── Figure: AEV (area+volume) Planet vs SAR per site ──────────────────────────
if panels:
    n = len(panels)
    fig = plt.figure(figsize=(4.6 * n, 4.4))
    gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.32)
    fig.suptitle('Cross-sensor check — PlanetScope (optical, 3 m) vs Sentinel-1 (SAR) bathymetry',
                 fontsize=12, fontweight='bold')
    for i, (site, P) in enumerate(panels.items()):
        ax = fig.add_subplot(gs[i])
        ax.plot(P['a_sar'], P['levels'], color='#6a1b9a', lw=2.2, label='SAR DEM')
        ax.plot(P['a_pl'], P['levels'], color='#2e7d32', lw=2.2, ls='--', label='PlanetScope DEM')
        r = df[df.reservoir == site].iloc[0]
        sub = f"vol {r.get('planet_vol_Mm3','?')}/{r.get('sar_vol_Mm3','?')} Mm³"
        if pd.notna(r.get('planet_vs_sar_rmse_m')):
            sub += f" · RMSE {r.planet_vs_sar_rmse_m} m"
        ax.set_title(f"{site} (A/P {P['ap']:.0f})\n{sub}", fontsize=9)
        ax.set_xlabel('Area (ha)'); ax.set_ylabel('Water level (m ASL)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
    fig.subplots_adjust(top=0.85)
    fig.savefig(OUT / 'planet_vs_sar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUT/'planet_vs_sar.png'}")
