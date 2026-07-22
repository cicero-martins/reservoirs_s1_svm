"""
diag_wl_source_gap.py  (diagnostic, not part of the main pipeline)

Investigates the "staircase" artifact visible in the Period-B 3D bathymetry:
large single-step gaps between consecutive stacked water levels, caused by
area-percentile date stratification inside a short, non-linear drought-refill
window (see 2026-07-21 audit). Produces two things per reservoir:

1. A time-series figure: daily gauge WL (line), SWOT WL (markers) over the
   Period-B export window, with the water level ACTUALLY assigned to each
   chosen SAR mask date highlighted and colour-coded by source.
   -> analysis/schwatke_output/diag_wl_source/wl_source_{res}.png

2. Two independent Period-B DEMs per reservoir with SWOT coverage: one built
   using ONLY the gauge (ignoring any bad-window override), one built using
   ONLY SWOT (interpolated for every date, not just the bad-window dates).
   Reuses the same reconstruction as the main pipeline (_dem_recon.build_dem
   via schwatke_bathymetry_3d.build_dem_from_arrays), so the comparison is
   apples-to-apples with the production DEMs.
   -> analysis/schwatke_output/dem_{res}_B_gaugeonly.tif
   -> analysis/schwatke_output/dem_{res}_B_swotonly.tif
   -> analysis/schwatke_output/diag_wl_source/dem_compare_{res}.png
"""
import sys, json, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

import rasterio

REPO = pathlib.Path('.')
OUT = m.OUT_DIR / 'diag_wl_source'
OUT.mkdir(parents=True, exist_ok=True)
SWOT_DIR = REPO / 'validation_data' / 'SWOT'

# Period-B export windows (from export_windowed_masks.py RESERVOIRS), used only
# to bound the diagnostic time-series plot -- not re-fetched from GEE here.
WINDOWS = {
    'Poma':       ('2025-12-15', '2026-05-10'),
    'Rosamarina': ('2025-09-15', '2026-05-15'),
    'Castello':   ('2025-09-15', '2026-04-25'),
    'Olivo':      ('2025-09-25', '2026-04-05'),
    'Arancio':    ('2025-09-01', '2026-04-15'),
    'Ancipa':     ('2024-11-10', '2025-03-01'),
    'Nicoletti':  ('2025-10-15', '2026-03-20'),
    'Pozzillo':   ('2025-10-01', '2026-03-25'),
    'Garcia':     ('2025-08-06', '2026-05-31'),
}

SOURCE_COLOR = {'gauge': 'tab:blue', 'swot': 'tab:red', 'boletin': 'tab:green',
                'model': 'tab:purple', 'none': 'grey'}


def load_swot_series(res):
    f = SWOT_DIR / f'{res}_swot.csv'
    if not f.exists():
        return pd.Series(dtype=float, name='wl_swot', index=pd.DatetimeIndex([]))
    return m.load_swot(f)


def diag_plot(res):
    cfg = m.CONFIGS[res]
    win_lo, win_hi = pd.Timestamp(WINDOWS[res][0]), pd.Timestamp(WINDOWS[res][1])
    pad = pd.Timedelta(days=20)

    try:
        gauge = m.load_gauge(cfg)
    except Exception:
        gauge = pd.Series(dtype=float)
    swot = load_swot_series(res)

    gauge_win = gauge.loc[(gauge.index >= win_lo - pad) & (gauge.index <= win_hi + pad)]
    swot_win = swot.loc[(swot.index >= win_lo - pad) & (swot.index <= win_hi + pad)]

    pairs_f = m.OUT_DIR / f'mask_wl_pairs_{res}.csv'
    pairs = pd.read_csv(pairs_f, parse_dates=['date'])
    sub = pairs[(pairs['period'] == 'B')].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    if len(gauge_win) > 0:
        ax.plot(gauge_win.index, gauge_win.values, '-', color='tab:blue', lw=1.2,
                alpha=0.6, label=f'Gauge (daily, n={len(gauge_win)})')
    if len(swot_win) > 0:
        ax.plot(swot_win.index, swot_win.values, 'o', color='tab:red', ms=6,
                mfc='none', mew=1.6, label=f'SWOT (n={len(swot_win)})')

    for src, grp in sub.groupby('wl_source'):
        ax.scatter(grp['date'], grp['wl_m'], s=90, marker='*',
                   color=SOURCE_COLOR.get(src, 'black'), edgecolor='black',
                   linewidth=0.6, zorder=5, label=f'Applied ({src}, n={len(grp)})')

    ax.axvline(win_lo, color='grey', ls=':', lw=1)
    ax.axvline(win_hi, color='grey', ls=':', lw=1)
    ax.set_ylabel('Water level (m)')
    ax.set_title(f'{res} — Period-B water-level sources and applied values')
    ax.legend(loc='best', fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUT / f'wl_source_{res}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  {res}: saved {out.name}  (gauge_win={len(gauge_win)}, swot_win={len(swot_win)}, applied={len(sub)})')


def build_forced_dem(res, source):
    """Build a Period-B DEM using ONLY `source` ('gauge' or 'swot') for every
    mask date, ignoring the production pipeline's bad-window override logic."""
    cfg = m.CONFIGS[res]
    pairs_f = m.OUT_DIR / f'mask_wl_pairs_{res}.csv'
    pairs = pd.read_csv(pairs_f, parse_dates=['date'])
    sub = pairs[pairs['period'] == 'B'].copy()

    if source == 'gauge':
        try:
            series = m.load_gauge(cfg)
        except Exception:
            return None, None
    else:
        series = load_swot_series(res)
        if len(series) == 0:
            return None, None

    raw_arrays, wls = [], []
    for _, row in sub.iterrows():
        dt = row['date']
        val = m.interp_wl(series, dt, m.MAX_DT)
        if np.isnan(val):
            continue
        date_str = dt.strftime('%Y-%m-%d')
        tif_path = m.MASK_DIR / f'mask_{res}_{date_str}.tif'
        if not tif_path.exists():
            continue
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            meta = src.meta.copy()
        raw_arrays.append(arr)
        wls.append(val)

    if len(raw_arrays) < 3:
        return None, None

    order = np.argsort(wls)
    arrs_s = [raw_arrays[i] for i in order]
    wls_s = [wls[i] for i in order]
    dem = m.build_dem_from_arrays(arrs_s, wls_s)

    out_meta = meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    out_tif = m.OUT_DIR / f'dem_{res}_B_{"gaugeonly" if source == "gauge" else "swotonly"}.tif'
    with rasterio.open(out_tif, 'w', **out_meta) as dst:
        dst.write(dem[np.newaxis, :, :])
    return dem, wls_s


def compare_dems(res):
    dem_g, wl_g = build_forced_dem(res, 'gauge')
    dem_s, wl_s = build_forced_dem(res, 'swot')
    if dem_g is None and dem_s is None:
        print(f'  {res}: neither gauge-only nor swot-only DEM buildable (< 3 masks with WL), skipping')
        return
    if dem_s is None:
        print(f'  {res}: no SWOT data available for a swot-only DEM, skipping comparison')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def _show(ax, dem, wls, title):
        if dem is None:
            ax.set_title(f'{title}\n(unavailable)')
            ax.axis('off')
            return
        im = ax.imshow(dem, cmap='viridis')
        floor, top = np.nanmin(dem), np.nanmax(dem)
        ax.set_title(f'{title}\nWL {min(wls):.1f}-{max(wls):.1f} m, n={len(wls)}\nfloor={floor:.1f} top={top:.1f}')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Elevation (m)')

    _show(axes[0], dem_g, wl_g, 'Gauge-only DEM')
    _show(axes[1], dem_s, wl_s, 'SWOT-only DEM')

    if dem_g is not None and dem_s is not None and dem_g.shape == dem_s.shape:
        diff = dem_s - dem_g
        im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-5, vmax=5)
        valid = ~np.isnan(diff)
        mean_d = np.nanmean(diff) if valid.any() else np.nan
        axes[2].set_title(f'SWOT - Gauge\nmean diff={mean_d:.2f} m, n_valid={int(valid.sum())}')
        plt.colorbar(im, ax=axes[2], fraction=0.046, label='Elevation diff (m)')
    else:
        axes[2].set_title('Shapes differ, no diff map')
        axes[2].axis('off')

    fig.suptitle(f'{res} — Period-B DEM: gauge-only vs SWOT-only')
    fig.tight_layout()
    out = OUT / f'dem_compare_{res}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    g_range = f'{min(wl_g):.1f}-{max(wl_g):.1f}' if wl_g else 'n/a'
    s_range = f'{min(wl_s):.1f}-{max(wl_s):.1f}' if wl_s else 'n/a'
    print(f'  {res}: saved {out.name}  (gauge WL {g_range} n={len(wl_g) if wl_g else 0}, '
          f'swot WL {s_range} n={len(wl_s) if wl_s else 0})')


if __name__ == '__main__':
    print('=== Diagnostic: WL source time series (Period B) ===')
    for res in m.CONFIGS:
        diag_plot(res)

    print('\n=== Diagnostic: gauge-only vs SWOT-only DEM ===')
    for res in m.CONFIGS:
        compare_dems(res)

    print(f'\nAll figures saved to {OUT}')
