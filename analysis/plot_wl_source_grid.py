"""
plot_wl_source_grid.py

Publication figure: gauge vs SWOT water level, SWOT-era only (2023-07 onward),
for all 9 reservoirs in one grid, ordered by A/P. Complements (does not
replace) Table "wlrmse" in Results -- gives the reader the actual shape of
agreement/divergence behind each RMSE/KGE number, not just the summary
statistic.

SWOT is shown AFTER Paper-1-style outlier removal (see
wl_gauge_swot_validation.py::clean_series); removed raw points are also
plotted, faded, so the cleaning itself is visible rather than silently
applied. Gauge windows with a confirmed, documented malfunction
(CONFIGS[...]['gauge_bad_window']) are shaded and excluded from the RMSE/KGE
computation (still drawn, faded, for context).

Output: manuscript_paper2/figures/wl_source_grid.png
"""
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m
from wl_gauge_swot_validation import clean_series, bad_windows_for

SWOT_DIR = pathlib.Path('validation_data/SWOT')
SWOT_START = pd.Timestamp('2023-07-28')

# (name, ap_m) ordered by A/P, matching Table tab:capacity / tab:wlrmse
ORDER = [
    ('Olivo', 50.7), ('Ancipa', 90.5), ('Nicoletti', 119.7), ('Castello', 126.7),
    ('Garcia', 167.7), ('Arancio', 182.2), ('Rosamarina', 187.4), ('Poma', 190.1),
    ('Pozzillo', 240.5),
]

METRICS = pd.read_csv('analysis/schwatke_output/fullrs/wl_gauge_swot_clean.csv').set_index('reservoir')


def load_swot_series(res):
    """Datum-corrected (swot_bias_corr) -- matches the RMSE/KGE in METRICS below,
    which are computed post-correction (wl_gauge_swot_validation.py)."""
    f = SWOT_DIR / f'{res}_swot.csv'
    if not f.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    swot = m.load_swot(f)
    corr = m.CONFIGS.get(res, {}).get('swot_bias_corr', 0.0)
    return (swot + corr).rename('wl_swot') if corr and len(swot) else swot


fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=False)
for ax, (name, ap) in zip(axes.flat, ORDER):
    cfg = m.CONFIGS[name]
    try:
        gauge_full = m.load_gauge(cfg)
    except Exception:
        gauge_full = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    gauge_full = gauge_full.loc[gauge_full.index >= SWOT_START]

    bad = bad_windows_for(name)
    in_bad = pd.Series(False, index=gauge_full.index)
    for lo, hi in bad:
        in_bad |= (gauge_full.index >= lo) & (gauge_full.index <= hi)
    gauge_ok, gauge_bad_pts = gauge_full[~in_bad], gauge_full[in_bad]

    swot_raw = load_swot_series(name)
    swot_raw = swot_raw.loc[swot_raw.index >= SWOT_START]
    swot_clean = clean_series(swot_raw) if len(swot_raw) > 0 else swot_raw
    swot_removed = swot_raw.loc[swot_raw.index.difference(swot_clean.index)]

    for lo, hi in bad:
        ax.axvspan(lo, hi, color='0.85', zorder=0, label='_nolegend_')

    if len(gauge_ok) > 0:
        ax.plot(gauge_ok.index, gauge_ok.values, '-', color='tab:blue', lw=1.1,
                alpha=0.8, label='Gauge')
    if len(gauge_bad_pts) > 0:
        ax.plot(gauge_bad_pts.index, gauge_bad_pts.values, '-', color='tab:blue',
                lw=1.1, alpha=0.3, label='Gauge (malfunction window)')
    if len(swot_clean) > 0:
        ax.plot(swot_clean.index, swot_clean.values, 'o', color='tab:red', ms=3.5,
                mfc='none', mew=1.0, label='SWOT')
    if len(swot_removed) > 0:
        ax.plot(swot_removed.index, swot_removed.values, 'x', color='0.6', ms=4,
                mew=1.0, label='SWOT (removed outlier)')

    title = f'{name} (A/P {ap:.0f})'
    if name in METRICS.index:
        row = METRICS.loc[name]
        if pd.notna(row['rmse_m']):
            title += f"\nRMSE {row['rmse_m']:.2f} m · KGE {row['kge']:.2f}"
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, loc='best')

fig.suptitle('Gauge vs SWOT water level, SWOT era (2023-07 onward) — cleaned', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = pathlib.Path('manuscript_paper2/figures/wl_source_grid.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f'Saved {out}')
