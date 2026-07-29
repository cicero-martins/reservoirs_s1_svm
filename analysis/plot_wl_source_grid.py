"""
plot_wl_source_grid.py

Publication figure: gauge vs SWOT water level, SWOT-era only (2023-07 onward),
for all 9 reservoirs in one grid, ordered by A/P. Complements (does not
replace) Table "wlrmse" in Results -- gives the reader the actual shape of
agreement/divergence behind each RMSE/KGE number, not just the summary
statistic.

SWOT is shown after Paper-1-style outlier removal (see
wl_gauge_swot_validation.py::clean_series) and datum correction
(CONFIGS[...]['swot_bias_corr']); the raw, pre-correction series is also
plotted faintly for context, and the applied correction is annotated in each
panel's corner. Gauge windows with a confirmed, documented malfunction
(CONFIGS[...]['gauge_bad_window']) are shaded and excluded from the RMSE/KGE
computation (still drawn, as faded markers only -- no connecting line, so
gaps in the malfunction window aren't bridged by a misleading straight
segment).

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
    swot = m.load_swot(f, res)
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

    # Raw (pre-datum-correction) SWOT, shown faintly for context alongside the
    # corrected series actually used for RMSE/KGE.
    f = SWOT_DIR / f'{name}_swot.csv'
    swot_uncorrected = m.load_swot(f, name) if f.exists() else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    swot_uncorrected = swot_uncorrected.loc[swot_uncorrected.index >= SWOT_START]
    corr = m.CONFIGS.get(name, {}).get('swot_bias_corr', 0.0)

    swot_raw = load_swot_series(name)
    swot_raw = swot_raw.loc[swot_raw.index >= SWOT_START]
    swot_clean = clean_series(swot_raw) if len(swot_raw) > 0 else swot_raw

    for lo, hi in bad:
        ax.axvspan(lo, hi, color='0.85', zorder=0, label='_nolegend_')

    if len(gauge_ok) > 0:
        # Plot as separate contiguous segments split at each bad-window boundary --
        # gauge_ok has those windows removed entirely, so a single connected line
        # would otherwise bridge the gap with a straight segment that isn't real data.
        bounds = sorted(b for lo, hi in bad for b in (pd.Timestamp(lo), pd.Timestamp(hi)))
        cuts = [gauge_ok.index.min()] + bounds + [gauge_ok.index.max()]
        first = True
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            seg = gauge_ok.loc[lo:hi]
            if len(seg) < 2:
                continue
            ax.plot(seg.index, seg.values, '-', color='tab:blue', lw=1.1, alpha=0.8,
                     label='Gauge' if first else '_nolegend_')
            first = False
    if len(gauge_bad_pts) > 0:
        # Markers only (no connecting line): a connected line would bridge gaps in the
        # malfunction window with a straight segment that isn't real data.
        ax.plot(gauge_bad_pts.index, gauge_bad_pts.values, '.', color='tab:blue',
                ms=2.5, alpha=0.3, label='Gauge (malfunction window)')
    if len(swot_uncorrected) > 0:
        ax.plot(swot_uncorrected.index, swot_uncorrected.values, '.', color='tab:red',
                ms=2.5, alpha=0.25, label='SWOT (raw, pre-datum-correction)')
    if len(swot_clean) > 0:
        ax.plot(swot_clean.index, swot_clean.values, 'o', color='tab:red', ms=3.5,
                mfc='none', mew=1.0, label='SWOT')

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
    if corr:
        ax.text(0.02, 0.02, f'datum corr. {corr:+.2f} m', transform=ax.transAxes,
                fontsize=6.5, color='0.4', ha='left', va='bottom')

fig.suptitle('Gauge vs SWOT water level, SWOT era (2023-07 onward) — cleaned', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = pathlib.Path('manuscript_paper2/figures/wl_source_grid.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f'Saved {out}')
