"""
plot_wl_pairs_check.py  (ad-hoc visualization only -- NOT wired into the manuscript)

Same gauge/SWOT context as plot_wl_source_grid.py, but overlays the actual
(date, wl_m) pairs from mask_wl_pairs_<name>.csv (Period B) that fed the
level-slicing reconstruction, colour-coded by wl_source, so it's visible
exactly which points -- gauge- or SWOT-sourced -- built the bathymetry.

Output: analysis/schwatke_output/wl_pairs_check.png
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
OUT_DIR = pathlib.Path('analysis/schwatke_output')
SWOT_START = pd.Timestamp('2023-07-28')

ORDER = [
    ('Olivo', 50.7), ('Ancipa', 90.5), ('Nicoletti', 119.7), ('Castello', 126.7),
    ('Garcia', 167.7), ('Arancio', 182.2), ('Rosamarina', 187.4), ('Poma', 190.1),
    ('Pozzillo', 240.5),
]

SOURCE_STYLE = {
    'gauge':   dict(marker='^', color='tab:blue',   ms=7, label='pair: gauge'),
    'swot':    dict(marker='*', color='tab:red',    ms=11, label='pair: SWOT'),
    'boletin': dict(marker='s', color='tab:green',  ms=6, label='pair: boletin'),
    'model':   dict(marker='D', color='tab:orange', ms=6, label='pair: model'),
    'none':    dict(marker='x', color='0.3',        ms=6, label='pair: none (unpaired)'),
}


def load_swot_series(res):
    f = SWOT_DIR / f'{res}_swot.csv'
    if not f.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    swot = m.load_swot(f, res)
    corr = m.CONFIGS.get(res, {}).get('swot_bias_corr', 0.0)
    return (swot + corr).rename('wl_swot') if corr and len(swot) else swot


fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), sharex=False)
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
    gauge_ok = gauge_full[~in_bad]

    for lo, hi in bad:
        ax.axvspan(lo, hi, color='0.9', zorder=0, label='_nolegend_')

    bounds = sorted(b for lo, hi in bad for b in (pd.Timestamp(lo), pd.Timestamp(hi)))
    cuts = ([gauge_ok.index.min()] + bounds + [gauge_ok.index.max()]) if len(gauge_ok) else []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        seg = gauge_ok.loc[lo:hi]
        if len(seg) >= 2:
            ax.plot(seg.index, seg.values, '-', color='tab:blue', lw=1.0, alpha=0.4,
                     label='_nolegend_')

    swot_raw = load_swot_series(name)
    swot_raw = swot_raw.loc[swot_raw.index >= SWOT_START]
    swot_clean = clean_series(swot_raw) if len(swot_raw) > 0 else swot_raw
    if len(swot_clean) > 0:
        ax.plot(swot_clean.index, swot_clean.values, 'o', color='tab:red', ms=2.5,
                mfc='none', mew=0.6, alpha=0.4, label='_nolegend_')

    pairs_f = OUT_DIR / f'mask_wl_pairs_{name}.csv'
    if pairs_f.exists():
        pairs = pd.read_csv(pairs_f, parse_dates=['date'])
        pairs = pairs[(pairs.period == 'B') & pairs.wl_source.notna()]
        seen = set()
        for src, grp in pairs.groupby('wl_source'):
            style = SOURCE_STYLE.get(src, dict(marker='o', color='k', ms=6, label=src))
            ax.plot(grp.date, grp.wl_m, linestyle='none', markeredgecolor='k',
                    markeredgewidth=0.6, zorder=5,
                    label=style['label'] if src not in seen else '_nolegend_',
                    marker=style['marker'], color=style['color'], ms=style['ms'])
            seen.add(src)

    ax.set_title(f'{name} (A/P {ap:.0f})', fontsize=10)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc='best')

fig.suptitle('Period-B mask/WL pairs actually used for reconstruction, by source '
             '(faint lines/dots = full WL context)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / 'wl_pairs_check.png'
fig.savefig(out, dpi=150)
print(f'Saved {out}')
