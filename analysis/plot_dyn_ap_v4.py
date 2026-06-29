"""
plot_dyn_ap_v4.py  -  Dynamic A/P time series for pilot v4 (32 reservoirs).

Each panel:
  - Teal dots  : dynamic A/P per SAR acquisition (area / perimeter of classified polygon)
  - Teal line  : LOWESS trend (window=60 days)
  - Dashed red : static A/P reference (from JRC max_extent polygon)

Sorted by static A/P ascending (same order as ts_v4_fullperiod.png).
Output: analysis/schwatke_output/dyn_ap_v4.png
"""

import pathlib
import re as _re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


def _lowess(dates, values, window_days=60, bandwidth=20):
    smoothed = []
    for t0 in dates:
        diff_d = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask   = diff_d <= window_days
        w      = np.exp(-(diff_d[mask] / bandwidth) ** 2)
        smoothed.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(smoothed)


SAR_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
KGE_CSV = pathlib.Path('analysis/pilot_kge_v4.csv')
OUT_PNG = pathlib.Path('analysis/schwatke_output/dyn_ap_v4.png')

SAR_MIN_FRAC = 0.02
AREA_MIN     = {'Saint_Cassien': 200}
DYN_COLOR    = '#00796B'


def load_dyn_ap(name):
    p = SAR_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if 'ap_m_dynamic' not in df.columns:
        return None
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    if df.empty:
        return None
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    return df[['date', 'ap_m_dynamic']].dropna().reset_index(drop=True)


kge_df = pd.read_csv(KGE_CSV).dropna(subset=['ap_m', 'kge']).sort_values('ap_m').reset_index(drop=True)
names  = kge_df['name'].tolist()

NCOLS = 5
NROWS = -(-len(names) // NCOLS)

fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * 4.2, NROWS * 2.5),
                         sharex=False)
axes_flat = axes.flatten()

for i, name in enumerate(names):
    ax  = axes_flat[i]
    row = kge_df[kge_df['name'] == name].iloc[0]
    ap_m    = row['ap_m']
    ap_dyn  = row.get('ap_m_dynamic', np.nan)
    kge_val = row['kge']

    df = load_dyn_ap(name)

    if df is not None and not df.empty:
        ax.scatter(df['date'], df['ap_m_dynamic'],
                   s=4, color=DYN_COLOR, alpha=0.35, linewidths=0, zorder=2)
        if len(df) >= 5:
            trend = _lowess(df['date'], df['ap_m_dynamic'])
            ax.plot(df['date'], trend,
                    color=DYN_COLOR, lw=1.2, alpha=0.85, zorder=3)

    # Static A/P reference
    if not np.isnan(ap_m):
        ax.axhline(ap_m, color='#C62828', lw=1.0, ls='--', alpha=0.7, zorder=1)

    label     = name.replace('_', ' ')
    kge_color = '#1B5E20' if kge_val >= 0.5 else ('#E65100' if kge_val >= 0.0 else '#B71C1C')
    dyn_label = f'{ap_dyn:.0f}' if not np.isnan(ap_dyn) else '?'
    ax.set_title(f'{label}  A/P={ap_m:.0f} (dyn~{dyn_label})', fontsize=6.5, fontweight='bold', pad=2)
    ax.text(0.02, 0.97, f'KGE={kge_val:.2f}', transform=ax.transAxes,
            fontsize=6.5, ha='left', va='top', color=kge_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    # y-axis: 0 to 1.4 × static A/P (or max observed + margin)
    if df is not None and not df.empty:
        ymax = max(df['ap_m_dynamic'].quantile(0.97) * 1.25,
                   ap_m * 1.2 if not np.isnan(ap_m) else 10)
    else:
        ymax = ap_m * 1.4 if not np.isnan(ap_m) else 10
    ax.set_ylim(0, ymax)
    ax.set_ylabel('A/P (m)', fontsize=5.5, labelpad=2)
    ax.set_xlim(pd.Timestamp('2014-07-01'), pd.Timestamp('2022-03-01'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(True, alpha=0.2, linewidth=0.4)

for j in range(len(names), len(axes_flat)):
    axes_flat[j].set_visible(False)

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=DYN_COLOR,
           markersize=5, label='Dynamic A/P (per acquisition)'),
    Line2D([0],[0], color=DYN_COLOR, lw=1.5, label='Dynamic A/P LOWESS trend'),
    Line2D([0],[0], color='#C62828', lw=1.0, ls='--', label='Static A/P (JRC max_extent)'),
]
fig.legend(handles=legend_handles, loc='lower right',
           bbox_to_anchor=(0.99, 0.01), fontsize=7.5, framealpha=0.9)

fig.suptitle('Dynamic A/P time series  -  pilot v4, N=32, 2014-2021  (sorted by static A/P asc)',
             fontsize=10, fontweight='bold', y=1.002)
fig.tight_layout(rect=[0, 0.03, 1, 1])
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_PNG}')
