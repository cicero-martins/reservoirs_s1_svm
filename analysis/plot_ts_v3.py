"""
plot_ts_v3.py  —  Time-series figure for pilot_v3 (full period 2014-2021).

Three layers per panel:
  • Grey dots       — JRC rejected (valid_frac < 0.80)
  • Red line+dots   — JRC valid (used in KGE pairing)
  • Blue/orange pts — SAR raw obs (by pass direction)
  • Open black □    — SAR monthly mean co-located with valid JRC (used in KGE)

22 reservoirs sorted by A/P ascending.
Output: analysis/schwatke_output/ts_v3_fullperiod.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

SAR_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
JRC_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
KGE_CSV = pathlib.Path('analysis/pilot_kge_v3.csv')
OUT_PNG = pathlib.Path('analysis/schwatke_output/ts_v3_fullperiod.png')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
DIR_COLOR      = {'ASCENDING': '#1565C0', 'DESCENDING': '#E65100'}


def load_sar_raw(name):
    p = SAR_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if df.empty:
        return None
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    m, s = df['area_ha'].mean(), df['area_ha'].std()
    if s > 0:
        df = df[np.abs(df['area_ha'] - m) <= 3.0 * s].copy()
    return df[['date', 'area_ha', 'passDirection']].sort_values('date')


def load_jrc(name):
    p = JRC_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None, None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    has_vf   = 'valid_frac' in df.columns
    df_valid = df[df['valid_frac'] >= VALID_FRAC_MIN].copy() if has_vf else df.copy()
    return df, df_valid


def make_pairs(df_sar, df_jrcv):
    if df_sar is None or df_sar.empty or df_jrcv is None or df_jrcv.empty:
        return None
    sar = df_sar.copy()
    sar['ym'] = sar['date'].dt.to_period('M')
    sar_m = sar.groupby('ym')['area_ha'].mean().reset_index()
    sar_m.columns = ['ym', 'sar_ha']

    jrc = df_jrcv.copy()
    jrc['ym'] = jrc['date'].dt.to_period('M')

    merged = pd.merge(sar_m, jrc[['ym', 'jrc_area_ha']], on='ym', how='inner')
    merged['date'] = merged['ym'].dt.to_timestamp(how='start') + pd.offsets.Day(15)
    return merged


# ── Load KGE table ────────────────────────────────────────────────────────────
kge_df = pd.read_csv(KGE_CSV).dropna(subset=['ap_m', 'kge']).sort_values('ap_m').reset_index(drop=True)
names  = kge_df['name'].tolist()

NCOLS = 5
NROWS = -(-len(names) // NCOLS)

fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * 4.2, NROWS * 2.8),
                         sharex=False)
axes_flat = axes.flatten()

for i, name in enumerate(names):
    ax  = axes_flat[i]
    row = kge_df[kge_df['name'] == name].iloc[0]
    ap_m, kge_val = row['ap_m'], row['kge']

    df_sar          = load_sar_raw(name)
    df_jrc, df_jrcv = load_jrc(name)
    pairs           = make_pairs(df_sar, df_jrcv)

    # JRC rejected
    if df_jrc is not None and 'valid_frac' in df_jrc.columns:
        rej = df_jrc[df_jrc['valid_frac'] < VALID_FRAC_MIN]
        if not rej.empty:
            ax.scatter(rej['date'], rej['jrc_area_ha'],
                       s=8, color='#BDBDBD', alpha=0.6, linewidths=0, zorder=1)

    # JRC valid
    if df_jrcv is not None and not df_jrcv.empty:
        ax.plot(df_jrcv['date'], df_jrcv['jrc_area_ha'],
                color='#C62828', lw=1.0, alpha=0.85, zorder=2)
        ax.scatter(df_jrcv['date'], df_jrcv['jrc_area_ha'],
                   s=12, color='#C62828', zorder=3, linewidths=0)

    # SAR raw
    if df_sar is not None and not df_sar.empty:
        for direction, grp in df_sar.groupby('passDirection'):
            ax.scatter(grp['date'], grp['area_ha'],
                       s=6, color=DIR_COLOR.get(direction, '#666'),
                       alpha=0.45, linewidths=0, zorder=3)

    # SAR monthly means used in pairing
    if pairs is not None and not pairs.empty:
        ax.scatter(pairs['date'], pairs['sar_ha'],
                   s=40, facecolors='none', edgecolors='black',
                   linewidths=0.9, zorder=5, marker='s')

    # Formatting
    label     = name.replace('_', ' ')
    kge_color = '#1B5E20' if kge_val >= 0.5 else ('#E65100' if kge_val >= 0.0 else '#B71C1C')
    ax.set_title(f'{label}  A/P={ap_m:.0f} m', fontsize=7.5, fontweight='bold', pad=2)
    ax.text(0.98, 0.97, f'KGE={kge_val:.2f}', transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color=kge_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    ax.set_xlim(pd.Timestamp('2014-07-01'), pd.Timestamp('2022-03-01'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=5.5)
    ax.set_ylabel('Area (ha)', fontsize=5.5, labelpad=2)
    ax.grid(True, alpha=0.2, linewidth=0.4)

for j in range(len(names), len(axes_flat)):
    axes_flat[j].set_visible(False)

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#BDBDBD',
           markersize=5, label='JRC rejected (vf < 0.80)'),
    Line2D([0],[0], color='#C62828', lw=1.2, marker='o',
           markerfacecolor='#C62828', markersize=5, label='JRC valid'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1565C0',
           markersize=5, label='SAR ascending'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E65100',
           markersize=5, label='SAR descending'),
    Line2D([0],[0], marker='s', color='black', markerfacecolor='none',
           markersize=7, lw=0, label='SAR monthly mean (KGE pair)'),
]
fig.legend(handles=legend_handles, loc='lower right',
           bbox_to_anchor=(0.99, 0.01), fontsize=7.5, framealpha=0.9, ncol=2)

fig.suptitle('SAR vs JRC water area — pilot v3, N=22, 2014–2021  (sorted by A/P)',
             fontsize=11, fontweight='bold', y=1.002)
fig.tight_layout(rect=[0, 0.04, 1, 1])
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_PNG}')
