"""
plot_kge_pairs_2019_2020.py

Time-series figure for the 2019-2020 KGE reference window.
Three layers per reservoir:
  • JRC rejected (grey dots) — low valid_frac months, excluded from pairing
  • JRC valid (red markers + line) — used in KGE pairing
  • SAR raw (blue/orange scatter by direction, or green for app-extracted)
      — individual observations after SAR_MIN_FRAC + 3-sigma filter
  • SAR monthly mean (open black squares) — the value actually used in the KGE pair

KGE value annotated per panel.
Reservoirs sorted by A/P ascending (left-to-right, top-to-bottom).

Output: analysis/method_comparison_output/kge_pairs_2019_2020.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy import stats

DATA_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2b')
APP_DIR  = pathlib.Path('raw_data')
KGE_CSV  = pathlib.Path('analysis/pilot_kge_2019_2020.csv')
OUT_PNG  = pathlib.Path('analysis/method_comparison_output/kge_pairs_2019_2020.png')

PERIOD_START   = pd.Timestamp('2019-01-01')
PERIOD_END     = pd.Timestamp('2020-12-31')
VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
EXCLUDE        = {'Elwell', 'Sterkfontein'}
APP_OVERRIDES  = {'Harlan_County': APP_DIR / 'ee-chart_HarlanCounty.csv'}

DIR_COLOR = {'ASCENDING': '#1565C0', 'DESCENDING': '#E65100'}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_sar_raw(name):
    """Raw SAR obs in 2019-2020 after SAR_MIN_FRAC + 3σ filter.
    Returns (df_raw, source) where source='exp' or 'app'."""
    if name in APP_OVERRIDES:
        df = pd.read_csv(APP_OVERRIDES[name])
        df.columns = ['date', 'area_ha'] + [f'_c{i}' for i in range(len(df.columns)-2)]
        df['date']    = pd.to_datetime(df['date'])
        df['area_ha'] = df['area_ha'].astype(str).str.replace(',','',regex=False).astype(float)
        df['passDirection'] = 'APP'
        df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
        return df[['date','area_ha','passDirection']], 'app'

    p = DATA_DIR / f'SAR_area_{name}.csv'
    if not p.exists():
        return None, 'exp'
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['area_ha'] > 0].copy()
    df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
    if df.empty:
        return None, 'exp'
    p99 = df['area_ha'].quantile(0.99)
    df  = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    m, s = df['area_ha'].mean(), df['area_ha'].std()
    if s > 0:
        df = df[np.abs(df['area_ha'] - m) <= 3.0 * s].copy()
    return df[['date','area_ha','passDirection']].sort_values('date'), 'exp'


def load_jrc(name):
    """JRC monthly in 2019-2020. Returns (df_all, df_valid)."""
    p = DATA_DIR / f'JRC_area_{name}.csv'
    if not p.exists():
        return None, None
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    df = df[(df['date'] >= PERIOD_START) & (df['date'] <= PERIOD_END)].copy()
    has_vf = 'valid_frac' in df.columns
    df_valid = df[df['valid_frac'] >= VALID_FRAC_MIN].copy() if has_vf else df.copy()
    return df, df_valid


def make_sar_monthly(df_sar):
    """Monthly mean of raw SAR obs."""
    if df_sar is None or df_sar.empty:
        return None
    df = df_sar.copy()
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index()
    m['date'] = m['ym'].dt.to_timestamp(how='start') + pd.offsets.Day(15)
    return m


def make_pairs(df_sar, df_jrc_valid):
    """Monthly SAR means co-located with valid JRC months (the actual KGE pairs)."""
    if df_sar is None or df_sar.empty or df_jrc_valid is None or df_jrc_valid.empty:
        return None
    sar_m = make_sar_monthly(df_sar)
    if sar_m is None:
        return None
    jrc_m = df_jrc_valid.copy()
    jrc_m['ym'] = jrc_m['date'].dt.to_period('M')
    merged = pd.merge(sar_m[['ym','area_ha']], jrc_m[['ym','jrc_area_ha']], on='ym', how='inner')
    merged['date'] = merged['ym'].dt.to_timestamp(how='start') + pd.offsets.Day(15)
    return merged


# ── Load KGE results for panel annotation ─────────────────────────────────────

kge_df = pd.read_csv(KGE_CSV).dropna(subset=['ap_m','kge'])
kge_df = kge_df[~kge_df['name'].isin(EXCLUDE)].sort_values('ap_m').reset_index(drop=True)
names  = kge_df['name'].tolist()

# ── Figure ────────────────────────────────────────────────────────────────────

NCOLS = 5
NROWS = -(-len(names) // NCOLS)   # ceil division

fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * 4.0, NROWS * 2.6),
                         sharex=False)
axes_flat = axes.flatten()

for i, name in enumerate(names):
    ax  = axes_flat[i]
    row = kge_df[kge_df['name'] == name].iloc[0]
    ap_m, kge_val = row['ap_m'], row['kge']

    df_sar, src      = load_sar_raw(name)
    df_jrc, df_jrcv  = load_jrc(name)
    pairs            = make_pairs(df_sar, df_jrcv)

    # ── JRC rejected ──────────────────────────────────────────────────────
    if df_jrc is not None and 'valid_frac' in df_jrc.columns:
        rej = df_jrc[df_jrc['valid_frac'] < VALID_FRAC_MIN]
        if not rej.empty:
            ax.scatter(rej['date'], rej['jrc_area_ha'],
                       s=10, color='#BDBDBD', alpha=0.6, linewidths=0, zorder=1)

    # ── JRC valid ─────────────────────────────────────────────────────────
    if df_jrcv is not None and not df_jrcv.empty:
        ax.plot(df_jrcv['date'], df_jrcv['jrc_area_ha'],
                color='#C62828', lw=1.2, alpha=0.9, zorder=2)
        ax.scatter(df_jrcv['date'], df_jrcv['jrc_area_ha'],
                   s=18, color='#C62828', zorder=3, linewidths=0)

    # ── SAR raw observations ───────────────────────────────────────────────
    if df_sar is not None and not df_sar.empty:
        if src == 'app':
            ax.scatter(df_sar['date'], df_sar['area_ha'],
                       s=8, color='#2E7D32', alpha=0.55, linewidths=0, zorder=3)
        else:
            for direction, grp in df_sar.groupby('passDirection'):
                ax.scatter(grp['date'], grp['area_ha'],
                           s=8, color=DIR_COLOR.get(direction, '#666'),
                           alpha=0.5, linewidths=0, zorder=3)

    # ── SAR monthly means used in KGE pairing ─────────────────────────────
    if pairs is not None and not pairs.empty:
        ax.scatter(pairs['date'], pairs['area_ha'],
                   s=45, facecolors='none', edgecolors='black',
                   linewidths=1.0, zorder=5, marker='s')

    # ── Axes formatting ───────────────────────────────────────────────────
    label = name.replace('_', ' ')
    kge_color = '#1B5E20' if kge_val >= 0.5 else ('#E65100' if kge_val >= 0.0 else '#B71C1C')
    ax.set_title(f'{label}  A/P={ap_m:.0f} m', fontsize=7.5, fontweight='bold', pad=2)
    ax.text(0.98, 0.97, f'KGE={kge_val:.2f}', transform=ax.transAxes,
            fontsize=7, ha='right', va='top', color=kge_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

    ax.set_xlim(pd.Timestamp('2018-11-01'), pd.Timestamp('2021-02-01'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.tick_params(axis='both', labelsize=5.5)
    ax.set_ylabel('Area (ha)', fontsize=5.5, labelpad=2)
    ax.grid(True, alpha=0.2, linewidth=0.4)

for j in range(len(names), len(axes_flat)):
    axes_flat[j].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#BDBDBD',
           markersize=5, label='JRC rejected (vf < 0.80)'),
    Line2D([0],[0], color='#C62828', lw=1.2, marker='o',
           markerfacecolor='#C62828', markersize=5, label='JRC valid (used in pairing)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1565C0',
           markersize=5, label='SAR asc — raw obs (export)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E65100',
           markersize=5, label='SAR desc — raw obs (export)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2E7D32',
           markersize=5, label='SAR raw obs (app, Harlan County)'),
    Line2D([0],[0], marker='s', color='black', markerfacecolor='none',
           markersize=7, lw=0, label='SAR monthly mean (used in KGE)'),
]
fig.legend(handles=legend_handles, loc='lower right',
           bbox_to_anchor=(0.99, 0.01), fontsize=7.5, framealpha=0.9, ncol=2)

fig.suptitle('SAR vs JRC water area — 2019-2020 reference window  '
             '(pilot v2, N=20, sorted by A/P)',
             fontsize=11, fontweight='bold', y=1.002)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_PNG}')
