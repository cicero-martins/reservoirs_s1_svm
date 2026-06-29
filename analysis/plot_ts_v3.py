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


# ── App-equivalent cleaning pipeline (mirrors original.js) ───────────────────
def _remove_global(s, threshold=2.0):
    m, sd = s.mean(), s.std()
    return s[np.abs(s - m) <= threshold * sd]

def _remove_local(s, window=5, threshold=1.5):
    arr = s.values.copy()
    idx = s.index.tolist()
    keep = []
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        m, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - m) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]

def _lowess(dates, values, window_days=20, bandwidth=7):
    smoothed = []
    for t0, _ in zip(dates, values):
        diff_d = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask   = diff_d <= window_days
        w      = np.exp(-(diff_d[mask] / bandwidth) ** 2)
        smoothed.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(smoothed)

def clean_and_smooth(df, col='area_ha'):
    """Mirror of original.js: removeOutliers(2) + 3x local + lowessSmoothing(20,7)."""
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 5,  1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s  = df.loc[s.index, 'date'].reset_index(drop=True)
    smoothed = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': smoothed})

SAR_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
JRC_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
KGE_CSV = pathlib.Path('analysis/pilot_kge_v3.csv')
OUT_PNG = pathlib.Path('analysis/schwatke_output/ts_v3_fullperiod.png')

NO_SMOOTH = {'Hubbard_Creek'}

APP_OVERRIDES = {
    'Ancipa': (
        pathlib.Path('C:/Users/Unipa/Documents/GEE/Results/fractaldim/area_ancipa_2014-25.csv'),
        'date', 'areaLago',
    ),
}

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
DIR_COLOR      = {'ASCENDING': '#1565C0', 'DESCENDING': '#E65100'}


def load_sar_raw(name):
    if name in APP_OVERRIDES:
        path, dcol, acol = APP_OVERRIDES[name]
        df = pd.read_csv(path, parse_dates=[dcol])
        df = df.rename(columns={dcol: 'date', acol: 'area_ha'})
        df['passDirection'] = 'ASCENDING'  # unknown; default colour
        df = df[df['date'] <= '2021-12-31'].copy()
    else:
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

    # SAR cleaned + LOWESS line (mirrors original.js cleanAndSmooth pipeline)
    if df_sar is not None and not df_sar.empty:
        df_line = df_sar.reset_index(drop=True)
        if name not in NO_SMOOTH:
            c_df = clean_and_smooth(df_line)
            ax.plot(c_df['date'], c_df['area_ha'],
                    color='#757575', lw=1.2, alpha=0.85, zorder=4)
        else:
            ax.plot(df_line['date'], df_line['area_ha'],
                    color='#757575', lw=0.7, alpha=0.75, zorder=4)

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
    Line2D([0],[0], color='#757575', lw=1.5, label='SAR cleaned + LOWESS'),
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
