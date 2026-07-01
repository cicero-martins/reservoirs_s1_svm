"""
plot_timeseries_panel.py

Small-multiples panel (one subplot per reservoir, sorted by A/P) of the monthly
water-area series that feed the KGE comparison:
  - dual  : VV+VH SVM   (blue)
  - vv    : VV-only Otsu (green)
  - JRC   : optical reference (orange, dashed markers)
plus PER-OVERPASS ERA5 wind on a secondary axis (light grey fill+trace) — the same
wind quantity used in analyze_wind_divergence.py (NOT a monthly mean), so the gusty
acquisitions can be read against any area divergence.

Series are processed with the SAME clean+smooth+monthly pipeline as
compute_kge_compare.py (so the panel matches the ΔKGE numbers); window is the
JRC-overlap period (≤ 2021-12-31).

Reads:
  dual : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4/SAR_area_*.csv
  vv   : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu/SAR_area_*.csv
  JRC  : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC/JRC_area_*.csv
  wind : raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind/Era5Wind_*.csv

Output: analysis/method_comparison_output/timeseries_panel.png
"""

import pathlib
import re as _re
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

SAR_DUAL_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
SAR_VV_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu')
JRC_DIR      = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
WIND_DIR     = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_Era5Wind')
OUT          = pathlib.Path('analysis/method_comparison_output/timeseries_panel.png')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
AREA_MIN = {'Saint_Cassien': 200}


# ── clean+smooth (identical to compute_kge_compare.py) ────────────────────────
def _remove_global(s, threshold=2.0):
    m, sd = s.mean(), s.std()
    return s[np.abs(s - m) <= threshold * sd]

def _remove_local(s, window=5, threshold=1.5):
    arr, idx = s.values.copy(), s.index.tolist()
    keep, half = [], window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        m, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - m) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]

def _lowess(dates, values, window_days=20, bandwidth=7):
    out = []
    for t0 in dates:
        dd = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask = dd <= window_days
        w = np.exp(-(dd[mask] / bandwidth) ** 2)
        out.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(out)

def clean_and_smooth(df, col='area_ha'):
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s = df.loc[s.index, 'date'].reset_index(drop=True)
    sm = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': sm})


def _jrc_path(name):
    cands = sorted(JRC_DIR.glob(f'JRC_area_{name}*.csv'))
    plain = [p for p in cands if not _re.search(r'\s*\(\d+\)', p.stem)]
    return plain[0] if plain else (cands[0] if cands else None)


def load_sar_monthly(name, sar_dir):
    p = sar_dir / f'SAR_area_{name}.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    df = df[df['date'] <= '2021-12-31'].copy()
    if df.empty:
        return None
    p99 = df['area_ha'].quantile(0.99)
    df = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    if df.empty:
        return None
    df = clean_and_smooth(df.reset_index(drop=True))
    if df.empty:
        return None
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index()
    m['date'] = m['ym'].dt.to_timestamp()
    return m[['date', 'area_ha']]


def load_jrc_monthly(name):
    p = _jrc_path(name)
    if p is None:
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    df = df.sort_values('date').reset_index(drop=True)
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if sd > 0:
        df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd].copy()
    df = df[df['date'] <= '2021-12-31']
    return df[['date', 'jrc_area_ha']].dropna()


def break_month_gaps(df, datecol, valcol):
    """Reindex a monthly series onto a complete month grid so missing months
    become NaN — matplotlib then BREAKS the line across gaps instead of drawing
    a straight interpolation through months that have no measurement. Markers
    skip NaN too, so only real points are drawn."""
    if df is None or df.empty:
        return None
    s = df.copy()
    s[datecol] = s[datecol].dt.to_period('M').dt.to_timestamp()  # snap to month start
    s = s.groupby(datecol)[valcol].mean()
    full = pd.period_range(s.index.min(), s.index.max(), freq='M').to_timestamp()
    return s.reindex(full)   # Series indexed by month; gaps = NaN


def load_wind_overpass(name):
    """Per-overpass wind at the actual S1 acquisition dates — the SAME quantity
    used in analyze_wind_divergence.py (mean if duplicate overpasses per date).
    NOT a monthly mean: shows the gusty overpass peaks the Bragg mechanism cares
    about."""
    p = WIND_DIR / f'Era5Wind_{name}.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if df.empty or 'wind_ms' not in df.columns:
        return None
    df = df[df['date'] <= '2021-12-31'].copy()
    return df.groupby('date')['wind_ms'].mean().reset_index().sort_values('date')


# ── reservoir order: by A/P from the comparison table ─────────────────────────
cmp = pd.read_csv('analysis/pilot_kge_compare.csv').sort_values('ap_m')
names = cmp['name'].tolist()
ap_lookup    = dict(zip(cmp['name'], cmp['ap_m']))
delta_lookup = dict(zip(cmp['name'], cmp['delta_kge']))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ncol = 4
nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 2.7 * nrow))
axes = np.atleast_1d(axes).ravel()

C_DUAL, C_VV, C_JRC, C_WIND = '#1f77b4', '#2ca02c', '#e65100', '#9aa7b0'

for ax in axes[len(names):]:
    ax.axis('off')

for i, name in enumerate(names):
    ax = axes[i]
    dual = load_sar_monthly(name, SAR_DUAL_DIR)
    vv   = load_sar_monthly(name, SAR_VV_DIR)
    jrc  = load_jrc_monthly(name)
    wind = load_wind_overpass(name)

    # per-overpass wind on a secondary axis, drawn first (background). Thin spiky
    # trace (not a monthly mean) so gusty acquisitions are visible.
    if wind is not None and not wind.empty:
        axw = ax.twinx()
        axw.fill_between(wind['date'], 0, wind['wind_ms'], color=C_WIND,
                         alpha=0.25, zorder=1, linewidth=0)
        axw.plot(wind['date'], wind['wind_ms'], color='#6b7b88', lw=0.4,
                 alpha=0.6, zorder=1)
        axw.set_ylim(0, max(10, wind['wind_ms'].max() * 1.1))
        axw.tick_params(axis='y', labelsize=6, colors='#5b6b78')
        if (i % ncol) == (ncol - 1):
            axw.set_ylabel('wind (m/s)', fontsize=7, color='#5b6b78')

    if jrc is not None and not jrc.empty:
        jrc_g = break_month_gaps(jrc, 'date', 'jrc_area_ha')   # NaN at missing months → line breaks
        ax.plot(jrc_g.index, jrc_g.values, color=C_JRC, lw=1.0, ls='--',
                marker='o', ms=2.5, zorder=4, label='JRC')
    if dual is not None and not dual.empty:
        ax.plot(dual['date'], dual['area_ha'], color=C_DUAL, lw=1.8, zorder=5,
                label='dual (VV+VH)')
    if vv is not None and not vv.empty:
        ax.plot(vv['date'], vv['area_ha'], color=C_VV, lw=1.8, zorder=5,
                label='VV-only')

    ax.set_zorder(2); ax.patch.set_visible(False)   # area lines above wind fill
    ap = ap_lookup.get(name, np.nan)
    dk = delta_lookup.get(name, np.nan)
    winner = ('dual' if dk > 0.02 else 'VV' if dk < -0.02 else 'tie') if pd.notna(dk) else '?'
    ax.set_title(f"{name.replace('_', ' ')}   A/P={ap:.0f} m   "
                 f"ΔKGE={dk:+.2f} ({winner})", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.margins(x=0.02)
    ax.grid(alpha=0.2)
    if (i % ncol) == 0:
        ax.set_ylabel('area (ha)', fontsize=7)

handles = [Line2D([0], [0], color=C_DUAL, lw=2, label='dual (VV+VH SVM)'),
           Line2D([0], [0], color=C_VV, lw=2, label='VV-only Otsu'),
           Line2D([0], [0], color=C_JRC, lw=1.2, ls='--', marker='o', ms=4, label='JRC optical'),
           plt.Rectangle((0, 0), 1, 1, fc=C_WIND, alpha=0.3, label='ERA5 wind (per overpass)')]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 1.005), frameon=False)
fig.suptitle('Monthly water area — VV-only vs dual vs JRC, with wind overlay '
             '(reservoirs sorted by A/P)', fontsize=12, fontweight='bold', y=1.015)
fig.tight_layout(rect=[0, 0, 1, 0.99])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}  ({len(names)} reservoirs)')
