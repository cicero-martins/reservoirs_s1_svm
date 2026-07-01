"""
plot_scatter_panels.py

Three paired scatter panels of monthly water area on COMMON months, pooled across
all reservoirs (sorted/colored by A/P):
  (a) VV-only Otsu  vs  JRC optical   (reference = JRC)
  (b) VV+VH SVM     vs  JRC optical   (reference = JRC)
  (c) VV-only Otsu  vs  VV+VH SVM     (reference = dual)

Log-log axes + 1:1 line (areas span ~10–5000 ha). Each panel reports N points,
Pearson r on log-area (shape agreement), and the median ratio y/x (systematic
bias: >1 means the y-method reads larger). Same clean+smooth+monthly pipeline as
compute_kge_compare.py, so panels match the ΔKGE story.

Output: analysis/method_comparison_output/scatter_panels.png
"""

import pathlib
import re as _re
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

SAR_DUAL_DIR = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
SAR_VV_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu')
JRC_DIR      = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
OUT          = pathlib.Path('analysis/method_comparison_output/scatter_panels.png')

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
AREA_MIN = {'Saint_Cassien': 200}


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
    return df.groupby('ym')['area_ha'].mean().rename('v').reset_index()


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
    df = df[df['date'] <= '2021-12-31'].copy()
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['jrc_area_ha'].mean().rename('v').reset_index()


cmp = pd.read_csv('analysis/pilot_kge_compare.csv').sort_values('ap_m')
names = cmp['name'].tolist()
ap_lookup = dict(zip(cmp['name'], cmp['ap_m']))

# ── Build pooled common-month pairs ──────────────────────────────────────────
pairs = {'vv_jrc': [], 'dual_jrc': [], 'vv_dual': []}
for name in names:
    dual = load_sar_monthly(name, SAR_DUAL_DIR)
    vv   = load_sar_monthly(name, SAR_VV_DIR)
    jrc  = load_jrc_monthly(name)
    ap   = ap_lookup.get(name, np.nan)

    if vv is not None and jrc is not None:
        m = vv.merge(jrc, on='ym', suffixes=('_vv', '_jrc')).dropna()
        if not m.empty:
            pairs['vv_jrc'].append(pd.DataFrame({'x': m['v_jrc'], 'y': m['v_vv'],
                                                 'ap': ap, 'name': name}))
    if dual is not None and jrc is not None:
        m = dual.merge(jrc, on='ym', suffixes=('_dual', '_jrc')).dropna()
        if not m.empty:
            pairs['dual_jrc'].append(pd.DataFrame({'x': m['v_jrc'], 'y': m['v_dual'],
                                                   'ap': ap, 'name': name}))
    if vv is not None and dual is not None:
        m = vv.merge(dual, on='ym', suffixes=('_vv', '_dual')).dropna()
        if not m.empty:
            pairs['vv_dual'].append(pd.DataFrame({'x': m['v_dual'], 'y': m['v_vv'],
                                                  'ap': ap, 'name': name}))

data = {k: pd.concat(v, ignore_index=True) for k, v in pairs.items() if v}

# shared log limits
allvals = np.concatenate([np.r_[d['x'].values, d['y'].values] for d in data.values()])
allvals = allvals[allvals > 0]
lo, hi = allvals.min() * 0.8, allvals.max() * 1.2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [
    ('vv_jrc',   'JRC optical area (ha)', 'VV-only Otsu area (ha)', '(a) VV-only vs JRC'),
    ('dual_jrc', 'JRC optical area (ha)', 'VV+VH SVM area (ha)',    '(b) dual vs JRC'),
    ('vv_dual',  'VV+VH SVM area (ha)',   'VV-only Otsu area (ha)', '(c) VV-only vs dual'),
]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
sc = None
for ax, (key, xlab, ylab, title) in zip(axes, PANELS):
    d = data[key]
    d = d[(d['x'] > 0) & (d['y'] > 0)]
    sc = ax.scatter(d['x'], d['y'], c=d['ap'], cmap='viridis', s=12,
                    alpha=0.55, edgecolors='none', zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7, zorder=4, label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    r, _ = stats.pearsonr(np.log10(d['x']), np.log10(d['y']))
    ratio = np.median(d['y'] / d['x'])
    txt = (f"N = {len(d)}\n"
           f"r(log) = {r:.3f}\n"
           f"median y/x = {ratio:.2f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.85))
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.2, which='both')
    print(f'{title}: N={len(d)}  r(log)={r:.3f}  median y/x={ratio:.3f}')

cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label('A/P static (m)', fontsize=10)
fig.suptitle('Monthly area agreement on common months (pooled, log-log, colored by A/P)',
             fontsize=13, fontweight='bold')
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT}')
