"""
plot_timeseries_new15.py

Same small-multiples view as plot_timeseries_all.py, filtered to just the 15
reservoirs added 7-8 Jul 2026 (5 Temperate/continental + 10 A/P dip-bin
reinforcement), for visual sanity-checking of the new data before trusting it
in the pooled statistics.

Output: analysis/method_comparison_output/timeseries_new15.png
"""
import pathlib, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def _rg(s, t=2.0):
    m, sd = s.mean(), s.std(); return s[np.abs(s - m) <= t * sd]
def _rl(s, w=5, t=1.5):
    a, idx = s.values.copy(), s.index.tolist(); keep, h = [], w // 2
    for i in range(len(a)):
        lo, hi = max(0, i - h), min(len(a), i + h + 1); win = a[lo:hi]
        m, sd = win.mean(), win.std()
        if sd == 0 or abs(a[i] - m) <= t * sd: keep.append(idx[i])
    return s.loc[keep]
def _lw(dates, vals, wd=20, bw=7):
    out = []
    for t0 in dates:
        dd = np.abs((dates - t0).dt.total_seconds().values / 86400); mk = dd <= wd
        wt = np.exp(-(dd[mk] / bw) ** 2); out.append(float((vals[mk] * wt).sum() / wt.sum()))
    return np.array(out)
def clean_and_smooth(df):
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    return pd.DataFrame({'date': ds, 'area_ha': _lw(ds, s.reset_index(drop=True), 20, 7)})

def _P(*p): return [pathlib.Path(x) for x in p]
METHOD_DIRS = {
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
}
JRC_DIRS = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
              'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
COL = {'adapt': '#1f77b4', 'vv': '#2ca02c'}
LAB = {'adapt': 'adapt (per-scene SVM)', 'vv': 'VV-only Otsu'}
SAR_MIN_FRAC, AREA_MIN = 0.02, {}

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None
def load_sar(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None
    if df.empty: return None
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if df.empty: return None
    df = clean_and_smooth(df.reset_index(drop=True))
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index()
    m['date'] = m['ym'].dt.to_timestamp()
    return m[['date', 'area_ha']]
from jrc_filter import load_jrc_monthly as _load_jrc_shared
def load_jrc(n):
    return _load_jrc_shared(n, JRC_DIRS, despike=True)
def break_gaps(df, dc, vc):
    s = df.copy(); s[dc] = s[dc].dt.to_period('M').dt.to_timestamp()
    s = s.groupby(dc)[vc].mean()
    full = pd.period_range(s.index.min(), s.index.max(), freq='M').to_timestamp()
    return s.reindex(full)

NEW15 = ['Vranov', 'Roxburgh', 'Conestogo', 'Yedang', 'Loch_Doon',
         'Mundaring_Weir', 'Kartalkaya', 'Lago_de_Almafuerte', 'Hassan_Addakhil',
         'Da_Mi_1', 'Ambuklao', 'Barekese', 'Kotmale', 'Cinco_de_Noviembre', 'Kidatu']

bof = pd.read_csv('analysis/bestof_kge.csv').set_index('name')
names = [n for n in NEW15 if n in bof.index]
missing = [n for n in NEW15 if n not in bof.index]
if missing:
    print('[warn] not in bestof_kge.csv:', missing)
names = sorted(names, key=lambda n: bof.loc[n, 'ap_m'])
kge = bof['best'].to_dict(); ap = bof['ap_m'].to_dict(); winner = bof['winner'].to_dict()

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ncol = 5; nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.3 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(names):]:
    ax.axis('off')

for i, n in enumerate(names):
    ax = axes[i]
    jrc = load_jrc(n)
    if jrc is not None and not jrc.empty:
        g = break_gaps(jrc, 'date', 'jrc_area_ha')
        ax.plot(g.index, g.values, color='k', lw=0.9, ls='--', marker='o', ms=2, zorder=6, label='JRC')
    for meth, dirs in METHOD_DIRS.items():
        s = load_sar(dirs, n)
        if s is not None and not s.empty:
            ax.plot(s['date'], s['area_ha'], color=COL[meth], lw=1.4, zorder=5, label=LAB[meth])
    k = kge.get(n, np.nan); w = winner.get(n, '')
    if pd.notna(k) and k < 0.5:
        ax.set_facecolor('#fdecec')
    ax.set_title(f"{n.replace('_',' ')}  A/P={ap.get(n,np.nan):.0f}  KGE={k:+.2f} ({w})"
                 if pd.notna(k) else n.replace('_', ' '), fontsize=8.5)
    ax.tick_params(axis='both', labelsize=6.5); ax.margins(x=0.02); ax.grid(alpha=0.2)

handles = [Line2D([0], [0], color='k', ls='--', marker='o', ms=4, label='JRC (optical ref)')] + \
          [Line2D([0], [0], color=COL[m], lw=2, label=LAB[m]) for m in METHOD_DIRS]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.suptitle('The 15 newly-added reservoirs (5 Temperate + 10 A/P dip-bin), sorted by A/P\n'
             '(title KGE = best-of method; pink = best<0.5)',
             fontsize=13, fontweight='bold', y=1.05)
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT = pathlib.Path('analysis/method_comparison_output/timeseries_new15.png')
fig.savefig(OUT, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {OUT}  ({len(names)} reservoirs)')
