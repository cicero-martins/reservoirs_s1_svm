"""
plot_timeseries_all.py

Small-multiples of the monthly water-area series for EVERY analysed reservoir
(sorted by A/P), so the low-KGE diagnosis can be read visually:
  JRC   : optical reference (black dashed + markers, line breaks at data gaps)
  adapt : VV+VH SVM per-scene   (blue)
  dual  : VV+VH SVM fixed 2023   (purple)
  vv    : VV-only Otsu           (green)

Same clean+smooth+monthly pipeline as compute_kge_4way.py; window ≤2021-12-31.
Panel title = A/P + KGE(dual vs JRC) from low_kge_diagnosis.csv; panels with
KGE<0.5 get a light red tint so the failures pop.

Output: analysis/method_comparison_output/timeseries_all.png
"""
import pathlib, re as _re, sys
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
# dual-FIXED retired from the story (arbitrary 2023 baseline; adds noise) → only the two
# justified methods: adapt (per-scene SVM) + vv (Otsu).
METHOD_DIRS = {
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
}
JRC_DIRS = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
              'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
COL = {'adapt': '#1f77b4', 'vv': '#2ca02c'}
LAB = {'adapt': 'adapt (per-scene SVM)', 'vv': 'VV-only Otsu'}
VALID_FRAC_MIN, SAR_MIN_FRAC, AREA_MIN = 0.80, 0.02, {'Saint_Cassien': 200}

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
    return _load_jrc_shared(n, JRC_DIRS, despike=True)  # validated vf-gated de-spike
def break_gaps(df, dc, vc):
    s = df.copy(); s[dc] = s[dc].dt.to_period('M').dt.to_timestamp()
    s = s.groupby(dc)[vc].mean()
    full = pd.period_range(s.index.min(), s.index.max(), freq='M').to_timestamp()
    return s.reindex(full)

# order + BEST-OF KGE (per-reservoir best method) + winner, from compute_bestof_kge.py
# drop reservoirs with no adapt/vv (best=NaN) — the 4 Sicilian (JRC period = dual-only),
# which belong to the PlanetScope near-truth validation, not the JRC analysis.
bof = pd.read_csv('analysis/bestof_kge.csv').dropna(subset=['best']).sort_values('ap_m').reset_index(drop=True)
names = bof['name'].tolist()
kge = dict(zip(bof['name'], bof['best'])); ap = dict(zip(bof['name'], bof['ap_m']))
winner = dict(zip(bof['name'], bof['winner']))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ncol = 6; nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.1 * nrow))
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
            ax.plot(s['date'], s['area_ha'], color=COL[meth], lw=1.3, zorder=5, label=LAB[meth])
    k = kge.get(n, np.nan); w = winner.get(n, '')
    if pd.notna(k) and k < 0.5:
        ax.set_facecolor('#fdecec')
    ax.set_title(f"{n.replace('_',' ')}  A/P={ap.get(n,np.nan):.0f}  KGE={k:+.2f} ({w})"
                 if pd.notna(k) else n.replace('_', ' '), fontsize=7.5)
    ax.tick_params(axis='both', labelsize=6); ax.margins(x=0.02); ax.grid(alpha=0.2)

handles = [Line2D([0], [0], color='k', ls='--', marker='o', ms=4, label='JRC (optical ref)')] + \
          [Line2D([0], [0], color=COL[m], lw=2, label=LAB[m]) for m in METHOD_DIRS]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 1.004), frameon=False)
fig.suptitle('Monthly water area — JRC vs SAR methods, all reservoirs sorted by A/P '
             '(title KGE = BEST-OF method, in parens; pink = best<0.5)',
             fontsize=13, fontweight='bold', y=1.012)
fig.tight_layout(rect=[0, 0, 1, 0.99])
OUT = pathlib.Path('analysis/method_comparison_output/timeseries_all.png')
fig.savefig(OUT, dpi=120, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {OUT}  ({len(names)} reservoirs)')
