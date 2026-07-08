"""
plot_scatter_panels.py  (v2 — full current dataset, de-spiked JRC)

Pooled area-agreement scatter panels over EVERY analysed reservoir (one point per
reservoir-month, common months only), coloured by A/P:
  (a) VV-Otsu  vs JRC
  (b) SVM adapt vs JRC
  (c) VV-Otsu  vs SVM adapt

Monthly SAR = clean+smooth pipeline; JRC = validated de-spiked reference (jrc_filter.py).
Each panel: 1:1 line, Pearson r on log10, median ratio, N points.

Output: analysis/method_comparison_output/scatter_panels.png
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
from jrc_filter import load_jrc_monthly as load_jrc

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

def _P(*p): return [pathlib.Path(x) for x in p]
MDIR = {'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
        'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu')}
JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
SAR_MIN_FRAC, AREA_MIN = 0.02, {'Saint_Cassien': 200}

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None
def sar_monthly(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None, np.nan
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None, np.nan
    ap = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan
    df = df[(df['area_ha'] > 0) & (df['date'] <= '2021-12-31')].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if len(df) < 5: return None, ap
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    sm = pd.DataFrame({'date': ds, 'area': _lw(ds, s.reset_index(drop=True), 20, 7)})
    sm['ym'] = sm['date'].dt.to_period('M')
    return sm.groupby('ym')['area'].mean().reset_index(), ap

best = pd.read_csv('analysis/bestof_kge.csv')
names = best['name'].tolist()
try:
    _rn = pd.read_csv('analysis/reference_noise.csv')
    REF_NOISE = set(_rn.loc[_rn.ref_noise, 'name'])
except FileNotFoundError:
    REF_NOISE = set()
CHAPADO = set(best.loc[best['chapado'] == True, 'name']) if 'chapado' in best.columns else set()
EXCLUDE = REF_NOISE | CHAPADO
names = [n for n in names if n not in EXCLUDE]
pool = []
for n in names:
    ad, ap = sar_monthly(MDIR['adapt'], n)
    vv, _ = sar_monthly(MDIR['vv'], n)
    jr = load_jrc(n, JRC, despike=True)
    if ad is None or vv is None or jr is None: continue
    jr = jr.copy(); jr['ym'] = jr['date'].dt.to_period('M')
    m = (ad.rename(columns={'area': 'adapt'})
         .merge(vv.rename(columns={'area': 'vv'}), on='ym')
         .merge(jr[['ym', 'jrc_area_ha']].rename(columns={'jrc_area_ha': 'jrc'}), on='ym').dropna())
    if m.empty: continue
    m['ap_m'] = ap; m['name'] = n
    pool.append(m[['name', 'ap_m', 'jrc', 'adapt', 'vv']])

d = pd.concat(pool, ignore_index=True)
d = d[(d[['jrc', 'adapt', 'vv']] > 0).all(axis=1)]
print(f'{len(d)} reservoir-month points across {d["name"].nunique()} reservoirs')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('jrc', 'vv',    'JRC area (ha)',        'VV-Otsu area (ha)',   '(a) VV-Otsu vs JRC'),
          ('jrc', 'adapt', 'JRC area (ha)',        'SVM adapt area (ha)', '(b) SVM adapt vs JRC'),
          ('adapt', 'vv',  'SVM adapt area (ha)',  'VV-Otsu area (ha)',   '(c) VV-Otsu vs SVM adapt')]
lo = max(1.0, d[['jrc', 'adapt', 'vv']].min().min() * 0.7)
hi = d[['jrc', 'adapt', 'vv']].max().max() * 1.4

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (xc, yc, xl, yl, tt) in zip(axes, PANELS):
    sc = ax.scatter(d[xc], d[yc], c=d['ap_m'], cmap='viridis', s=8, alpha=0.35,
                    linewidths=0, zorder=3)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7, zorder=4, label='1:1')
    r = stats.pearsonr(np.log10(d[xc]), np.log10(d[yc]))[0]
    ratio = np.median(d[yc] / d[xc])
    ax.text(0.04, 0.96, f'N={len(d)}\nr(log)={r:.3f}\nmed y/x={ratio:.2f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='#bbb', alpha=0.85))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
    ax.set_title(tt, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2, which='both'); ax.legend(fontsize=9, loc='lower right')
cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01); cb.set_label('A/P (m)', fontsize=10)
fig.suptitle(f'Area agreement — pooled reservoir-months (N={len(d)}, {d["name"].nunique()} reservoirs), '
             'coloured by A/P', fontsize=13, fontweight='bold')
OUT = pathlib.Path('analysis/method_comparison_output/scatter_panels.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {OUT}')

for xc, yc, *_ in PANELS:
    r = stats.pearsonr(np.log10(d[xc]), np.log10(d[yc]))[0]
    print(f'  {yc:>5} vs {xc:<5}: r(log)={r:.3f}  med ratio={np.median(d[yc]/d[xc]):.3f}')
