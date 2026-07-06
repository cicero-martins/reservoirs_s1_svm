"""
investigate_sparse.py

Robustness check on the low-KGE reservoirs: is the low score a data-poverty artifact
rather than a real monitorability limit?

Two mechanisms tested:
  (1) SPARSE JRC reference → few comparison months → KGE ill-constrained / unlucky.
  (2) SPARSE SAR before ~2017 (S1A-only era, before S1B) → the early monthly SAR values
      are a coarse LOWESS interpolation over few scenes, dragging the KGE down.

For each reservoir it reports data density (raw SAR scenes pre/post 2017, valid JRC months)
and recomputes best-of(adapt,vv) KGE on the FULL window vs the DENSE window (≥2017-01),
so we can see which reservoirs are rescued by dropping the thin early period.

Output: analysis/sparse_robustness.csv  + printed tables.
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

from jrc_filter import load_jrc_monthly as load_jrc

# ── SAR clean+smooth (identical to pipeline) ──────────────────────────────────
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
DIRS = {
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
}
JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
SAR_MIN_FRAC, AREA_MIN = 0.02, {'Saint_Cassien': 200}
FLAG = ['Chenderoh', 'Sarrans', 'Rappbode', 'Wusijiang', 'Bilancino', 'Amir_Kabir',
        'Songhuaba', 'East_Canyon']

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None

def raw_sar(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None
    df = df[(df['area_ha'] > 0) & (df['date'] <= '2021-12-31')].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    if df.empty: return None
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    return df[['date', 'area_ha']].reset_index(drop=True)

def sar_monthly(dirs, n):
    df = raw_sar(dirs, n)
    if df is None or len(df) < 5: return None
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    sm = pd.DataFrame({'date': ds, 'area': _lw(ds, s.reset_index(drop=True), 20, 7)})
    sm['ym'] = sm['date'].dt.to_period('M')
    return sm.groupby('ym')['area'].mean().reset_index()

def kge(o, s):
    if len(o) < 3 or np.std(o) == 0 or np.std(s) == 0: return np.nan
    r = stats.pearsonr(o, s)[0]
    return 1 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/np.mean(o)-1)**2)

names = pd.read_csv('analysis/bestof_kge.csv').dropna(subset=['best'])['name'].tolist()
CUT = pd.Period('2017-01', 'M')
rows = []
for n in names:
    jr = load_jrc(n, JRC, despike=True)
    if jr is None: continue
    jr = jr.copy(); jr['ym'] = jr['date'].dt.to_period('M')
    jrm = jr[['ym', 'jrc_area_ha']]
    # data density
    rs = raw_sar(DIRS['adapt'], n)
    sar_pre = int((rs['date'] < '2017-01-01').sum()) if rs is not None else 0
    sar_post = int((rs['date'] >= '2017-01-01').sum()) if rs is not None else 0
    njrc = len(jrm); njrc_pre = int((jrm['ym'] < CUT).sum())

    best_full = best_dense = -9; nf = nd = 0
    for meth in DIRS:
        sm = sar_monthly(DIRS[meth], n)
        if sm is None: continue
        mg = sm.rename(columns={'area': 'sar'}).merge(jrm, on='ym').dropna()
        if len(mg) >= 12:
            k = kge(mg['jrc_area_ha'].values, mg['sar'].values)
            if k > best_full: best_full, nf = k, len(mg)
        md = mg[mg['ym'] >= CUT]
        if len(md) >= 12:
            k = kge(md['jrc_area_ha'].values, md['sar'].values)
            if k > best_dense: best_dense, nd = k, len(md)
    if best_full == -9: continue
    rows.append({'name': n, 'sar_pre17': sar_pre, 'sar_post17': sar_post,
                 'njrc': njrc, 'njrc_pre17': njrc_pre, 'n_full': nf, 'n_dense': nd,
                 'kge_full': round(best_full, 3),
                 'kge_dense': round(best_dense, 3) if best_dense > -9 else np.nan})

df = pd.DataFrame(rows)
df['dkge'] = df['kge_dense'] - df['kge_full']
df.to_csv('analysis/sparse_robustness.csv', index=False)

print('=== FLAGGED reservoirs: data density + full vs dense(≥2017) KGE ===')
f = df[df.name.isin(FLAG)].sort_values('kge_full')
print(f[['name', 'sar_pre17', 'sar_post17', 'njrc', 'njrc_pre17', 'n_full', 'n_dense',
         'kge_full', 'kge_dense', 'dkge']].to_string(index=False))

print('\n=== does sparse data explain low KGE? (all N=%d) ===' % len(df))
r1, p1 = stats.spearmanr(df['n_full'], df['kge_full'])
r2, p2 = stats.spearmanr(df['njrc'], df['kge_full'])
r3, p3 = stats.spearmanr(df['sar_pre17'], df['kge_full'])
print(f'  Spearman(n_common, KGE_full)   = {r1:+.3f} p={p1:.3f}')
print(f'  Spearman(n_JRC months, KGE)    = {r2:+.3f} p={p2:.3f}')
print(f'  Spearman(SAR scenes pre-2017, KGE) = {r3:+.3f} p={p3:.3f}')

low = df[df.kge_full < 0.5]
print(f'\n=== effect of dropping pre-2017 on the low-KGE set (N={len(low)}) ===')
print(f'  median ΔKGE (dense − full) = {low["dkge"].median():+.3f}   mean = {low["dkge"].mean():+.3f}')
print(f'  low reservoirs improved by dense window (Δ>+0.05): '
      f'{int((low["dkge"] > 0.05).sum())}/{len(low)}')
rescued = low[(low.kge_full < 0.5) & (low.kge_dense >= 0.5)]
print(f'  crossed 0.5 when restricted to ≥2017: {len(rescued)}  {list(rescued["name"])}')
print(f'\n  full set: median KGE_full={df["kge_full"].median():.3f}  '
      f'median KGE_dense={df["kge_dense"].median():.3f}')
