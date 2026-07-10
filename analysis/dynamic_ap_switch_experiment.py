"""
dynamic_ap_switch_experiment.py

Tests whether an operational VV-Otsu -> SVM switch, triggered by Otsu's OWN
per-scene dynamic A/P (perimeter/area of that scene's water polygon, already
exported as `ap_m_dynamic` in every SAR_area_*.csv), improves monthly accuracy
relative to always-VV or always-SVM.

Design (agreed with user, 9 Jul 2026):
  X = Otsu's own dynamic A/P that month (not JRC's, not SVM's -- avoids
      circularity and is the only signal actually available in real-time
      operation, since Otsu runs first/cheap and already reports its own
      geometry per scene).
  Y = |err_vv| - |err_svm|, monthly, err = log(SAR_area / JRC_area) against
      the de-spiked JRC monthly reference (only ground truth we have, so the
      test resolution is capped at monthly).
  H0: no relationship. H1 (mechanism seen at Ancipa/Sicily): Y grows as X
      shrinks, i.e. SVM's local edge grows as the scene-level shoreline gets
      more complex / lower A/P.

Stages: (1) pooled + per-reservoir correlation, (2) Youden-J threshold search
on the binary "SVM better this month" label, (3) backtest a hybrid monthly
series (switch to SVM when Otsu's own dynamic A/P < T*) against pure-VV and
pure-SVM KGE, per reservoir and pooled.

Output:
  analysis/dynamic_ap_switch_pooled.csv       (reservoir-month pooled table)
  analysis/dynamic_ap_switch_backtest.csv     (per-reservoir KGE: vv/svm/hybrid)
  analysis/method_comparison_output/dynamic_ap_switch_scatter.png
  analysis/method_comparison_output/dynamic_ap_switch_backtest.png
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

from jrc_filter import load_jrc_monthly

# ── shared cleaning pipeline (identical to compute_bestof_kge.py) ─────────────
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
    'vv':   _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
}
JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
SAR_MIN_FRAC, AREA_MIN, MIN_PAIRS = 0.02, {'Saint_Cassien': 200}, 12

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None

def load_sar_monthly(dirs, n, with_apdyn=False):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    if df.empty or (with_apdyn and 'ap_m_dynamic' not in df.columns):
        return None
    df = df[df['area_ha'] > 0].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if len(df) < 5: return None
    df = df.reset_index(drop=True)
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    keep = s.index
    ds = df.loc[keep, 'date'].reset_index(drop=True)
    area_sm = _lw(ds, s.reset_index(drop=True), 20, 7)
    out = pd.DataFrame({'date': ds, 'area_ha': area_sm})
    if with_apdyn:
        out['ap_m_dynamic'] = df.loc[keep, 'ap_m_dynamic'].reset_index(drop=True)
    out['ym'] = out['date'].dt.to_period('M')
    agg = {'area_ha': 'mean'}
    if with_apdyn: agg['ap_m_dynamic'] = 'mean'
    return out.groupby('ym').agg(agg).reset_index()

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    if len(o) < 3 or np.std(o) == 0 or np.std(s) == 0: return np.nan
    r = stats.pearsonr(o, s)[0]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (np.mean(s) / np.mean(o) - 1) ** 2)

# ── reservoir universe: the final, clean apcurve set ──────────────────────────
NAMES = pd.read_csv('analysis/pilot_kge_apcurve.csv')['name'].tolist()
print(f'Testing {len(NAMES)} reservoirs (final reference-quality-screened apcurve set)')

pooled = []
per_reservoir_merged = {}
for n in NAMES:
    vv = load_sar_monthly(DIRS['vv'], n, with_apdyn=True)
    sv = load_sar_monthly(DIRS['adapt'], n, with_apdyn=False)
    jr = load_jrc_monthly(n, JRC, despike=True)
    if vv is None or sv is None or jr is None:
        continue
    vv = vv.rename(columns={'area_ha': 'area_vv'})
    sv = sv.rename(columns={'area_ha': 'area_svm'})
    jr = jr.copy(); jr['ym'] = jr['date'].dt.to_period('M')
    m = (vv.merge(sv[['ym', 'area_svm']], on='ym')
           .merge(jr[['ym', 'jrc_area_ha']].rename(columns={'jrc_area_ha': 'area_jrc'}), on='ym')
           .dropna())
    if len(m) < MIN_PAIRS:
        continue
    m = m[(m['area_vv'] > 0) & (m['area_svm'] > 0) & (m['area_jrc'] > 0)]
    if len(m) < MIN_PAIRS:
        continue
    m['err_vv'] = np.log(m['area_vv'] / m['area_jrc'])
    m['err_svm'] = np.log(m['area_svm'] / m['area_jrc'])
    m['y_svm_edge'] = m['err_vv'].abs() - m['err_svm'].abs()
    m['name'] = n
    per_reservoir_merged[n] = m
    pooled.append(m[['name', 'ym', 'ap_m_dynamic', 'y_svm_edge', 'area_vv', 'area_svm', 'area_jrc']])

pooled = pd.concat(pooled, ignore_index=True)
pooled['ym'] = pooled['ym'].astype(str)
pooled.to_csv('analysis/dynamic_ap_switch_pooled.csv', index=False)
print(f'\nPooled: {len(pooled)} reservoir-months across {pooled.name.nunique()} reservoirs')

# ── (1) correlation ────────────────────────────────────────────────────────
rho, p = stats.spearmanr(pooled['ap_m_dynamic'], pooled['y_svm_edge'])
print(f'\nPooled Spearman(ap_dyn_vv, SVM local edge) = {rho:+.3f}  p = {p:.2e}')

within = []
for n, g in pooled.groupby('name'):
    if len(g) < 8 or g['ap_m_dynamic'].std() == 0: continue
    r, pp = stats.spearmanr(g['ap_m_dynamic'], g['y_svm_edge'])
    within.append({'name': n, 'n': len(g), 'r': r, 'p': pp})
within = pd.DataFrame(within)
print(f'Per-reservoir (n={len(within)}): median rho = {within.r.median():+.3f}, '
      f'{int((within.r < 0).sum())}/{len(within)} negative (SVM edge grows as A/P falls), '
      f'{int(((within.r < 0) & (within.p < 0.05)).sum())} significant negative')

# binned medians
pooled['bin'] = pd.qcut(pooled['ap_m_dynamic'], 8, duplicates='drop')
binned = pooled.groupby('bin', observed=True).agg(
    ap_med=('ap_m_dynamic', 'median'), y_med=('y_svm_edge', 'median'), n=('y_svm_edge', 'size'))
print('\nBinned median SVM local edge by dynamic A/P (Otsu):')
print(binned.to_string())

# ── (2) Youden-J threshold search ──────────────────────────────────────────
label = (pooled['y_svm_edge'] > 0).values
x = pooled['ap_m_dynamic'].values
grid = np.arange(40, 340, 5)
best_T, best_J = None, -np.inf
for T in grid:
    pred = x < T
    tp = (pred & label).sum(); fn = (~pred & label).sum()
    tn = (~pred & ~label).sum(); fp = (pred & ~label).sum()
    if tp + fn == 0 or tn + fp == 0: continue
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    J = sens + spec - 1
    if J > best_J: best_J, best_T = J, T
print(f'\nYouden-J optimal threshold: T* = {best_T} m  (J = {best_J:.3f})')

# ── (3) backtest: hybrid monthly series per reservoir ──────────────────────
rows = []
for n, m in per_reservoir_merged.items():
    hybrid = np.where(m['ap_m_dynamic'] < best_T, m['area_svm'], m['area_vv'])
    kge_vv = kge(m['area_jrc'], m['area_vv'])
    kge_svm = kge(m['area_jrc'], m['area_svm'])
    kge_hyb = kge(m['area_jrc'], hybrid)
    n_switch = int((m['ap_m_dynamic'] < best_T).sum())
    rows.append({'name': n, 'n_months': len(m), 'n_switched_to_svm': n_switch,
                 'ap_dyn_min': m['ap_m_dynamic'].min(), 'ap_dyn_max': m['ap_m_dynamic'].max(),
                 'kge_vv': kge_vv, 'kge_svm': kge_svm, 'kge_hybrid': kge_hyb})
bt = pd.DataFrame(rows).sort_values('ap_dyn_min')
bt.to_csv('analysis/dynamic_ap_switch_backtest.csv', index=False)

print(f'\n=== Backtest (T*={best_T} m), N={len(bt)} reservoirs ===')
print(f'Median KGE   always-VV: {bt.kge_vv.median():.3f}   always-SVM: {bt.kge_svm.median():.3f}   '
      f'hybrid: {bt.kge_hybrid.median():.3f}')
print(f'hybrid beats always-VV in {int((bt.kge_hybrid > bt.kge_vv).sum())}/{len(bt)}, '
      f'ties in {int((bt.kge_hybrid == bt.kge_vv).sum())}, '
      f'loses in {int((bt.kge_hybrid < bt.kge_vv).sum())}')
ever_switched = bt[bt.n_switched_to_svm > 0]
print(f'\nReservoirs that ever cross T* (n={len(ever_switched)}):')
print(f'  median KGE always-VV: {ever_switched.kge_vv.median():.3f}   '
      f'hybrid: {ever_switched.kge_hybrid.median():.3f}   '
      f'delta median: {(ever_switched.kge_hybrid - ever_switched.kge_vv).median():+.3f}')
print(ever_switched[['name', 'n_months', 'n_switched_to_svm', 'ap_dyn_min',
                      'kge_vv', 'kge_svm', 'kge_hybrid']].to_string(index=False))

# ── figures ─────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pooled['ap_m_dynamic'], pooled['y_svm_edge'], s=8, alpha=0.25, color='#1565C0', linewidths=0)
ax.plot(binned['ap_med'], binned['y_med'], 'o-', color='#C62828', ms=7, lw=2, label='median per bin', zorder=5)
ax.axhline(0, color='gray', lw=1, ls=':')
ax.axvline(best_T, color='#2E7D32', lw=1.3, ls='--', label=f'T* = {best_T} m (Youden-J)')
ax.set_xlabel("Otsu's own dynamic A/P that month (m)")
ax.set_ylabel('SVM local monthly edge  (|err$_{VV}$| - |err$_{SVM}$|)')
ax.set_title(f'Dynamic A/P vs SVM local accuracy edge\n'
             f'pooled Spearman r={rho:.2f}, p={p:.1e}, N={len(pooled)} reservoir-months', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('analysis/method_comparison_output/dynamic_ap_switch_scatter.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(bt))))
y = np.arange(len(bt))
ax.scatter(bt.kge_vv, y, color='#1565C0', s=40, label='always VV-Otsu', zorder=3)
ax.scatter(bt.kge_hybrid, y, color='#2E7D32', s=40, marker='D', label='hybrid switch', zorder=4)
for i, (_, r) in enumerate(bt.iterrows()):
    ax.plot([r.kge_vv, r.kge_hybrid], [i, i], color='#999', lw=1, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['name']} (A/P {r['ap_dyn_min']:.0f}-{r['ap_dyn_max']:.0f})" for _, r in bt.iterrows()],
                    fontsize=6.5)
ax.set_xlabel('KGE vs JRC')
ax.set_title(f'Backtest: always-VV vs hybrid switch (T*={best_T} m)', fontsize=10)
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig('analysis/method_comparison_output/dynamic_ap_switch_backtest.png', dpi=150)
plt.close(fig)
print('\nSaved figures to analysis/method_comparison_output/')
