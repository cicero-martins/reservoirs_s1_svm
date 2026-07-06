"""
screen_reference_noise.py

Size the "JRC reference is noisier than the SAR" phenomenon across the dataset (option b:
keep everything, report as a limitation of the optical reference — which favours SAR).

Signature (from the Chenderoh case): the JRC reference swings month-to-month while two
independent SAR methods stay smooth, at HIGH valid_frac (so it is misclassification, not
coverage). Metric = high-frequency ROUGHNESS ratio on the common months:

    rough_ratio = std(ΔJRC month-to-month) / std(ΔSAR month-to-month)

Real dynamics are smooth (both differ slowly → ratio ~1); reference noise makes JRC jumpy
while SAR is smooth → ratio ≫ 1. SAR method = the reservoir's best-of winner.

Reads bestof_kge.csv (name, winner, ap_m) + candidates (climate). Flags rough_ratio ≥ 2.5.
Output: analysis/reference_noise.csv + method_comparison_output/reference_noise.png
"""
import pathlib, re as _re, sys
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
    if p is None: return None
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[(df['area_ha'] > 0) & (df['date'] <= '2021-12-31')].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if len(df) < 5: return None
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    sm = pd.DataFrame({'date': ds, 'area': _lw(ds, s.reset_index(drop=True), 20, 7)})
    sm['ym'] = sm['date'].dt.to_period('M')
    return sm.groupby('ym')['area'].mean().reset_index()

bof = pd.read_csv('analysis/bestof_kge.csv').dropna(subset=['best', 'winner'])
clim = dict(zip(pd.read_csv('analysis/global_pilot_v4_candidates.csv')['name'],
                pd.read_csv('analysis/global_pilot_v4_candidates.csv')['climate_zone']))
def family(z):
    z = str(z)
    if 'Mediterranean' in z: return 'Mediterranean'
    if 'Semi-arid' in z or 'arid' in z: return 'Semi-arid/arid'
    if 'temperate' in z or 'continental' in z: return 'Temperate/continental'
    if any(k in z for k in ('subtropical', 'tropical', 'Tropical')): return '(Sub)tropical'
    return 'Other'

rows = []
for _, b in bof.iterrows():
    n, w = b['name'], b['winner']
    jr = load_jrc(n, JRC, despike=True)
    if jr is None: continue
    jr = jr.copy(); jr['ym'] = jr['date'].dt.to_period('M')
    sm = sar_monthly(MDIR[w], n)
    if sm is None: continue
    m = jr[['ym', 'jrc_area_ha']].merge(sm.rename(columns={'area': 'sar'}), on='ym').dropna().sort_values('ym')
    if len(m) < 15: continue
    dj = m['jrc_area_ha'].diff().dropna(); ds_ = m['sar'].diff().dropna()
    rj, rs = dj.std(), ds_.std()
    if rs <= 1e-6: continue
    r = stats.pearsonr(m['jrc_area_ha'], m['sar'])[0]
    rows.append({'name': n, 'ap_m': b['ap_m'], 'winner': w, 'best': b['best'],
                 'rough_jrc': round(rj, 1), 'rough_sar': round(rs, 1),
                 'rough_ratio': round(rj / rs, 2), 'r': round(r, 3),
                 'biome': family(clim.get(n, '?'))})

df = pd.DataFrame(rows).sort_values('rough_ratio', ascending=False).reset_index(drop=True)
df['ref_noise'] = df['rough_ratio'] >= 2.5
df.to_csv('analysis/reference_noise.csv', index=False)

print(f'N={len(df)}  |  reference-noise flagged (rough_ratio≥2.5): {int(df.ref_noise.sum())}\n')
print(df[df.ref_noise][['name', 'biome', 'ap_m', 'rough_ratio', 'r', 'best']].to_string(index=False))
print('\n=== rough_ratio by biome (median) — is reference noise concentrated in the humid tropics? ===')
print(df.groupby('biome')['rough_ratio'].agg(['median', 'max', 'count']).round(2).to_string())
print(f'\nof the {int(df.ref_noise.sum())} flagged, biome counts:')
print(df[df.ref_noise]['biome'].value_counts().to_string())

# ── figure ────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BCOL = {'Mediterranean': '#1f77b4', 'Semi-arid/arid': '#d62728',
        'Temperate/continental': '#2ca02c', '(Sub)tropical': '#ff7f0e', 'Other': '#999'}
fig, ax = plt.subplots(figsize=(12, 6))
for bm, c in BCOL.items():
    s = df[df.biome == bm]
    if s.empty: continue
    ax.scatter(s['ap_m'], s['rough_ratio'], color=c, s=60, edgecolors='white',
               linewidths=.6, label=f'{bm} (n={len(s)})', zorder=4)
ax.axhline(2.5, color='k', ls='--', lw=1, alpha=.6, label='ref-noise flag (≥2.5)')
ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=.6)
for _, r_ in df[df.ref_noise].iterrows():
    ax.annotate(r_['name'].replace('_', ' '), (r_['ap_m'], r_['rough_ratio']), fontsize=7,
                xytext=(3, 2), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('A/P (m)')
ax.set_ylabel('roughness ratio  std(ΔJRC) / std(ΔSAR)  [log]')
ax.set_title('Reference-noise screen: where does JRC jump while SAR stays smooth?\n'
             '(high ratio = optical reference unreliable — clusters in humid/tropical)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.25, which='both')
OUT = pathlib.Path('analysis/method_comparison_output/reference_noise.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'\nSaved: analysis/reference_noise.csv  and  {OUT}')
