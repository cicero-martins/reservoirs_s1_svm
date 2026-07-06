"""
diag_chenderoh_sarrans.py

Two targeted diagnostics:
  Chenderoh — are the ±100 ha JRC swings a REFERENCE artifact (cloud, equatorial Malaysia)
    or real operation? Test: do the swings coincide with lower valid_frac, and does the SAR
    (adapt & vv) reproduce them or stay smooth? If JRC jumps while SAR is smooth → JRC noise.
  Sarrans — is there a temporal SHIFT between JRC and SAR? Test: cross-correlation of the
    monthly series over lags −4..+4 months; report the lag that maximises r and the KGE there.
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
DIRS = {'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
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

def rawjrc(n):
    for d in JRC:
        c = sorted(d.glob(f'JRC_area_{n}*.csv'))
        pl = [p for p in c if not _re.search(r'\s*\(\d+\)', p.stem)]
        p = pl[0] if pl else (c[0] if c else None)
        if p: return pd.read_csv(p, parse_dates=['date'])
    return None

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(17, 9))

# ── Chenderoh ─────────────────────────────────────────────────────────────────
n = 'Chenderoh'
raw = rawjrc(n); raw = raw[raw['date'] <= '2021-12-31'].sort_values('date')
jm = load_jrc(n, JRC, despike=True); jm['ym'] = jm['date'].dt.to_period('M')
ad = sar_monthly(DIRS['adapt'], n); vv = sar_monthly(DIRS['vv'], n)
ax = axes[0, 0]
sc = ax.scatter(raw['date'], raw['jrc_area_ha'], c=raw['valid_frac'], cmap='RdYlGn',
                vmin=0.5, vmax=1, s=30, zorder=3, edgecolors='#999', linewidths=.3, label='raw JRC (col=vf)')
ax.plot(jm['date'], jm['jrc_area_ha'], 'k-', lw=1, zorder=4, label='JRC de-spiked (vf≥.9)')
if ad is not None: ax.plot(ad['ym'].dt.to_timestamp(), ad['area'], color='#1f77b4', lw=1.6, label='adapt SVM')
if vv is not None: ax.plot(vv['ym'].dt.to_timestamp(), vv['area'], color='#2ca02c', lw=1.6, label='VV-Otsu')
ax.set_title(f'{n}: JRC (col by valid_frac) vs SAR', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.set_ylabel('area (ha)')

# Chenderoh artifact test: JRC month-to-month |change| vs valid_frac of the swinging point
jj = jm.sort_values('date').reset_index(drop=True)
jj['dch'] = jj['jrc_area_ha'].diff().abs()
merged_vf = jj.merge(raw[['date', 'valid_frac']], on='date', how='left')
ax = axes[0, 1]
ax.scatter(merged_vf['valid_frac'], merged_vf['dch'], s=30, color='#8a2d04')
ax.set_xlabel('valid_frac of the point'); ax.set_ylabel('|Δ JRC area| vs previous (ha)')
# variance comparison
sd_j = jm['jrc_area_ha'].std()
sd_a = ad['area'].std() if ad is not None else np.nan
sd_v = vv['area'].std() if vv is not None else np.nan
ax.set_title(f'{n}: JRC std={sd_j:.0f} ha  vs  adapt std={sd_a:.0f}  vv std={sd_v:.0f}\n'
             f'(JRC swings {sd_j/max(sd_a,1e-9):.1f}× the SAR → reference noise)', fontweight='bold')
ax.grid(alpha=.25)
print(f'=== Chenderoh ===  JRC std={sd_j:.0f}  adapt std={sd_a:.0f}  vv std={sd_v:.0f}  '
      f'(ratio JRC/adapt={sd_j/max(sd_a,1e-9):.1f})')
big = merged_vf[merged_vf['dch'] > 60]
print(f'  big JRC jumps (>60 ha): {len(big)} — their valid_frac: ' +
      ', '.join(f"{v:.2f}" for v in big['valid_frac'].dropna()))

# ── Sarrans lag test ──────────────────────────────────────────────────────────
n = 'Sarrans'
jm = load_jrc(n, JRC, despike=True); jm['ym'] = jm['date'].dt.to_period('M')
ad = sar_monthly(DIRS['adapt'], n); vv = sar_monthly(DIRS['vv'], n)
ax = axes[1, 0]
ax.plot(jm['date'], jm['jrc_area_ha'], 'k.-', lw=1, ms=4, label='JRC de-spiked')
if ad is not None: ax.plot(ad['ym'].dt.to_timestamp(), ad['area'], color='#1f77b4', lw=1.6, label='adapt SVM')
if vv is not None: ax.plot(vv['ym'].dt.to_timestamp(), vv['area'], color='#2ca02c', lw=1.6, label='VV-Otsu')
ax.set_title(f'{n}: JRC vs SAR (look for a temporal shift)', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.set_ylabel('area (ha)')

# cross-correlation over lags for both methods
ax = axes[1, 1]
jser = jm.set_index('ym')['jrc_area_ha']
for meth, sm, col in [('adapt', ad, '#1f77b4'), ('vv', vv, '#2ca02c')]:
    if sm is None: continue
    sser = sm.set_index('ym')['area']
    lags = range(-4, 5); rs = []
    for L in lags:
        s_sh = sser.copy(); s_sh.index = s_sh.index + L      # shift SAR forward by L months
        m = pd.concat([jser, s_sh], axis=1, join='inner').dropna()
        rs.append(stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])[0] if len(m) >= 8 else np.nan)
    rs = np.array(rs); best = list(lags)[int(np.nanargmax(rs))]
    ax.plot(list(lags), rs, 'o-', color=col, label=f'{meth}: best lag={best:+d} mo (r={np.nanmax(rs):.2f})')
    # KGE at lag 0 vs best lag
    def kge_at(L):
        s_sh = sser.copy(); s_sh.index = s_sh.index + L
        m = pd.concat([jser, s_sh], axis=1, join='inner').dropna()
        if len(m) < 12: return np.nan
        o, s = m.iloc[:, 0].values, m.iloc[:, 1].values
        r = stats.pearsonr(o, s)[0]
        return 1 - np.sqrt((r-1)**2 + (np.std(s)/np.std(o)-1)**2 + (np.mean(s)/np.mean(o)-1)**2)
    print(f'=== Sarrans {meth} ===  best lag={best:+d} mo  r@0={rs[4]:.2f} r@best={np.nanmax(rs):.2f}  '
          f'KGE@0={kge_at(0):.2f}  KGE@best={kge_at(best):.2f}')
ax.axvline(0, color='gray', ls=':'); ax.set_xlabel('SAR lag vs JRC (months)'); ax.set_ylabel('Pearson r')
ax.set_title('Sarrans: cross-correlation vs lag\n(peak away from 0 → temporal shift)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.25)

fig.suptitle('Chenderoh (JRC noise?) & Sarrans (temporal shift?) — targeted diagnostics',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT = pathlib.Path('analysis/method_comparison_output/diag_chenderoh_sarrans.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'\nSaved: {OUT}')
