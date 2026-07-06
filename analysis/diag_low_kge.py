"""
diag_low_kge.py

Diagnose WHY reservoirs score low KGE (dual vs JRC) by decomposing KGE into its three
terms and attributing the shortfall:
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
      r      = Pearson corr (temporal tracking)      low  -> radiometric noise / timing
      alpha  = std_sar/std_jrc (variability ratio)   off  -> series too flat / too spiky
      beta   = mean_sar/mean_jrc (bias ratio)         off  -> over/under-detection; polygon/AOI

Also reports JRC cv (reference dynamics): a near-flat JRC (cv small) makes KGE
uninformative rather than a genuine "failure".

Same loader/pipeline as compute_kge_apcurve.py (dual vs JRC, own months, 42+coverage set).

Output: analysis/low_kge_diagnosis.csv  (+ printed table for KGE < THRESH)
        analysis/method_comparison_output/low_kge_components.png
"""
import pathlib, re as _re, sys
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
THRESH = 0.5

# ── clean+smooth (identical to compute_kge_apcurve.py) ────────────────────────
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
def clean_and_smooth(df, col='area_ha'):
    s = df[col].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    return pd.DataFrame({'date': ds, 'area_ha': _lw(ds, s.reset_index(drop=True), 20, 7)})

def _P(*p): return [pathlib.Path(x) for x in p]
V4_SAR = _P('raw_data/GEE_GlobalPilotV4', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
V4_JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
V2C_SAR = _P('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
V2C_JRC = _P('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
V3_SICILY = {'Ancipa', 'Poma', 'Pozzillo', 'Rosamarina'}
V3_OTHER = {'Yesa', 'Caia', 'Forggen', 'Garcia', 'Hubbard_Creek', 'Harlan_County',
            'Umbuluzi', 'Erfenis', 'Paraibuna', 'Contas'}
V4_EXCLUDE = {'Oued_Makhazine', 'Guajaraz', 'Antero', 'Miyagase', 'Welbedacht', 'Tzaneen'}
VALID_FRAC_MIN, SAR_MIN_FRAC, MIN_PAIRS, AREA_MIN = 0.80, 0.02, 12, {'Saint_Cassien': 200}

cand = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
_V4C = set(cand['name']); CLIM = dict(zip(cand['name'], cand['climate_zone']))
RESV = sorted(((_V4C - (V3_SICILY | V3_OTHER)) - V4_EXCLUDE) | V3_SICILY | V3_OTHER)

def sources(n):
    if n in V3_SICILY: return V2C_SAR, V2C_JRC, 'v3'
    if n in V3_OTHER:  return V4_SAR + V2C_SAR, V4_JRC + V2C_JRC, 'v3'
    return V4_SAR, V4_JRC, ('v4' if n in {'Sau','Susqueda','El_Atazar','Siurana','Bleiloch','Rappbode','Castillon','Saint_Cassien','Salto','Turano','Katse','Mohale','Blyde','Cachi','Miyagase','Yamba','El_Burguillo','Boadella','Puentes_Viejas','Panneciere','Sarrans','Bilancino','Cecita','Karapuzha','Saguaro','Boegoeberg','Woodstock','Googong','Cardinia','Triouzoune','Grandval','Deer_Creek','East_Canyon','Pineview','Rockport','Shaharchay','Occhito'} else 'new')

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None
def _jrc(dirs, n):
    for d in dirs:
        c = sorted(d.glob(f'JRC_area_{n}*.csv'))
        pl = [p for p in c if not _re.search(r'\s*\(\d+\)', p.stem)]
        h = pl[0] if pl else (c[0] if c else None)
        if h: return h
    return None
def load_sar(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None, np.nan
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None, np.nan
    if df.empty: return None, np.nan
    ap = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None, ap
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if df.empty: return None, ap
    df = clean_and_smooth(df.reset_index(drop=True))
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index(); m.columns = ['ym', 'sar']
    return m, ap
def load_jrc(dirs, n):
    p = _jrc(dirs, n)
    if p is None: return None
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None
    if 'valid_frac' in df.columns: df = df[df['valid_frac'] >= VALID_FRAC_MIN]
    if df.empty: return None
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if sd > 0: df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd]
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()

rows = []
for n in RESV:
    sd, jd, coh = sources(n)
    sar, ap = load_sar(sd, n); jrc = load_jrc(jd, n)
    if sar is None or jrc is None: continue
    mg = pd.merge(sar, jrc, on='ym').dropna()
    if len(mg) < MIN_PAIRS: continue
    o, s = mg['jrc_area_ha'].values, mg['sar'].values
    if np.std(o) == 0 or np.std(s) == 0: continue
    r = stats.pearsonr(o, s)[0]; al = np.std(s) / np.std(o); be = np.mean(s) / np.mean(o)
    kge = 1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2)
    jrc_cv = np.std(o) / np.mean(o)
    terms = {'r': (r - 1) ** 2, 'alpha': (al - 1) ** 2, 'beta': (be - 1) ** 2}
    dom = max(terms, key=terms.get)
    rows.append({'name': n, 'cohort': coh, 'ap_m': round(ap, 0), 'kge': round(kge, 3),
                 'r': round(r, 3), 'alpha': round(al, 2), 'beta': round(be, 2),
                 'jrc_cv': round(jrc_cv, 2), 'n': len(mg), 'dom_fail': dom,
                 'climate': CLIM.get(n, '?')})

df = pd.DataFrame(rows).sort_values('kge').reset_index(drop=True)
df.to_csv('analysis/low_kge_diagnosis.csv', index=False)

def cause(row):
    tags = []
    if row['dom_fail'] == 'beta':
        tags.append('OVER-detect (β>1, polygon/threshold?)' if row['beta'] > 1.2
                    else 'UNDER-detect (β<1, drying/mask?)' if row['beta'] < 0.8 else 'mild bias')
    if row['dom_fail'] == 'alpha':
        tags.append('SAR too SPIKY (α>1)' if row['alpha'] > 1.3 else 'SAR too FLAT (α<1)')
    if row['r'] < 0.5:
        tags.append('poor tracking (r<0.5: radiometric/timing)')
    if row['ap_m'] < 100:
        tags.append('low A/P (geometric, expected)')
    if row['jrc_cv'] < 0.05:
        tags.append('flat JRC (KGE uninformative)')
    return '; '.join(tags) or '—'

low = df[df['kge'] < THRESH].copy()
low['likely_cause'] = low.apply(cause, axis=1)
print(f'{len(low)}/{len(df)} reservoirs with KGE < {THRESH}\n')
print(low[['name', 'cohort', 'ap_m', 'kge', 'r', 'alpha', 'beta', 'jrc_cv', 'dom_fail',
           'climate']].to_string(index=False))
print('\n=== likely cause per low reservoir ===')
for _, r_ in low.iterrows():
    print(f"  {r_['name']:<22} KGE={r_['kge']:+.2f}  -> {r_['likely_cause']}")
print('\n=== dominant failing term among the low set ===')
print(low['dom_fail'].value_counts().to_string())

# ── figure: KGE vs A/P, marker by dominant failing term ───────────────────────
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
MK = {'r': ('o', '#d62728', 'r (tracking)'), 'alpha': ('^', '#1f77b4', 'α (variance)'),
      'beta': ('s', '#2ca02c', 'β (bias)')}
fig, ax = plt.subplots(figsize=(10, 6.5))
for term, (mk, col, lab) in MK.items():
    s = df[df['dom_fail'] == term]
    ax.scatter(s['ap_m'], s['kge'], marker=mk, c=col, s=55, edgecolors='white',
               linewidths=0.6, label=f'dominant fail: {lab}', zorder=4, alpha=0.9)
ax.axhline(THRESH, color='k', ls='--', lw=1, alpha=0.6, label=f'KGE={THRESH}')
for _, r_ in df[df['kge'] < THRESH].iterrows():
    ax.annotate(r_['name'].replace('_', ' '), (r_['ap_m'], r_['kge']), fontsize=6,
                xytext=(3, 2), textcoords='offset points', color='#444')
ax.set_xlabel('A/P (m)'); ax.set_ylabel('KGE dual vs JRC')
ax.set_title('Low-KGE diagnosis: KGE vs A/P, coloured by dominant failing KGE term',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
OUT = pathlib.Path('analysis/method_comparison_output/low_kge_components.png')
fig.savefig(OUT, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'\nSaved: analysis/low_kge_diagnosis.csv  and  {OUT}')
