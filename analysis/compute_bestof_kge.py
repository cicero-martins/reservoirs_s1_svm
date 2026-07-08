"""
compute_bestof_kge.py

(1) VERIFY that KGE is computed only on the JRC-overlap period (inner join on month),
    not the full SAR period: for each reservoir report n_pairs, the KGE date span, the
    SAR span, and the JRC-valid span — KGE span must sit INSIDE the JRC span.
(2) Compute per-method KGE vs JRC (adapt / dual-fixed / vv-Otsu) on each method's OWN
    overlap months, then BEST-OF and the winning method — so cases where VV-Otsu equals
    or beats the SVM are made explicit.

Output: analysis/bestof_kge.csv  (name, ap_m, kge_adapt/dual/vv, best, winner, n_pairs, spans)
"""
import pathlib, re as _re, sys
import numpy as np, pandas as pd
from scipy import stats
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
def sar_clean(df):
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    return pd.DataFrame({'date': ds, 'area_ha': _lw(ds, s.reset_index(drop=True), 20, 7)})

def _P(*p): return [pathlib.Path(x) for x in p]
DIRS = {
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'dual':  _P('raw_data/GEE_GlobalPilotV4', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4',
                'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
}
JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
VF, SAR_MIN_FRAC, MIN_PAIRS, AREA_MIN = 0.80, 0.02, 12, {'Saint_Cassien': 200}

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None
def load_sar(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None, np.nan, (None, None)
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None, np.nan, (None, None)
    if df.empty: return None, np.nan, (None, None)
    ap = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan
    df = df[df['area_ha'] > 0].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None, ap, (None, None)
    span = (df['date'].min(), df['date'].max())
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if df.empty: return None, ap, span
    df = sar_clean(df.reset_index(drop=True)); df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index(); m.columns = ['ym', 'sar']
    return m, ap, span
from jrc_filter import load_jrc_monthly as _load_jrc_shared
DESPIKE_JRC = True   # validated valid_frac-gated de-spike (jrc_filter.py)

def load_jrc(n):
    df = _load_jrc_shared(n, JRC, despike=DESPIKE_JRC)
    if df is None or df.empty:
        return None, (None, None)
    span = (df['date'].min(), df['date'].max())
    df = df.copy(); df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna(), span
def kge(o, s):
    if len(o) < 3 or np.std(o) == 0 or np.std(s) == 0: return np.nan
    r = stats.pearsonr(o, s)[0]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (np.mean(s) / np.mean(o) - 1) ** 2)

# Name source: the curated master candidate list (kept in sync with every export
# batch), NOT low_kge_diagnosis.csv (a one-off diagnostic snapshot that silently
# goes stale whenever new reservoirs are added — it missed all 15 from the
# Temperate/dip-bin expansion on 7-8 Jul 2026). The 4 Sicilian (JRC-period
# dual-only) are appended explicitly: they drop out naturally below (best=NaN)
# but are needed elsewhere (PlanetScope near-truth track).
names = pd.read_csv('analysis/global_pilot_v4_candidates.csv')['name'].tolist()
names += ['Ancipa', 'Poma', 'Pozzillo', 'Rosamarina']
rows, checks = [], []
for n in names:
    jr, jspan = load_jrc(n)
    if jr is None: continue
    row = {'name': n}; ap = np.nan; kge_span = None; npairs = 0
    for meth, dirs in DIRS.items():
        sar, a, sspan = load_sar(dirs, n)
        if not np.isnan(a): ap = a
        if sar is None: row[f'kge_{meth}'] = np.nan; continue
        mg = pd.merge(sar, jr, on='ym').dropna()
        if len(mg) < MIN_PAIRS: row[f'kge_{meth}'] = np.nan; continue
        row[f'kge_{meth}'] = round(kge(mg['jrc_area_ha'].values, mg['sar'].values), 3)
        if meth == 'dual':
            kge_span = (mg['ym'].min().to_timestamp(), mg['ym'].max().to_timestamp())
            npairs = len(mg); dual_sspan = sspan
    row['ap_m'] = round(ap, 0)
    # best-of is over the JUSTIFIED methods only: adapt (per-scene SVM) + vv (Otsu).
    # dual-FIXED (arbitrary 2023 baseline) is kept in the CSV for reference but EXCLUDED
    # from best/winner — it is retired from the story. Reservoirs with neither adapt nor vv
    # (the 4 Sicilian, JRC period = dual-only) get best=NaN and drop out of the JRC analysis
    # (they live in the PlanetScope near-truth validation instead).
    ks = {m: row[f'kge_{m}'] for m in ['adapt', 'vv'] if pd.notna(row.get(f'kge_{m}'))}
    if ks:
        win = max(ks, key=ks.get); row['best'] = round(ks[win], 3); row['winner'] = win
    else:
        row['best'] = np.nan; row['winner'] = None
    row['n_pairs'] = npairs
    rows.append(row)
    # verification: KGE period must sit inside JRC valid span
    if kge_span and jspan[0] is not None:
        checks.append({'name': n, 'kge_from': kge_span[0], 'kge_to': kge_span[1],
                       'sar_from': dual_sspan[0], 'jrc_from': jspan[0], 'jrc_to': jspan[1],
                       'inside_jrc': kge_span[0] >= jspan[0] and kge_span[1] <= jspan[1]})

# flat-JRC reservoirs ("chapados") where the reference barely varies → KGE is
# mathematically ill-posed (degenerate r). Flagged for exclusion downstream.
CHAPADO = {'Egorlyskaia', 'Boegoeberg', 'Itauba', 'Saguaro'}
df = pd.DataFrame(rows)
df['chapado'] = df['name'].isin(CHAPADO)
df.to_csv('analysis/bestof_kge.csv', index=False)

chk = pd.DataFrame(checks)
print('=== VERIFICATION: is KGE limited to the JRC-overlap period? ===')
print(f'  reservoirs checked: {len(chk)}')
print(f'  KGE window inside JRC valid span: {chk["inside_jrc"].sum()}/{len(chk)}')
bad = chk[~chk['inside_jrc']]
if len(bad): print('  OUTSIDE (investigate):', bad['name'].tolist())
ex = chk.iloc[0]
print(f"  example {ex['name']}: SAR starts {ex['sar_from']:%Y-%m}, JRC {ex['jrc_from']:%Y-%m}..{ex['jrc_to']:%Y-%m}, "
      f"KGE uses {ex['kge_from']:%Y-%m}..{ex['kge_to']:%Y-%m}")
print(f'  → KGE is computed ONLY on months with valid JRC (inner-join), NOT the full SAR period.\n')

print('=== BEST-OF winners ===')
print(df['winner'].value_counts().to_string())
print(f"\nVV-Otsu is the BEST method for {int((df.winner=='vv').sum())} reservoirs "
      f"(where 1-band Otsu ties/beats the 2-band SVM):")
print(df[df.winner == 'vv'].sort_values('ap_m')[['name', 'ap_m', 'kge_vv', 'kge_adapt', 'kge_dual']].to_string(index=False))
print(f"\nbest-of: median={df['best'].median():.3f}  KGE<0.5={int((df['best']<0.5).sum())}/{len(df)}")
print('Saved: analysis/bestof_kge.csv')
