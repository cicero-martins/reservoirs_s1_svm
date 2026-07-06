"""
improve_jrc_test.py

Hypothesis: many low KGEs are driven by SPIKES in the JRC reference (cloud / partial-
coverage months) that survive the current filter (valid_frac≥0.80 + global 2.5σ clip),
NOT by SAR failure. The SAR side already gets full clean+smooth+local-outlier removal;
the JRC side does not — an asymmetry. Test a SYMMETRIC JRC filter and measure ΔKGE.

OLD JRC filter : valid_frac ≥ 0.80, then global 2.5σ clip.
NEW JRC filter : valid_frac ≥ 0.90, then LOCAL-outlier spike removal (rolling-median MAD),
                 the same idea used on the SAR series.

Recomputes KGE(dual vs JRC) OLD vs NEW for all reservoirs; reports the rescue for the
low-KGE set; and plots before/after JRC for a flagged subset.

Output: analysis/jrc_refilter_kge.csv  +  method_comparison_output/jrc_refilter_before_after.png
"""
import pathlib, re as _re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

# ── SAR clean+smooth (unchanged, identical to compute_kge_4way) ───────────────
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
SAR = _P('raw_data/GEE_GlobalPilotV4', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
         'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
VF_OLD, VF_NEW = 0.80, 0.90
SAR_MIN_FRAC, MIN_PAIRS, AREA_MIN = 0.02, 12, {'Saint_Cassien': 200}

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None
def _jrcp(n):
    for d in JRC:
        c = sorted(d.glob(f'JRC_area_{n}*.csv'))
        pl = [p for p in c if not _re.search(r'\s*\(\d+\)', p.stem)]
        p = pl[0] if pl else (c[0] if c else None)
        if p: return p
    return None

def load_sar_monthly(n):
    p = _res(SAR, f'SAR_area_{n}.csv')
    if p is None: return None, np.nan
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None, np.nan
    if df.empty: return None, np.nan
    ap = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan
    df = df[df['area_ha'] > 0].sort_values('date')
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None, ap
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if df.empty: return None, ap
    df = sar_clean(df.reset_index(drop=True))
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index(); m.columns = ['ym', 'sar']
    return m, ap

def jrc_local_despike(a, vf=None, k=4.0, minrel=0.12, vf_gate=0.95):
    """Remove ISOLATED single-month spikes, GATED on valid_frac (contamination evidence).
    A point is dropped only if it is BOTH (i) an isolated excursion from the linear
    interpolation of its neighbours AND (ii) its valid_frac is below vf_gate — i.e. the
    month had partial/cloudy coverage that plausibly caused the bad area. Real drawdowns
    happen at near-full coverage (vf≈1) → kept even if they look like excursions."""
    a = np.asarray(a, float); n = len(a)
    if n < 5: return np.ones(n, bool)
    interp = a.copy(); interp[1:-1] = 0.5 * (a[:-2] + a[2:])
    resid = np.abs(a - interp); resid[0] = resid[-1] = 0.0
    scale = np.median(resid[resid > 0]) if np.any(resid > 0) else 1.0
    thr = max(k * scale, minrel * np.median(a))
    is_spike = resid > thr
    if vf is not None:
        vf = np.asarray(vf, float)
        is_spike &= (vf < vf_gate)   # gate: only contaminated months
    return ~is_spike

def load_jrc_monthly(n, mode='old'):
    p = _jrcp(n)
    if p is None: return None
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None
    df = df.sort_values('date').reset_index(drop=True)
    vf = VF_OLD if mode == 'old' else VF_NEW
    if 'valid_frac' in df.columns: df = df[df['valid_frac'] >= vf].copy()
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None
    if mode == 'old':
        m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
        if sd > 0: df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd]
    else:
        m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
        if sd > 0: df = df[np.abs(df['jrc_area_ha'] - m) <= 3.0 * sd]  # loose global net
        vf_arr = df['valid_frac'].values if 'valid_frac' in df.columns else None
        keep = jrc_local_despike(df['jrc_area_ha'].values, vf=vf_arr)  # LOCAL spikes, vf-gated
        df = df[keep]
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()

def kge(o, s):
    if len(o) < 3 or np.std(o) == 0 or np.std(s) == 0: return np.nan, np.nan
    r = stats.pearsonr(o, s)[0]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (np.mean(s) / np.mean(o) - 1) ** 2), r

diag = pd.read_csv('analysis/low_kge_diagnosis.csv')
rows = []
for n in diag['name']:
    sar, ap = load_sar_monthly(n)
    if sar is None: continue
    out = {'name': n, 'ap_m': ap}
    for mode in ('old', 'new'):
        jr = load_jrc_monthly(n, mode)
        if jr is None: out[f'kge_{mode}'] = np.nan; out[f'r_{mode}'] = np.nan; out[f'n_{mode}'] = 0; continue
        mg = pd.merge(sar, jr, on='ym').dropna()
        if len(mg) < MIN_PAIRS: out[f'kge_{mode}'] = np.nan; out[f'r_{mode}'] = np.nan; out[f'n_{mode}'] = len(mg); continue
        k, r = kge(mg['jrc_area_ha'].values, mg['sar'].values)
        out[f'kge_{mode}'] = round(k, 3); out[f'r_{mode}'] = round(r, 3); out[f'n_{mode}'] = len(mg)
    rows.append(out)

df = pd.DataFrame(rows)
df['dkge'] = df['kge_new'] - df['kge_old']
df.to_csv('analysis/jrc_refilter_kge.csv', index=False)

print('=== JRC re-filter effect (dual vs JRC) ===')
print(f'reservoirs recomputed: {len(df)}')
low = df[df['kge_old'] < 0.5].sort_values('dkge', ascending=False)
print(f'\nlow-KGE-old set (N={len(low)}), ΔKGE from symmetric JRC de-spiking:')
print(low[['name', 'ap_m', 'kge_old', 'kge_new', 'dkge', 'r_old', 'r_new', 'n_old', 'n_new']].to_string(index=False))
resc = low[(low['kge_old'] < 0.5) & (low['kge_new'] >= 0.5)]
print(f'\nRESCUED across 0.5 (KGE_old<0.5 → KGE_new≥0.5): {len(resc)}  {list(resc["name"])}')
print(f'median ΔKGE (low set) = {low["dkge"].median():+.3f}   mean = {low["dkge"].mean():+.3f}')
print(f'median ΔKGE (all)     = {df["dkge"].median():+.3f}')

# ── before/after JRC for a flagged subset ─────────────────────────────────────
FLAG = ['Rockport', 'Baisha', 'Hubbard_Creek', 'Bilancino', 'Castillon', 'El_Atazar', 'Egorlyskaia', 'Panneciere']
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(20, 8)); axes = axes.ravel()
for ax, n in zip(axes, FLAG):
    p = _jrcp(n)
    if p is None: ax.axis('off'); continue
    raw = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    raw = raw[raw['date'] <= '2021-12-31']
    old = load_jrc_monthly(n, 'old'); new = load_jrc_monthly(n, 'new')
    ax.plot(raw['date'], raw['jrc_area_ha'], '.', color='#ccc', ms=4, label='raw JRC (all vf)')
    if old is not None:
        od = old.copy(); od['date'] = od['ym'].dt.to_timestamp()
        ax.plot(od['date'], od['jrc_area_ha'], 'o-', color='#d62728', ms=3, lw=0.8, label='OLD filter')
    if new is not None:
        nd = new.copy(); nd['date'] = nd['ym'].dt.to_timestamp()
        ax.plot(nd['date'], nd['jrc_area_ha'], 'o-', color='#1f77b4', ms=3, lw=0.8, label='NEW (vf≥.9+despike)')
    ko = df.loc[df.name == n, 'kge_old'].values; kn = df.loc[df.name == n, 'kge_new'].values
    ttl = f"{n.replace('_',' ')}"
    if len(ko) and len(kn): ttl += f"  KGE {ko[0]:+.2f}→{kn[0]:+.2f}"
    ax.set_title(ttl, fontsize=10, fontweight='bold'); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7)
fig.suptitle('JRC reference: raw vs OLD filter vs NEW (valid_frac≥0.90 + local de-spike)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT = pathlib.Path('analysis/method_comparison_output/jrc_refilter_before_after.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'\nSaved: analysis/jrc_refilter_kge.csv  and  {OUT}')
