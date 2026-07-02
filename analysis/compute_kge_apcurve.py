"""
compute_kge_apcurve.py

Pooled A/P → KGE curve for the HEADLINE method (dual-pol VV+VH SVM, fixed training)
against JRC, across BOTH pilot cohorts on ONE identical pipeline:

  v3 : raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2(_JRC)   (14, incl. the 4 Sicilian)
  v4 : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4(_JRC)   (28, method-comparison cohort)

Purpose: show the A/P→performance relationship is a single continuous trend across
42 reservoirs from two independently-selected cohorts — i.e. the study area was NOT
tuned to the result. v3 fills the HIGH-A/P tail (up to ~460 m) that v4 under-samples;
together they span the full geometric range the paper's thesis is about.

Both cohorts are processed with the SAME loader (clean+smooth, VALID_FRAC≥0.80,
SAR≥2% p99, ≤2021-12-31, JRC 2.5σ clip, ≥12 common months) so KGE is comparable.
KGE here is dual-vs-JRC on the method's OWN months (independent skill), NOT the
common-month ΔKGE of compute_kge_4way.py.

Output:
  analysis/pilot_kge_apcurve.csv
  analysis/method_comparison_output/ap_kge_curve_pooled.png
"""

import pathlib
import re as _re
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

# ── clean+smooth (identical to compute_kge_4way.py) ───────────────────────────
def _remove_global(s, threshold=2.0):
    m, sd = s.mean(), s.std()
    return s[np.abs(s - m) <= threshold * sd]

def _remove_local(s, window=5, threshold=1.5):
    arr, idx = s.values.copy(), s.index.tolist()
    keep, half = [], window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        m, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - m) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]

def _lowess(dates, values, window_days=20, bandwidth=7):
    out = []
    for t0 in dates:
        dd = np.abs((dates - t0).dt.total_seconds().values / 86400)
        mask = dd <= window_days
        w = np.exp(-(dd[mask] / bandwidth) ** 2)
        out.append(float((values[mask] * w).sum() / w.sum()))
    return np.array(out)

def clean_and_smooth(df, col='area_ha'):
    s = df[col].copy()
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 10, 1.5)
    dates_s = df.loc[s.index, 'date'].reset_index(drop=True)
    sm = _lowess(dates_s, s.reset_index(drop=True), 20, 7)
    return pd.DataFrame({'date': dates_s, 'area_ha': sm})


# ── cohorts ───────────────────────────────────────────────────────────────────
# The 42-reservoir study area = v4 cohort (28) + v3 cohort (14). Both are validated
# dual-vs-JRC on ONE pipeline. Source resolution per reservoir:
#   v4 (28)            → V4 export folders (nested-or-top-level fallback)
#   v3 non-Sicilian(10)→ PREFER the V4 re-export (scene-consistent with v4); fall back
#                        to the original V2c export until that download lands
#   v3 Sicilian (4)    → V2c only (they are method-compared vs PlanetScope, not re-run)
# So this figure shows 42 both BEFORE and AFTER the v3 re-export — the v3 dual just
# swaps from V2c to the V4-consistent export once present.
def _P(*parts):
    return [pathlib.Path(p) for p in parts]

V3_SICILY = {'Ancipa', 'Poma', 'Pozzillo', 'Rosamarina'}
V3_OTHER  = {'Yesa', 'Caia', 'Forggen', 'Garcia', 'Hubbard_Creek', 'Harlan_County',
             'Umbuluzi', 'Erfenis', 'Paraibuna', 'Contas'}
V3_NAMES  = V3_SICILY | V3_OTHER
V4_EXCLUDE = {'Oued_Makhazine', 'Guajaraz', 'Antero', 'Miyagase', 'Welbedacht', 'Tzaneen'}

V4_SAR = _P('raw_data/GEE_GlobalPilotV4', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
V4_JRC = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
V2C_SAR = _P('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2')
V2C_JRC = _P('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')

_V4_CAND = set(pd.read_csv('analysis/global_pilot_v4_candidates.csv')['name'])
V4_ONLY  = (_V4_CAND - V3_NAMES) - V4_EXCLUDE            # the 28
RESERVOIRS = sorted(V4_ONLY | V3_NAMES)                 # 28 + 14 = 42 target


def sources(name):
    """(sar_dirs, jrc_dirs, cohort) with the fallback logic described above."""
    if name in V3_SICILY:
        return V2C_SAR, V2C_JRC, 'v3'
    if name in V3_OTHER:
        return V4_SAR + V2C_SAR, V4_JRC + V2C_JRC, 'v3'   # prefer V4, fall back to V2c
    return V4_SAR, V4_JRC, 'v4'


OUT_CSV = pathlib.Path('analysis/pilot_kge_apcurve.csv')
OUT_PNG = pathlib.Path('analysis/method_comparison_output/ap_kge_curve_pooled.png')


def _resolve(dirs, filename):
    for d in dirs:
        p = d / filename
        if p.exists():
            return p
    return None

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
MIN_PAIRS      = 12
AREA_MIN       = {'Saint_Cassien': 200}
SICILY         = {'Ancipa', 'Poma', 'Pozzillo', 'Rosamarina'}


def kge(obs, sim):
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def _jrc_path(jrc_dirs, name):
    for d in jrc_dirs:
        cands = sorted(d.glob(f'JRC_area_{name}*.csv'))
        plain = [p for p in cands if not _re.search(r'\s*\(\d+\)', p.stem)]
        hit = plain[0] if plain else (cands[0] if cands else None)
        if hit is not None:
            return hit
    return None


def load_sar_monthly(sar_dirs, name):
    p = _resolve(sar_dirs, f'SAR_area_{name}.csv')
    if p is None:
        return None, np.nan
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None, np.nan
    if df.empty:
        return None, np.nan
    ap_m = float(df['ap_m'].iloc[0]) if 'ap_m' in df.columns else np.nan
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if name in AREA_MIN:
        df = df[df['area_ha'] >= AREA_MIN[name]].copy()
    df = df[df['date'] <= '2021-12-31'].copy()
    if df.empty:
        return None, ap_m
    p99 = df['area_ha'].quantile(0.99)
    df = df[df['area_ha'] >= SAR_MIN_FRAC * p99].copy()
    if df.empty:
        return None, ap_m
    df = clean_and_smooth(df.reset_index(drop=True))
    if df.empty:
        return None, ap_m
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index()
    m.columns = ['ym', 'sar_area_ha']
    return m, ap_m


def load_jrc_monthly(jrc_dirs, name):
    p = _jrc_path(jrc_dirs, name)
    if p is None:
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    df = df.sort_values('date').reset_index(drop=True)
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC_MIN].copy()
    if df.empty:
        return None
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if sd > 0:
        df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd].copy()
    df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── compute the 42-reservoir target set ───────────────────────────────────────
rows = []
for name in RESERVOIRS:
    sar_dirs, jrc_dirs, cohort = sources(name)
    jrc = load_jrc_monthly(jrc_dirs, name)
    sar, ap_m = load_sar_monthly(sar_dirs, name)
    if jrc is None or jrc.empty or sar is None or sar.empty:
        continue
    m = pd.merge(sar, jrc, on='ym').dropna()
    if len(m) < MIN_PAIRS:
        continue
    k = kge(m['jrc_area_ha'].values, m['sar_area_ha'].values)
    # which folder actually supplied the SAR series (transparency for the v3 swap)
    src = _resolve(sar_dirs, f'SAR_area_{name}.csv')
    rows.append({'name': name, 'cohort': cohort, 'ap_m': round(ap_m, 1),
                 'n_pairs': len(m), 'kge_dual': round(k, 4),
                 'sicily': name in SICILY,
                 'src': 'V2c' if 'V2c' in str(src) else 'V4'})

df = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f'Saved {len(df)} reservoirs -> {OUT_CSV}')
for c in ('v3', 'v4'):
    s = df[df.cohort == c]
    print(f'  {c}: N={len(s)}  A/P {s.ap_m.min():.0f}–{s.ap_m.max():.0f} m  '
          f'median KGE={s.kge_dual.median():.3f}')

# Spearman A/P vs KGE (thesis: skill rises with A/P)
r, p = stats.spearmanr(df['ap_m'], df['kge_dual'])
print(f'\nSpearman(A/P, KGE_dual) pooled N={len(df)}: r={r:+.3f}, p={p:.2e}')
lo = df[df.ap_m < 100]; hi = df[df.ap_m >= 200]
print(f'  A/P<100  (n={len(lo)}): median KGE={lo.kge_dual.median():.3f}')
print(f'  A/P>=200 (n={len(hi)}): median KGE={hi.kge_dual.median():.3f}   '
      f'(v4 alone had only 3 here; v3 adds {len(hi[hi.cohort=="v3"])})')

# ── figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 7))
STYLE = {'v3': dict(color='#e65100', label='v3 cohort (JRC pilot)'),
         'v4': dict(color='#1565C0', label='v4 cohort (method comparison)')}
for cohort, st in STYLE.items():
    s = df[(df.cohort == cohort) & (~df.sicily)]
    ax.scatter(s['ap_m'], s['kge_dual'], s=70, color=st['color'], alpha=0.85,
               edgecolors='white', linewidths=0.7, zorder=4, label=st['label'])
# Sicilian 4 marked (also PlanetScope-validated) with a ring
sic = df[df.sicily]
ax.scatter(sic['ap_m'], sic['kge_dual'], s=150, facecolors='none',
           edgecolors='#2ca02c', linewidths=1.8, zorder=5,
           label='Sicily 4 (also PlanetScope-validated)')
for _, r_ in df.iterrows():
    ax.annotate(r_['name'].replace('_', ' '), (r_['ap_m'], r_['kge_dual']),
                fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#555')

# binned-median trend (data-driven x) to show the A/P→KGE rise
edges = [df.ap_m.min() - 1, 90, 130, 180, 260, df.ap_m.max() + 1]
df['bin'] = pd.cut(df['ap_m'], edges)
g = df.groupby('bin', observed=True)
ax.plot(g['ap_m'].median().values, g['kge_dual'].median().values, 'k-o',
        lw=2, ms=7, zorder=6, label='binned median')

rho, pv = stats.spearmanr(df['ap_m'], df['kge_dual'])
ax.set_xlabel('A/P — static area/perimeter (m)', fontsize=11)
ax.set_ylabel('KGE — dual-pol VV+VH SVM vs JRC', fontsize=11)
ax.set_title(f'A/P → monitorability, pooled across two independent cohorts (N={len(df)})\n'
             f'single continuous trend (Spearman ρ={rho:+.2f}, p={pv:.1e}) — study area not '
             f'tuned to result', fontsize=12, fontweight='bold')
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(-0.1, 1.02)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT_PNG}')
