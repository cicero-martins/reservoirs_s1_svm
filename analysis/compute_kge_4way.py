"""
compute_kge_4way.py

Four-way method comparison against the SAME JRC reference, on COMMON months
(same dates for every method → isolates the method effect from coverage):

  dual   : VV+VH SVM, FIXED 2023 training mosaic     (the paper's headline method)
  adapt  : VV+VH SVM, PER-SCENE retraining           (SVM_ADAPTIVE — removes the
           arbitrary-baseline objection; parallels Otsu's per-scene adaptivity)
  vv     : VV-only per-scene Otsu                     (Tier 1)
  fast   : VV-only Otsu, NO vectorisation (pixel-count area)  (the cost lever)

Two scientific questions this answers:
  (1) adapt vs dual — does per-scene retraining change/improve the SVM? If
      adapt ≈ dual, the fixed-2023 baseline was harmless (defensible in the paper).
  (2) fast  vs vv   — does dropping vectorisation (the EECU cost driver) degrade
      the area estimate, or is the cheap pixel-count area just as good?

Reads:
  dual  : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4/SAR_area_*.csv
  vv    : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu/SAR_area_*.csv
  JRC   : raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC/JRC_area_*.csv
  adapt : raw_data/GEE_GlobalPilotV4_SVMadapt/SAR_area_*.csv
  fast  : raw_data/GEE_GlobalPilotV4_VVfast/SAR_area_*.csv

Output:
  analysis/pilot_kge_4way.csv
  analysis/method_comparison_output/kge_4way.png
"""

import pathlib
import re as _re
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

# ── clean+smooth (identical to compute_kge_compare.py) ────────────────────────
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


# ── config ────────────────────────────────────────────────────────────────────
# Each method resolves across BOTH possible download layouts: the v4 batch landed
# nested under GEE_GlobalPilotV4b/, while later batches (SVMadapt/VVfast + the v3
# re-export) land at raw_data top level. First existing file wins. This makes the
# script robust to wherever the user extracts each download.
def _P(*parts):
    return [pathlib.Path(p) for p in parts]

METHOD_DIRS = {
    'dual':  _P('raw_data/GEE_GlobalPilotV4', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4'),
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
    'fast':  _P('raw_data/GEE_GlobalPilotV4_VVfast', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVfast'),
}
JRC_DIRS = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
OUT_CSV = pathlib.Path('analysis/pilot_kge_4way.csv')
OUT_PNG = pathlib.Path('analysis/method_comparison_output/kge_4way.png')


def _resolve(dirs, filename):
    """First existing <dir>/<filename> across the fallback list, else None."""
    for d in dirs:
        p = d / filename
        if p.exists():
            return p
    return None

VALID_FRAC_MIN = 0.80
SAR_MIN_FRAC   = 0.02
MIN_COMMON     = 10

EXCLUDE  = {'Oued_Makhazine', 'Guajaraz', 'Miyagase', 'Welbedacht', 'Tzaneen',
            'Egorlyskaia', 'Boegoeberg', 'Itauba', 'Saguaro'}   # +4 flat-JRC chapados
# NOTE (9 Jul 2026): 'Antero' was previously hardcoded here on an old, undocumented
# "extreme geometry" judgment call that predates the systematic reference-noise screen
# below. It does NOT fail that screen and IS part of the trusted N=62 apcurve set, so
# excluding it only here (not from the primary accuracy analysis) was an inconsistency,
# not a real data-quality reason. Removed; Antero is now included if it clears MIN_COMMON.
# Reference-noise screen (screen_reference_noise.py, rough_ratio>=2.5): the JRC series
# itself jumps month-to-month while independent SAR methods stay smooth, at high
# valid_frac (so it is optical misclassification, not coverage) -- any KGE computed
# against these is comparing methods to a bad reference, not testing the methods.
try:
    _rn = pd.read_csv('analysis/reference_noise.csv')
    EXCLUDE |= set(_rn.loc[_rn.ref_noise, 'name'])
except FileNotFoundError:
    pass  # first run before reference_noise.csv exists: falls back to the hardcoded set
AREA_MIN = {'Saint_Cassien': 200}

cand  = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
NAMES = cand['name'].tolist()


def kge(obs, sim):
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r, _  = stats.pearsonr(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def _jrc_path(name):
    for d in JRC_DIRS:
        cands = sorted(d.glob(f'JRC_area_{name}*.csv'))
        plain = [p for p in cands if not _re.search(r'\s*\(\d+\)', p.stem)]
        hit = plain[0] if plain else (cands[0] if cands else None)
        if hit is not None:
            return hit
    return None


def load_sar_monthly(name, sar_dirs):
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
    if df.empty:
        return None, ap_m
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
    m.columns = ['ym', 'area']
    return m, ap_m


from jrc_filter import load_jrc_monthly as _load_jrc_shared
def load_jrc_monthly(name):
    df = _load_jrc_shared(name, JRC_DIRS, despike=True)   # vf-gated de-spike
    if df is None or df.empty:
        return None
    df = df.copy(); df['ym'] = df['date'].dt.to_period('M')
    return df[['ym', 'jrc_area_ha']].dropna()


# ── main loop: KGE per method on months COMMON to all available methods ───────
rows = []
for name in NAMES:
    if name in EXCLUDE:
        continue
    jrc_m = load_jrc_monthly(name)
    if jrc_m is None or jrc_m.empty:
        continue

    series, ap_m = {}, np.nan
    for meth, d in METHOD_DIRS.items():
        m, ap = load_sar_monthly(name, d)
        if meth == 'dual' and not np.isnan(ap):
            ap_m = ap
        if m is not None and not m.empty:
            series[meth] = m.rename(columns={'area': meth})

    # need dual at minimum
    if 'dual' not in series:
        continue

    # inner-join every available method + JRC on ym → strictly common months
    merged = jrc_m.copy()
    for meth, m in series.items():
        merged = merged.merge(m, on='ym', how='inner')
    merged = merged.dropna()
    n_common = len(merged)
    if n_common < MIN_COMMON:
        continue

    obs = merged['jrc_area_ha'].values
    row = {'name': name,
           'ap_m': round(ap_m, 1) if not np.isnan(ap_m) else np.nan,
           'n_common': n_common,
           'methods': '+'.join(series.keys())}
    for meth in METHOD_DIRS:
        row[f'kge_{meth}'] = round(kge(obs, merged[meth].values), 4) if meth in series else np.nan
    # headline deltas
    if 'adapt' in series:
        row['d_adapt_dual'] = round(row['kge_adapt'] - row['kge_dual'], 4)
    if 'vv' in series:
        row['d_dual_vv'] = round(row['kge_dual'] - row['kge_vv'], 4)
    if 'fast' in series and 'vv' in series:
        row['d_fast_vv'] = round(row['kge_fast'] - row['kge_vv'], 4)
    rows.append(row)

df = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f'Saved {len(df)} reservoirs -> {OUT_CSV}\n')
ids = pd.read_csv('analysis/reservoir_ids.csv')[['name', 'id']].set_index('name')
df['id'] = df['name'].map(ids['id']).fillna(df['name'].str.replace('_', ' '))


def summ(col, label):
    v = df[col].dropna()
    if v.empty:
        print(f'{label}: no data'); return
    print(f'{label}: N={len(v)}  mean={v.mean():+.3f}  median={v.median():+.3f}  '
          f'win>0.02={int((v>0.02).sum())}  tie={int((v.abs()<=0.02).sum())}  '
          f'lose<-0.02={int((v<-0.02).sum())}')

print('=== KGE medians per method (common-month rows) ===')
for meth, lab in [('dual', 'dual (SVM fixed 2023)'), ('adapt', 'adapt (SVM per-scene)'),
                  ('vv', 'vv (Otsu)'), ('fast', 'fast (Otsu no-vector)')]:
    v = df[f'kge_{meth}'].dropna()
    if not v.empty:
        print(f'  {lab:<26} N={len(v):>2}  median KGE={v.median():+.3f}  mean={v.mean():+.3f}')

print('\n=== headline deltas ===')
summ('d_adapt_dual', 'Q1  adapt − dual (per-scene retraining effect)')
summ('d_dual_vv',    '    dual  − vv   (dual-pol classifier gain)')
summ('d_fast_vv',    'Q2  fast  − vv   (dropping vectorisation)')

# Wilcoxon on the paired adapt-vs-dual (does per-scene retraining move KGE at all?)
pair = df.dropna(subset=['kge_adapt', 'kge_dual'])
if len(pair) >= 6:
    try:
        w, p = stats.wilcoxon(pair['kge_adapt'], pair['kge_dual'])
        print(f'\nWilcoxon adapt vs dual: W={w:.1f}, p={p:.3f}  '
              f'(N={len(pair)}) → {"no sig. difference" if p>0.05 else "SIGNIFICANT"}')
    except ValueError as e:
        print(f'\nWilcoxon adapt vs dual: {e}')

# ── figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)
lims = [-0.6, 1.05]

def _trend(ax, x, y):
    """OLS trend line across the panel (distinct from the dashed 1:1 reference)
    so the overall bias direction/magnitude is readable at a glance, not just
    the win/loss scatter."""
    slope, icpt, r, p, se = stats.linregress(x, y)
    xf = np.array(lims)
    ax.plot(xf, slope * xf + icpt, color='#d62728', lw=1.8, alpha=0.85, zorder=3.5,
            label=f'trend (slope={slope:.2f})')

def _label(ax, d, xcol, ycol):
    """Annotate only points within the plotted lims (others are clipped anyway);
    keeps adjust_text from dragging off-screen outliers' labels back into view."""
    v = d[d[xcol].between(*lims) & d[ycol].between(*lims)]
    texts = [ax.text(r[xcol], r[ycol], r['id'], fontsize=7.5,
                      color='#444', zorder=10) for _, r in v.iterrows()]
    adjust_text(texts, x=v[xcol].values, y=v[ycol].values, ax=ax,
                expand=(1.6, 1.8), force_text=(0.5, 0.6), force_static=(0.7, 0.8),
                arrowprops=dict(arrowstyle='-', color='#999', lw=0.5))

AP_VMIN, AP_VMAX = df['ap_m'].min(), df['ap_m'].max()   # shared colour scale, all panels

# (a) adapt vs dual, 1:1 — Q1
ax = axes[0, 0]
p1 = df.dropna(subset=['kge_adapt', 'kge_dual'])
sc = ax.scatter(p1['kge_dual'], p1['kge_adapt'], c=p1['ap_m'], cmap='viridis',
                vmin=AP_VMIN, vmax=AP_VMAX,
                s=55, edgecolors='white', linewidths=0.5, zorder=4)
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1 (adapt = dual)')
_trend(ax, p1['kge_dual'], p1['kge_adapt'])
ax.set_xlim(*lims); ax.set_ylim(*lims)
_label(ax, p1, 'kge_dual', 'kge_adapt')
ax.set_xlabel('KGE — SVM fixed 2023 training (dual)')
ax.set_ylabel('KGE — SVM per-scene retraining (adapt)')
ax.set_title('(a) Q1: per-scene retraining vs fixed baseline\n'
             'mostly above 1:1 → adaptive $\\geq$ fixed (median $\\Delta$KGE +0.07)',
             fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.set_box_aspect(1)

# (b) fast vs vv, 1:1 — Q2
ax = axes[0, 1]
p2 = df.dropna(subset=['kge_fast', 'kge_vv'])
ax.scatter(p2['kge_vv'], p2['kge_fast'], c=p2['ap_m'], cmap='viridis',
           vmin=AP_VMIN, vmax=AP_VMAX,
           s=55, edgecolors='white', linewidths=0.5, zorder=4)
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1 (Fast = vv)')
_trend(ax, p2['kge_vv'], p2['kge_fast'])
ax.set_xlim(*lims); ax.set_ylim(*lims)
_label(ax, p2, 'kge_vv', 'kge_fast')
ax.set_xlabel('KGE — VV Otsu, vectorised area (vv)')
ax.set_ylabel('KGE — Fast (VV Otsu, pixel-count area)')
ax.set_title('(b) Q2: dropping vectorisation (the cost lever)\n'
             'mostly below 1:1 → Fast costs a modest hit (median $\\Delta$KGE $-$0.07)',
             fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.set_box_aspect(1)

# (c) dual (fixed) vs vv, 1:1 — is polarisation count alone worth it?
ax = axes[1, 0]
p3 = df.dropna(subset=['kge_dual', 'kge_vv'])
ax.scatter(p3['kge_vv'], p3['kge_dual'], c=p3['ap_m'], cmap='viridis',
           vmin=AP_VMIN, vmax=AP_VMAX,
           s=55, edgecolors='white', linewidths=0.5, zorder=4)
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1 (dual = vv)')
_trend(ax, p3['kge_vv'], p3['kge_dual'])
ax.set_xlim(*lims); ax.set_ylim(*lims)
_label(ax, p3, 'kge_vv', 'kge_dual')
ax.set_xlabel('KGE — VV Otsu, per-scene (vv)')
ax.set_ylabel('KGE — SVM fixed 2023 training (dual)')
ax.set_title('(c) VV Otsu vs fixed-training dual SVM\n'
             'mostly below 1:1 → fixed dual ranks below the simple threshold',
             fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.set_box_aspect(1)

# (d) adapt vs vv, 1:1 — the paper's actual head-to-head. Unlike (a)-(c), this
# does NOT need the four-way df (which requires a 'dual' export to exist at
# all, capping it near N=47 even though this panel never plots dual): it
# reads bestof_kge.csv directly with the EXACT same filter compute_kge_apcurve.py
# uses (drop chapados, NO_DUAL_EXPORT, reference-noise-flagged), so this panel
# matches the N=62/59 and 38/24 winner split quoted in the text and
# Figure~\ref{fig:ap_curve}, not the smaller four-way subset.
_NO_DUAL_EXPORT = {'Forggen'}
try:
    _rn2 = pd.read_csv('analysis/reference_noise.csv')
    _REF_NOISE = set(_rn2.loc[_rn2.ref_noise, 'name'])
except FileNotFoundError:
    _REF_NOISE = set()
p4_full = pd.read_csv('analysis/bestof_kge.csv').dropna(subset=['best'])
p4_full = p4_full[~p4_full['chapado']]
p4_full = p4_full[~p4_full['name'].isin(_NO_DUAL_EXPORT)]
p4_full = p4_full[~p4_full['name'].isin(_REF_NOISE)].copy()
p4_full['id'] = p4_full['name'].map(ids['id']).fillna(p4_full['name'].str.replace('_', ' '))
ax = axes[1, 1]
p4 = p4_full.dropna(subset=['kge_adapt', 'kge_vv'])
ax.scatter(p4['kge_vv'], p4['kge_adapt'], c=p4['ap_m'], cmap='viridis',
           vmin=AP_VMIN, vmax=AP_VMAX,
           s=55, edgecolors='white', linewidths=0.5, zorder=4)
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1 (adapt = vv)')
_trend(ax, p4['kge_vv'], p4['kge_adapt'])
ax.set_xlim(*lims); ax.set_ylim(*lims)
_label(ax, p4, 'kge_vv', 'kge_adapt')
ax.set_xlabel('KGE — VV Otsu, per-scene (vv)')
ax.set_ylabel('KGE — SVM per-scene retraining (adapt)')
ax.set_title(f'(d) VV Otsu vs per-scene dual SVM (N={len(p4)})\n'
             'scattered around 1:1 → the two per-scene detectors are indistinguishable',
             fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
ax.set_box_aspect(1)

# One shared colorbar for all four panels (ax=list of all axes keeps their
# plotted areas equal-sized -- a colorbar attached to only one axis would
# shrink just that panel and leave the subplots visibly different sizes).
fig.colorbar(sc, ax=axes.ravel().tolist(), location='right', shrink=0.6, pad=0.02,
             label='A/P (m)')

fig.suptitle('Four-way method comparison against JRC (common months)',
             fontsize=13, fontweight='bold')
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT_PNG}')
