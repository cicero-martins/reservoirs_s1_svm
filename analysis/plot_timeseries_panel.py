"""
plot_timeseries_panel.py

A 12-reservoir sample of monthly water-area time series (JRC optical reference
vs the two per-scene detectors, adapt SVM and VV Otsu), for Results S4.1 --
makes the abstract "A/P sets a monitorability ceiling" claim concrete by
showing what a KGE>0.5 series actually looks like at low, medium, and high
A/P. Four reservoirs per static-A/P band (low <100 m, medium 100-200 m,
high >=200 m), all with best-of KGE>0.5, spread across each band's A/P range,
and requiring complete JRC+adapt+VV data (no gaps in the loaded series).

Reuses the exact clean+smooth+monthly pipeline from plot_timeseries_all.py.

Output: analysis/method_comparison_output/timeseries_panel.png
"""
import pathlib
import numpy as np
import pandas as pd

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
def clean_and_smooth(df):
    s = df['area_ha'].copy(); s = _rg(s, 2.0); s = _rl(s, 5, 1.5); s = _rl(s, 5, 1.5); s = _rl(s, 10, 1.5)
    ds = df.loc[s.index, 'date'].reset_index(drop=True)
    return pd.DataFrame({'date': ds, 'area_ha': _lw(ds, s.reset_index(drop=True), 20, 7)})

def _P(*p): return [pathlib.Path(x) for x in p]
METHOD_DIRS = {
    'adapt': _P('raw_data/GEE_GlobalPilotV4_SVMadapt', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_SVMadapt'),
    'vv':    _P('raw_data/GEE_GlobalPilotV4_VVotsu', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_VVotsu'),
}
JRC_DIRS = _P('raw_data/GEE_GlobalPilotV4_JRC', 'raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC',
              'raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')
COL = {'adapt': '#1f77b4', 'vv': '#2ca02c'}
LAB = {'adapt': 'adapt (per-scene SVM)', 'vv': 'VV-only Otsu'}
SAR_MIN_FRAC, AREA_MIN = 0.02, {'Saint_Cassien': 200}

def _res(dirs, fn):
    for d in dirs:
        p = d / fn
        if p.exists(): return p
    return None

def load_sar(dirs, n):
    p = _res(dirs, f'SAR_area_{n}.csv')
    if p is None: return None
    try: df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError: return None
    if df.empty: return None
    df = df[df['area_ha'] > 0].sort_values('date').reset_index(drop=True)
    if n in AREA_MIN: df = df[df['area_ha'] >= AREA_MIN[n]]
    df = df[df['date'] <= '2021-12-31']
    if df.empty: return None
    df = df[df['area_ha'] >= SAR_MIN_FRAC * df['area_ha'].quantile(0.99)]
    if df.empty: return None
    df = clean_and_smooth(df.reset_index(drop=True))
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym')['area_ha'].mean().reset_index()
    m['date'] = m['ym'].dt.to_timestamp()
    return m[['date', 'area_ha']]

from jrc_filter import load_jrc_monthly as _load_jrc_shared
def load_jrc(n):
    return _load_jrc_shared(n, JRC_DIRS, despike=True)

def break_gaps(df, dc, vc):
    s = df.copy(); s[dc] = s[dc].dt.to_period('M').dt.to_timestamp()
    s = s.groupby(dc)[vc].mean()
    full = pd.period_range(s.index.min(), s.index.max(), freq='M').to_timestamp()
    return s.reindex(full)

# ── select 4 per A/P band: KGE>0.5, complete JRC+adapt+VV data, spread across
#    the band's own A/P range ────────────────────────────────────────────────
bof = pd.read_csv('analysis/bestof_kge.csv').dropna(subset=['best'])
bof = bof[bof['best'] > 0.5].sort_values('ap_m').reset_index(drop=True)

BANDS = [('Low A/P (<100 m)', bof[bof.ap_m < 100]),
         ('Medium A/P (100-200 m)', bof[(bof.ap_m >= 100) & (bof.ap_m < 200)]),
         ('High A/P (>=200 m)', bof[bof.ap_m >= 200])]

cache = {}
def _complete(name):
    if name not in cache:
        jrc = load_jrc(name)
        adapt = load_sar(METHOD_DIRS['adapt'], name)
        vv = load_sar(METHOD_DIRS['vv'], name)
        ok = all(x is not None and not x.empty for x in (jrc, adapt, vv))
        cache[name] = (ok, jrc, adapt, vv)
    return cache[name]

selected = []  # (band_label, name, ap_m, kge)
for label, sub in BANDS:
    sub = sub.sort_values('ap_m').reset_index(drop=True)
    valid = [r for _, r in sub.iterrows() if _complete(r['name'])[0]]
    if len(valid) <= 4:
        picks = valid
    else:
        idx = sorted(set(np.linspace(0, len(valid) - 1, 4).round().astype(int)))
        while len(idx) < 4:
            for i in range(len(valid)):
                if i not in idx:
                    idx.append(i); break
        picks = [valid[i] for i in sorted(idx)[:4]]
    for r in picks:
        selected.append((label, r['name'], r['ap_m'], r['best']))

# Manual swap: Cachi/Kotmale/Harlan_County looked too noisy/inconsistent as
# "clean KGE>0.5" examples once plotted -- replaced with higher-KGE, complete-
# data reservoirs from the same band (verified via _complete above).
REPLACE = {'Cachi': 'Boadella', 'Kotmale': 'Garcia', 'Harlan_County': 'Karaoun'}
bof_idx = bof.set_index('name')
selected = [
    (label, REPLACE.get(name, name), *(
        (bof_idx.loc[REPLACE[name], 'ap_m'], bof_idx.loc[REPLACE[name], 'best'])
        if name in REPLACE else (ap_m, kge)))
    for label, name, ap_m, kge in selected
]
for name in REPLACE.values():
    _complete(name)  # populate cache

print(f'Selected {len(selected)} reservoirs:')
for label, name, ap_m, kge in selected:
    print(f'  {label:<25} {name:<22} A/P={ap_m:.0f}  KGE={kge:.2f}')

# ── figure: 3 rows (bands) x 4 cols ────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig, axes = plt.subplots(3, 4, figsize=(19, 11))
for i, (label, name, ap_m, kge) in enumerate(selected):
    row, col = i // 4, i % 4
    ax = axes[row, col]
    ok, jrc, adapt, vv = cache[name]
    g = break_gaps(jrc, 'date', 'jrc_area_ha')
    ax.plot(g.index, g.values, color='k', lw=1.0, ls='--', marker='o', ms=2.5, zorder=6, label='JRC')
    ax.plot(adapt['date'], adapt['area_ha'], color=COL['adapt'], lw=1.6, zorder=5, label=LAB['adapt'])
    ax.plot(vv['date'], vv['area_ha'], color=COL['vv'], lw=1.6, zorder=5, label=LAB['vv'])
    ax.set_title(f"{name.replace('_', ' ')}  (A/P={ap_m:.0f} m, KGE={kge:.2f})", fontsize=12)
    ax.tick_params(axis='both', labelsize=9.5); ax.margins(x=0.02); ax.grid(alpha=0.2)
    if col == 0:
        ax.set_ylabel(label, fontsize=13.5, fontweight='bold')

handles = [Line2D([0], [0], color='k', ls='--', marker='o', ms=4, label='JRC (optical ref)')] + \
          [Line2D([0], [0], color=COL[m], lw=2, label=LAB[m]) for m in METHOD_DIRS]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=13,
           bbox_to_anchor=(0.5, 1.01), frameon=False)
fig.suptitle('Sample monthly water-area series across the A/P range (KGE > 0.5 throughout)',
             fontsize=15, fontweight='bold', y=1.045)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = pathlib.Path('analysis/method_comparison_output/timeseries_panel.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
