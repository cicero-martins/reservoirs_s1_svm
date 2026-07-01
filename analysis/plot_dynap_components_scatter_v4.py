"""
plot_dynap_components_scatter_v4.py

Three scatter panels (dynamic A/P vs r, alpha, beta) — same style as
ap_components_scatter_v4.py but x-axis = mean dynamic A/P (ap_m_dynamic).
v4 only (N=29; v3 has no dynamic A/P).

Output: analysis/method_comparison_output/dynap_components_scatter_v4.png
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/method_comparison_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

KGE_THRESH = 0.5

# ── Load v4 only (has ap_m_dynamic) ──────────────────────────────────────────
df = pd.read_csv('analysis/pilot_kge_v4.csv')
df = df.dropna(subset=['ap_m_dynamic', 'kge', 'r', 'alpha', 'beta']).reset_index(drop=True)
df['adequate'] = df['kge'] >= KGE_THRESH
df['source']   = 'v4'

# ── Youden-J threshold on dynamic A/P ─────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m_dynamic'])
j_idx   = np.argmax(tpr - fpr)
AP_OPT  = float(thresholds[j_idx])

# ── Fail-mode colours ─────────────────────────────────────────────────────────
def fail_mode(row):
    if row['adequate']:      return 'adequate'
    if row['alpha'] > 1.5:  return 'noisy'
    if row['alpha'] < 0.5:  return 'flat'
    if row['beta']  < 0.6:  return 'bias'
    return 'low_r'

COLOR_MAP = {
    'adequate': '#2ca02c',
    'noisy':    '#d62728',
    'flat':     '#ff7f0e',
    'bias':     '#9467bd',
    'low_r':    '#8c564b',
}

df['fail']  = df.apply(fail_mode, axis=1)
df['color'] = df['fail'].map(COLOR_MAP)
df['label'] = df['name'].apply(lambda n: n.replace('_', ' ')
                                .removesuffix(' ES').removesuffix(' US')
                                .removesuffix(' AU').removesuffix(' MX')
                                .removesuffix(' IN'))

# ── Panel definitions ─────────────────────────────────────────────────────────
PANELS = [
    ('r',     'Pearson r',              1.0, (0.0,  1.12)),
    ('alpha', 'alpha  (σ_SAR / σ_JRC)', 1.0, (0.0,  2.6)),
    ('beta',  'beta  (μ_SAR / μ_JRC)',  1.0, (0.3,  1.7)),
]

def should_label(row, col):
    if col == 'r'     and (row['r']     < 0.3  or row['r']     > 0.95): return True
    if col == 'alpha' and (row['alpha'] < 0.3  or row['alpha'] > 1.8):  return True
    if col == 'beta'  and (row['beta']  < 0.55 or row['beta']  > 1.35): return True
    return False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                         gridspec_kw={'wspace': 0.35})

for ax, (col, ylabel, ref_val, ylim) in zip(axes, PANELS):
    xv = df['ap_m_dynamic'].values
    yv = df[col].values

    # Scatter (circles only — v4)
    for _, row in df.iterrows():
        ax.scatter(row['ap_m_dynamic'], row[col],
                   s=50, marker='o', color=row['color'],
                   edgecolors='white', linewidths=0.5, alpha=0.88, zorder=4)

    # Linear trend
    z      = np.polyfit(xv, yv, 1)
    xf     = np.linspace(xv.min() - 5, xv.max() + 5, 300)
    rp, pp = pearsonr(xv, yv)
    p_str  = f'p={pp:.3f}' if pp >= 0.001 else 'p<0.001'
    ax.plot(xf, np.polyval(z, xf), 'k-', lw=1.4, alpha=0.55, zorder=3,
            label=f'r={rp:.2f}, {p_str}')

    # Reference lines
    ax.axhline(ref_val, color='gray', lw=1.0, ls=':', alpha=0.7, zorder=2,
               label=f'ideal ({ref_val:.0f})')
    ax.axvline(AP_OPT, color='navy', lw=1.0, ls='--', alpha=0.6, zorder=2,
               label=f'dyn A/P = {AP_OPT:.0f} m')

    if col == 'alpha':
        ax.axhline(1.5, color='#d62728', lw=0.7, ls=':', alpha=0.45, label='α=1.5')
        ax.axhline(0.5, color='#ff7f0e', lw=0.7, ls=':', alpha=0.45, label='α=0.5')
    if col == 'beta':
        ax.axhline(0.6, color='#9467bd', lw=0.7, ls=':', alpha=0.45, label='β=0.6')

    # Annotate outliers
    for _, row in df.iterrows():
        if should_label(row, col):
            ax.annotate(row['label'], (row['ap_m_dynamic'], row[col]),
                        fontsize=5.5, xytext=(4, 3),
                        textcoords='offset points', color='#444')

    ax.set_xlabel('Dynamic A/P mean (m)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(xv.min() - 10, xv.max() + 10)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc='upper left')

fig.suptitle(
    f'Dynamic A/P (mean) vs KGE components  (pilot v4, N={len(df)}, '
    f'dyn A/P Youden-J threshold = {AP_OPT:.0f} m)',
    fontsize=10, fontweight='bold')

legend_elems = [
    *[Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
             markersize=7, label=lbl)
      for lbl, c in [
          ('adequate (KGE≥0.5)', '#2ca02c'),
          ('noisy (α>1.5)',       '#d62728'),
          ('flat (α<0.5)',        '#ff7f0e'),
          ('bias (β<0.6)',        '#9467bd'),
          ('low r',               '#8c564b'),
      ]],
]
fig.legend(handles=legend_elems, loc='lower center',
           bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7.5, framealpha=0.9)

out = OUT_DIR / 'dynap_components_scatter_v4.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}  (N={len(df)})')

for col, ylabel, *_ in PANELS:
    rp, pp = pearsonr(df['ap_m_dynamic'].values, df[col].values)
    p_str  = f'{pp:.4f}' if pp >= 0.0001 else '<0.0001'
    print(f'  dyn A/P vs {col:<6}  Pearson r = {rp:+.3f}  p = {p_str}')

# ── Compare static vs dynamic A/P correlations ───────────────────────────────
print(f'\n  Comparison (v4 only, N={len(df)}):')
print(f'  {"":6}  {"static A/P":>15}  {"dynamic A/P":>15}')
for col, *_ in PANELS:
    rs, ps = pearsonr(df['ap_m'].values, df[col].values)
    rd, pd_ = pearsonr(df['ap_m_dynamic'].values, df[col].values)
    ps_str  = f'r={rs:+.3f} p={ps:.3f}'
    pd_str  = f'r={rd:+.3f} p={pd_:.3f}'
    print(f'  {col:<6}  {ps_str:>15}  {pd_str:>15}')
