"""
plot_kge_components_v4.py — KGE decomposition per reservoir (combined v3 + v4).

Four stacked panels sharing the x-axis (reservoirs sorted by A/P ascending):
  (a) KGE          — reference at 0.5
  (b) r            — Pearson correlation, reference at 1.0
  (c) alpha        — variability ratio (sim_std / obs_std), reference at 1.0
  (d) beta         — bias ratio (sim_mean / obs_mean), reference at 1.0

Color = fail mode (same as AP_classification figure).
Hatch = v3; plain = v4.
Dashed vertical line at the Youden-J optimal A/P threshold.

Output: analysis/method_comparison_output/kge_components_v4.png
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/method_comparison_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

KGE_THRESH = 0.5

# ── Load and merge ─────────────────────────────────────────────────────────────
v3 = pd.read_csv('analysis/pilot_kge_v3.csv')
v3['source'] = 'v3'

v4 = pd.read_csv('analysis/pilot_kge_v4.csv')
v4['source'] = 'v4'

keep = ['name', 'ap_m', 'kge', 'r', 'alpha', 'beta', 'source']
df   = pd.concat([v3[keep], v4[keep]], ignore_index=True)
df   = df.dropna(subset=['ap_m', 'kge', 'r', 'alpha', 'beta']).reset_index(drop=True)
df['adequate'] = df['kge'] >= KGE_THRESH
df   = df.sort_values('ap_m').reset_index(drop=True)

# ── ROC → optimal A/P threshold ───────────────────────────────────────────────
auc = roc_auc_score(df['adequate'].astype(int), df['ap_m'])
fpr, tpr, thresholds = roc_curve(df['adequate'].astype(int), df['ap_m'])
j_idx  = np.argmax(tpr - fpr)
AP_OPT = float(thresholds[j_idx])

split_idx = df.index[df['ap_m'] >= AP_OPT].tolist()
split_x   = (split_idx[0] - 0.5) if split_idx else len(df) - 0.5

# ── Fail-mode colours ──────────────────────────────────────────────────────────
def fail_mode(row):
    if row['adequate']:
        return 'adequate'
    if row['alpha'] > 1.5:
        return 'noisy'
    if row['alpha'] < 0.5:
        return 'flat'
    if row['beta'] < 0.6:
        return 'bias'
    return 'low_r'

COLOR_MAP = {
    'adequate': '#2ca02c',
    'noisy':    '#d62728',
    'flat':     '#ff7f0e',
    'bias':     '#9467bd',
    'low_r':    '#8c564b',
}

df['fail']      = df.apply(fail_mode, axis=1)
df['bar_color'] = df['fail'].map(COLOR_MAP)
df['label']     = df['name'].apply(lambda n: n.replace('_', ' ')
                                   .removesuffix(' ES').removesuffix(' US')
                                   .removesuffix(' AU').removesuffix(' MX')
                                   .removesuffix(' IN'))

n_v3 = (df['source'] == 'v3').sum()
n_v4 = (df['source'] == 'v4').sum()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(20, 13),
                         gridspec_kw={'hspace': 0.55}, sharex=False)

x = np.arange(len(df))

PANEL_CFG = [
    ('kge',   'KGE',                 KGE_THRESH,  (-1.0, 1.1),  True),
    ('r',     'Pearson r',           1.0,          (0.0,  1.1),  False),
    ('alpha', 'alpha  (σ_sim/σ_obs)',1.0,          (0.0,  3.0),  False),
    ('beta',  'beta  (μ_sim/μ_obs)', 1.0,          (0.3,  1.7),  False),
]

for ax, (col, ylabel, ref_val, ylim, show_xlab) in zip(axes, PANEL_CFG):
    bars = ax.bar(x, df[col], color=df['bar_color'],
                  edgecolor='white', linewidth=0.4, zorder=3)

    for bar, src in zip(bars, df['source']):
        if src == 'v3':
            bar.set_hatch('//')
            bar.set_edgecolor('#333')
            bar.set_linewidth(0.6)

    ax.axhline(ref_val, color='gray', lw=1.1, ls=':', alpha=0.8, zorder=2)
    ax.axvline(split_x, color='navy', lw=1.5, ls='--', alpha=0.7, zorder=4)

    if col == 'alpha':
        ax.axhline(1.5, color='#d62728', lw=0.8, ls=':', alpha=0.5)
        ax.axhline(0.5, color='#ff7f0e', lw=0.8, ls=':', alpha=0.5)
    if col == 'beta':
        ax.axhline(0.6, color='#9467bd', lw=0.8, ls=':', alpha=0.5)
        ax.axhline(1.4, color='#9467bd', lw=0.8, ls=':', alpha=0.5)

    # A/P labels on a top twin-x
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    step = max(1, len(df) // 14)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([f'{v:.0f}' for v in df['ap_m'].iloc[::step]],
                        fontsize=5, rotation=45, ha='left')
    ax2.set_xlabel('A/P (m)', fontsize=7)

    ax.set_xlim(-0.7, len(df) - 0.3)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5 if col == 'kge' else 0.5))

    if show_xlab or col == 'beta':
        ax.set_xticks(x)
        ax.set_xticklabels(df['label'], rotation=55, ha='right', fontsize=5)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([], fontsize=0)

    # Region annotation on KGE panel only
    if col == 'kge':
        n_low  = (df['ap_m'] < AP_OPT).sum()
        n_high = (df['ap_m'] >= AP_OPT).sum()
        ax.text(split_x / 2, ylim[1] - 0.05,
                f'A/P < {AP_OPT:.0f} m  (n={n_low})',
                ha='center', va='top', fontsize=7.5, color='#555',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax.text(split_x + n_high / 2, ylim[1] - 0.05,
                f'A/P ≥ {AP_OPT:.0f} m  (n={n_high})',
                ha='center', va='top', fontsize=7.5, color='#333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(fc='#2ca02c', label=f'KGE ≥ {KGE_THRESH} (adequate)'),
    mpatches.Patch(fc='#d62728', label='Fail: alpha > 1.5 (noisy SAR)'),
    mpatches.Patch(fc='#ff7f0e', label='Fail: alpha < 0.5 (flat SAR)'),
    mpatches.Patch(fc='#9467bd', label='Fail: beta < 0.6 (SAR under-estimates)'),
    mpatches.Patch(fc='#8c564b', label='Fail: low r (poor timing)'),
    mpatches.Patch(fc='#aaa', hatch='//', label='pilot v3 (hatched)'),
]
fig.legend(handles=legend_patches, loc='lower right',
           bbox_to_anchor=(0.99, 0.01), fontsize=7.5, framealpha=0.9, ncol=3)

fig.suptitle(
    f'KGE components per reservoir  (v3 N={n_v3} + v4 N={n_v4} = {len(df)}, '
    f'sorted by A/P asc, dashed = Youden-J threshold {AP_OPT:.0f} m)',
    fontsize=10, fontweight='bold')

out = OUT_DIR / 'kge_components_v4.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
print(f'N = {len(df)}  ({n_v3} v3 + {n_v4} v4)  |  A/P threshold = {AP_OPT:.0f} m')
