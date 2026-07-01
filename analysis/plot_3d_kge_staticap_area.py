"""
plot_3d_kge_staticap_area.py

3D scatter (static A/P, reservoir area, KGE) with fitted regression plane
KGE ~ b0 + b1*ap_m + b2*log10(area).  Combined v3 + v4 (N=43).

Outputs:
  analysis/method_comparison_output/3d_kge_staticap_area.html   (plotly, interactive)
  analysis/method_comparison_output/3d_kge_staticap_area.png    (matplotlib, 4 viewpoints)
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR    = Path('analysis/method_comparison_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)
KGE_THRESH = 0.5

# ── Data (combined v3 + v4) ───────────────────────────────────────────────────
v3 = pd.read_csv('analysis/pilot_kge_v3.csv'); v3['source'] = 'v3'
v4 = pd.read_csv('analysis/pilot_kge_v4.csv'); v4['source'] = 'v4'

keep = ['name', 'ap_m', 'kge', 'r', 'alpha', 'beta', 'mean_jrc_ha', 'source']
df   = pd.concat([v3[keep], v4[keep]], ignore_index=True)
df   = df.dropna(subset=['ap_m', 'kge', 'mean_jrc_ha']).reset_index(drop=True)
df['adequate'] = df['kge'] >= KGE_THRESH
df['log_area'] = np.log10(df['mean_jrc_ha'])
df['label']    = df['name'].str.replace('_', ' ')

def fail_mode(row):
    if row['adequate']:     return 'adequate'
    if row['alpha'] > 1.5: return 'noisy'
    if row['alpha'] < 0.5: return 'flat'
    if row['beta']  < 0.6: return 'bias'
    return 'low_r'

COLOR_MAP = {
    'adequate': '#2ca02c', 'noisy': '#d62728',
    'flat':     '#ff7f0e', 'bias':  '#9467bd', 'low_r': '#8c564b',
}
COLOR_PLOTLY = {
    'adequate': 'green',    'noisy': 'crimson',
    'flat':     'darkorange','bias': 'mediumpurple', 'low_r': 'saddlebrown',
}

df['fail']   = df.apply(fail_mode, axis=1)
df['color']  = df['fail'].map(COLOR_MAP)
df['cplotly']= df['fail'].map(COLOR_PLOTLY)

n_v3 = (df['source'] == 'v3').sum()
n_v4 = (df['source'] == 'v4').sum()

# ── Fit regression plane: KGE ~ ap_m + log10(area) ───────────────────────────
X = df[['ap_m', 'log_area']].values
y = df['kge'].values

model  = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2     = r2_score(y, y_pred)
b0, b1, b2 = model.intercept_, model.coef_[0], model.coef_[1]

print(f'Regression plane: KGE = {b0:.3f} + {b1:.5f}*ap_m + {b2:.3f}*log10(area)')
print(f'R² = {r2:.3f}  (N={len(df)},  {n_v3} v3 + {n_v4} v4)')
print(f'  Pearson r(ap_m,     KGE) = {pearsonr(df["ap_m"],     y)[0]:+.3f}  '
      f'p={pearsonr(df["ap_m"],     y)[1]:.4f}')
print(f'  Pearson r(log_area, KGE) = {pearsonr(df["log_area"], y)[0]:+.3f}  '
      f'p={pearsonr(df["log_area"], y)[1]:.4f}')

# ── Mesh for plane ────────────────────────────────────────────────────────────
ap_min,  ap_max  = df['ap_m'].min()-10,   df['ap_m'].max()+10
la_min,  la_max  = df['log_area'].min()-0.1, df['log_area'].max()+0.1

ap_grid, la_grid = np.meshgrid(np.linspace(ap_min, ap_max, 40),
                                np.linspace(la_min, la_max, 40))
kge_grid = np.clip(b0 + b1*ap_grid + b2*la_grid, -0.6, 1.15)

# Adequacy boundary on regression plane (where plane crosses KGE=0.5)
la_line = np.linspace(la_min, la_max, 300)
ap_line = (KGE_THRESH - b0 - b2*la_line) / b1
valid   = (ap_line >= ap_min) & (ap_line <= ap_max)

# ── Log-area tick labels ───────────────────────────────────────────────────────
LA_TICKS = [2.0, 2.3, 2.6, 3.0, 3.3]
LA_LBLS  = ['100', '200', '400', '1000', '2000']


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLOTLY interactive
# ══════════════════════════════════════════════════════════════════════════════
fig_p = go.Figure()

# Regression plane
fig_p.add_trace(go.Surface(
    x=ap_grid, y=la_grid, z=kge_grid,
    colorscale=[[0,'#FFCDD2'],[0.5,'#FFF9C4'],[1,'#C8E6C9']],
    opacity=0.40, showscale=False, name='Regression plane', hoverinfo='skip',
))

# KGE=0.5 horizontal reference plane
ap_ref  = np.array([[ap_min, ap_max], [ap_min, ap_max]])
la_ref  = np.array([[la_min, la_min], [la_max, la_max]])
kge_ref = np.full_like(ap_ref, KGE_THRESH)
fig_p.add_trace(go.Surface(
    x=ap_ref, y=la_ref, z=kge_ref,
    colorscale=[[0,'rgba(70,130,180,0.12)'],[1,'rgba(70,130,180,0.12)']],
    opacity=0.3, showscale=False, name=f'KGE={KGE_THRESH}', hoverinfo='skip',
))

# Adequacy boundary (plane ∩ KGE=0.5)
if valid.any():
    kge_on_line = b0 + b1*ap_line[valid] + b2*la_line[valid]
    fig_p.add_trace(go.Scatter3d(
        x=ap_line[valid], y=la_line[valid], z=kge_on_line,
        mode='lines',
        line=dict(color='navy', width=5, dash='dash'),
        name=f'Adequacy boundary (KGE={KGE_THRESH})',
    ))

# Scatter — v3 triangles, v4 circles
for (src, fail), grp in df.groupby(['source', 'fail']):
    symbol = 'diamond' if src == 'v3' else 'circle'
    fig_p.add_trace(go.Scatter3d(
        x=grp['ap_m'], y=grp['log_area'], z=grp['kge'],
        mode='markers+text',
        text=grp['label'],
        textposition='top center',
        textfont=dict(size=8),
        marker=dict(size=8 if src=='v3' else 6, color=grp['cplotly'],
                    symbol=symbol,
                    line=dict(color='white', width=0.5)),
        name=f'{src} / {fail}',
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Static A/P: %{x:.0f} m<br>'
            'Area: %{customdata:.0f} ha<br>'
            'KGE: %{z:.3f}<extra></extra>'
        ),
        customdata=grp['mean_jrc_ha'],
    ))

fig_p.update_layout(
    title=dict(
        text=(f'KGE vs Static A/P vs Reservoir Area — v3+v4 (N={len(df)})<br>'
              f'<sub>KGE = {b0:.2f} + {b1:.5f}·AP + {b2:.2f}·log₁₀(area)  '
              f'|  R² = {r2:.2f}</sub>'),
        x=0.5, font=dict(size=13)),
    scene=dict(
        xaxis=dict(title='Static A/P (m)'),
        yaxis=dict(title='log₁₀(Area ha)',
                   tickvals=LA_TICKS, ticktext=LA_LBLS),
        zaxis=dict(title='KGE', range=[-0.6, 1.15]),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', font=dict(size=9)),
    margin=dict(l=0, r=0, t=90, b=0),
    width=1000, height=720,
)

out_html = OUT_DIR / '3d_kge_staticap_area.html'
fig_p.write_html(str(out_html))
print(f'Saved: {out_html}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. MATPLOTLIB static — 4 viewpoints
# ══════════════════════════════════════════════════════════════════════════════
VIEWS  = [(25, -60), (25, 30), (10, -90), (70, -60)]
TITLES = ['Vista oblíqua (principal)', 'Vista lateral direita',
          'Vista frontal (A/P axis)',  'Vista superior']

fig_m, axes_m = plt.subplots(2, 2, figsize=(14, 11),
                              subplot_kw={'projection': '3d'})
axes_m = axes_m.flatten()

for ax, (elev, azim), title in zip(axes_m, VIEWS, TITLES):
    ax.plot_surface(ap_grid, la_grid, kge_grid,
                    alpha=0.22, color='#FFF9C4', edgecolor='none', zorder=1)
    ax.plot_surface(ap_ref, la_ref, kge_ref,
                    alpha=0.12, color='steelblue', edgecolor='none', zorder=0)

    if valid.any():
        kge_on_line = b0 + b1*ap_line[valid] + b2*la_line[valid]
        ax.plot(ap_line[valid], la_line[valid], kge_on_line,
                'navy', lw=2, ls='--', zorder=5)

    for (src, fail), grp in df.groupby(['source', 'fail']):
        ax.scatter(grp['ap_m'], grp['log_area'], grp['kge'],
                   c=grp['color'],
                   s=60 if src == 'v3' else 38,
                   marker='^' if src == 'v3' else 'o',
                   edgecolors='white', linewidths=0.4,
                   alpha=0.90, zorder=6)

    ax.set_xlabel('A/P (m)',       fontsize=7, labelpad=4)
    ax.set_ylabel('log₁₀(Area)',   fontsize=7, labelpad=4)
    ax.set_zlabel('KGE',           fontsize=7, labelpad=3)
    ax.set_yticks(LA_TICKS); ax.set_yticklabels(LA_LBLS, fontsize=5)
    ax.tick_params(labelsize=5)
    ax.set_zlim(-0.6, 1.15)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

handles = [
    mpatches.Patch(fc='#2ca02c', label='adequate (KGE≥0.5)'),
    mpatches.Patch(fc='#d62728', label='noisy (α>1.5)'),
    mpatches.Patch(fc='#ff7f0e', label='flat (α<0.5)'),
    mpatches.Patch(fc='#9467bd', label='bias (β<0.6)'),
    mpatches.Patch(fc='#8c564b', label='low r'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#666',
               markersize=8, label='v3'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#666',
               markersize=6, label='v4'),
]
axes_m[0].legend(handles=handles, fontsize=6.5, loc='upper left')

fig_m.suptitle(
    f'KGE ~ Static A/P + log₁₀(Area)  |  '
    f'{b0:.2f} + {b1:.5f}·AP + {b2:.2f}·log₁₀(area)  |  '
    f'R²={r2:.2f}  (v3+v4, N={len(df)})',
    fontsize=9, fontweight='bold')
fig_m.tight_layout(rect=[0, 0, 1, 0.95])

out_png = OUT_DIR / '3d_kge_staticap_area.png'
fig_m.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig_m)
print(f'Saved: {out_png}')
