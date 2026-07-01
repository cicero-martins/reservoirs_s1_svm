"""
plot_kge_ceiling.py

Calcula e visualiza o KGE máximo teórico imposto pelo ruído de pixel misto do JRC.

MODELO:
  Seja A_true(t) a área real mensal. Os dois instrumentos medem:
      JRC (obs): A_obs(t)  = A_true(t) + ε_JRC(t),   ε_JRC ~ N(0, σ_noise)
      SAR (sim): A_sim(t)  = A_true(t) + ε_SAR(t)

  Se o SAR fosse perfeito (ε_SAR = 0):
      r_max   = σ_signal / sqrt(σ_signal² + σ_noise²)
              = sqrt(1 − (σ_noise / σ_obs)²)
      α_max   = σ_sim / σ_obs = σ_signal / σ_obs = r_max
      β_max   = 1  (sem bias sistemático)
      KGE_max = 1 − √[(r_max−1)² + (α_max−1)²]
              = 1 − √2 × (1 − r_max)

  onde:
      σ_obs   = desvio padrão observado da série JRC mensal (sinal + ruído)
      σ_signal= sqrt(max(σ_obs² − σ_noise², 0))  (sinal verdadeiro estimado)
      σ_noise = (d_JRC/2) × P / 10000  [ha]
              = (15 m) × mean_jrc_ha / ap_m   [ha]

  d_JRC = 30 m  (Landsat), d_S1 = 10 m (Sentinel-1)

HIPÓTESES e LIMITAÇÕES:
  1. O ruído de pixel misto é gaussiano e independente entre meses (upper bound
     conservador — para reservatórios estáveis é uma sobreestimação).
  2. Só se modela o ruído de pixel misto; ruído de nuvens / composição temporal
     não é incluído (KGE_max é portanto um tecto optimista).
  3. O SAR perfeito corresponderia a ε_SAR = 0 e β = 1 (sem bias).

Outputs:
  analysis/method_comparison_output/kge_ceiling.png
  Imprime tabela completa com σ_obs, σ_noise, SNR, KGE_max vs KGE_obs.
"""

import sys, warnings, pathlib, re as _re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR    = Path('analysis/method_comparison_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)
KGE_THRESH = 0.5
D_JRC      = 30.0   # Landsat pixel (m)
D_S1       = 10.0   # Sentinel-1 pixel (m)
VALID_FRAC = 0.80

# ── Localizar todos os ficheiros JRC disponíveis ──────────────────────────────
jrc_index = {}
for p in pathlib.Path('raw_data').rglob('JRC_area_*.csv'):
    name = _re.sub(r'\s*\(\d+\)$', '', p.stem).replace('JRC_area_', '')
    if name not in jrc_index:
        jrc_index[name] = p


def load_jrc_std(name):
    """Retorna std observado da série JRC mensal (após valid_frac + sigma-clip)."""
    p = jrc_index.get(name)
    if p is None:
        return np.nan
    try:
        df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    except Exception:
        return np.nan
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= VALID_FRAC].copy()
    if df.empty or 'jrc_area_ha' not in df.columns:
        return np.nan
    m, s = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if s > 0:
        df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * s]
    return float(df['jrc_area_ha'].std()) if len(df) > 2 else np.nan


# ── Carregar KGE combinado v3 + v4 ───────────────────────────────────────────
v3 = pd.read_csv('analysis/pilot_kge_v3.csv'); v3['source'] = 'v3'
v4 = pd.read_csv('analysis/pilot_kge_v4.csv'); v4['source'] = 'v4'
keep = ['name','ap_m','kge','r','alpha','beta','mean_jrc_ha','source']
df   = pd.concat([v3[keep], v4[keep]], ignore_index=True)
df   = df.dropna(subset=['ap_m','kge','mean_jrc_ha']).reset_index(drop=True)

# ── σ_obs por reservatório (da série JRC real) ────────────────────────────────
df['sigma_obs'] = df['name'].apply(load_jrc_std)

# ── σ_noise: ruído teórico de pixel misto do JRC ─────────────────────────────
# P ≈ A / (A/P)  →  σ_noise = (d_JRC/2) × P / 10000
# Em ha:  σ_noise = (d_JRC/2) × mean_jrc_ha / ap_m   (simplificação)
df['sigma_noise_JRC'] = (D_JRC / 2) * df['mean_jrc_ha'] / df['ap_m']
df['sigma_noise_S1']  = (D_S1  / 2) * df['mean_jrc_ha'] / df['ap_m']

# ── Fracção do ruído vs sinal observado ──────────────────────────────────────
df['noise_frac'] = df['sigma_noise_JRC'] / df['sigma_obs']   # θ = σ_noise / σ_obs

# ── r_max e KGE_max ──────────────────────────────────────────────────────────
# r_max = sqrt(max(1 - θ², 0))
# KGE_max = 1 - sqrt(2) * (1 - r_max)   [com α_max=r_max, β_max=1]
df['r_max']   = np.sqrt(np.maximum(1 - df['noise_frac']**2, 0))
df['kge_max'] = 1 - np.sqrt(2) * (1 - df['r_max'])

# ── Eficiência do SAR: quanto do tecto teórico é atingido ────────────────────
# Eficiência = KGE_obs / KGE_max (só faz sentido quando KGE_max > 0)
df['efficiency'] = np.where(df['kge_max'] > 0,
                            df['kge'] / df['kge_max'], np.nan)

df = df.sort_values('ap_m').reset_index(drop=True)

# ── Imprimir tabela ───────────────────────────────────────────────────────────
hdr = (f"{'Name':<22}  {'A/P':>5}  {'σ_obs':>6}  {'σ_noise':>7}  "
       f"{'θ=σn/σobs':>9}  {'r_max':>6}  {'KGE_max':>7}  "
       f"{'KGE_obs':>7}  {'effic.':>7}")
print(hdr)
print('-' * len(hdr))
for _, r in df.iterrows():
    print(f"  {r['name']:<22}  {r['ap_m']:5.0f}  "
          f"{r['sigma_obs']:6.1f}  {r['sigma_noise_JRC']:7.1f}  "
          f"{r['noise_frac']:9.2f}  {r['r_max']:6.3f}  "
          f"{r['kge_max']:7.3f}  {r['kge']:7.3f}  "
          f"{r['efficiency']:7.1%}")
print('-' * len(hdr))
print(f"\nKGE_max < 0.90 em {(df['kge_max'] < 0.90).sum()} reservatorios")
print(f"KGE_obs > KGE_max em {(df['kge'] > df['kge_max']).sum()} (inconsistencia modelo)")

# ── FIGURA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0.38})

COLOR_SRC = {'v3': '#1565C0', 'v4': '#E65100'}

# ── Painel (a): KGE_obs e KGE_max vs A/P ────────────────────────────────────
ax = axes[0]

# Banda entre KGE_max e KGE_obs = "margem por limitação JRC"
ap_s = df['ap_m'].values
ax.fill_between(ap_s, df['kge_max'], 1.0,
                color='#EF9A9A', alpha=0.25, label='Tecto JRC inacessível')
ax.fill_between(ap_s, df['kge'], df['kge_max'],
                color='#FFF9C4', alpha=0.55, label='Margem até tecto JRC')

# Linha KGE_max suave
ap_sort = np.linspace(ap_s.min()-5, ap_s.max()+5, 300)
# curva analítica: r_max(ap) quando σ_obs é aproximado
# (usamos valores reais — só traça os pontos)
ax.plot(df['ap_m'], df['kge_max'], 'v--',
        color='#C62828', ms=6, lw=1.2, alpha=0.80,
        label='KGE$_{max}$ (SAR perfeito, ruído JRC)')

# KGE observado
for src, grp in df.groupby('source'):
    ax.scatter(grp['ap_m'], grp['kge'],
               s=55 if src=='v3' else 38,
               marker='^' if src=='v3' else 'o',
               color=COLOR_SRC[src],
               edgecolors='white', linewidths=0.5,
               alpha=0.90, zorder=5,
               label=f'KGE$_{{obs}}$ ({src})')

ax.axhline(KGE_THRESH, color='gray', lw=1.0, ls=':', alpha=0.7,
           label=f'KGE = {KGE_THRESH}')
ax.axhline(1.0, color='black', lw=0.5, ls='-', alpha=0.3)
ax.set_xlabel('A/P estático (m)', fontsize=9)
ax.set_ylabel('KGE', fontsize=9)
ax.set_title('(a) KGE observado vs tecto teórico por ruído JRC (30 m pixel)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right')
ax.set_ylim(-0.5, 1.1)
ax.grid(alpha=0.25)

# ── Painel (b): eficiência relativa = KGE_obs / KGE_max ─────────────────────
ax = axes[1]
for src, grp in df.groupby('source'):
    ax.scatter(grp['ap_m'], grp['efficiency'],
               s=55 if src=='v3' else 38,
               marker='^' if src=='v3' else 'o',
               color=COLOR_SRC[src],
               edgecolors='white', linewidths=0.5,
               alpha=0.90, zorder=5, label=f'{src}')

# Anotar reservatórios com eficiência < 60% ou > 100%
for _, row in df.iterrows():
    if pd.isna(row['efficiency']):
        continue
    if row['efficiency'] < 0.60 or row['efficiency'] > 1.05:
        ax.annotate(row['name'].replace('_',' '),
                    (row['ap_m'], row['efficiency']),
                    fontsize=5.5, xytext=(4,3),
                    textcoords='offset points', color='#444')

ax.axhline(1.0, color='navy', lw=1.2, ls='--', alpha=0.7,
           label='100% do tecto teórico')
ax.axhline(0.0, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('A/P estático (m)', fontsize=9)
ax.set_ylabel('Eficiência  =  KGE$_{obs}$ / KGE$_{max}$', fontsize=9)
ax.set_title('(b) Eficiência do SAR relativa ao tecto JRC',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(-0.6, 1.25)
ax.grid(alpha=0.25)

# Nota metodológica
fig.text(0.5, -0.04,
         (r'$\sigma_{noise}^{JRC} = \frac{d_{JRC}}{2} \times \frac{A}{A/P}$  '
          r'$= \frac{15\,m \times \bar{A}_{JRC}}{A/P}$   |   '
          r'$r_{max} = \sqrt{1 - \left(\frac{\sigma_{noise}}{\sigma_{obs}}\right)^2}$   |   '
          r'$KGE_{max} = 1 - \sqrt{2}\,(1 - r_{max})$   |   '
          r'$d_{JRC}=30\,m$, $d_{S1}=10\,m$'),
         ha='center', fontsize=8, color='#555',
         bbox=dict(boxstyle='round', fc='#F5F5F5', alpha=0.8))

fig.suptitle(
    f'Tecto de KGE imposto pelo ruído de pixel misto do JRC (Landsat 30 m)  '
    f'— v3+v4, N={len(df)}',
    fontsize=10, fontweight='bold')
fig.tight_layout(rect=[0, 0.06, 1, 1])

out = OUT_DIR / 'kge_ceiling.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
