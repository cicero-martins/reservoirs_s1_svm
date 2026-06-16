"""
Diagnostic: compare KGE vs HydroLAKES A/P (max extent)
                          vs Proxy Median A/P (operational extent)

Proxy Median A/P = A/P_max × sqrt(median_fill_fraction)
  Rationale: for a self-similar shape, perimeter ∝ sqrt(area),
  so A/P = area/perimeter ∝ sqrt(area).
  This gives the effective compactness at the typical water level.

Also plots: fill_fraction vs KGE to check if partial-fill explains low KGE.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress, pearsonr

BASE    = Path('F:/reservoirs_s1_svm/validation_data')
GEE_DIR = BASE / 'GROWL_SAR_pilot'
RES_CSV = BASE / 'morphometric_analysis/shoreline_compactness/pilot_kge_results.csv'
OUT_DIR = BASE / 'morphometric_analysis/shoreline_compactness'

COLORS = {
    'Sicily (PlanetScope)': '#e41a1c',
    'Asia':       '#377eb8',
    'Europe':     '#4daf4a',
    'N. America': '#ff7f00',
    'Oceania':    '#984ea3',
}
MARKERS = {
    'Sicily (PlanetScope)': 's',
    'Asia': 'o', 'Europe': 'D', 'N. America': '^', 'Oceania': 'P',
}

# ── Load KGE results ───────────────────────────────────────────
records = {}
with open(RES_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row['kge'] and row['ap_m']:
            records[row['name']] = {
                'name':    row['name'],
                'group':   row['group'],
                'kge':     float(row['kge']),
                'ap_m':    float(row['ap_m']),
                'n_months': int(row['n_months']) if row['n_months'] else None,
            }

# ── Compute median fill fraction from SAR CSVs ─────────────────
print(f"\n{'Reservoir':30s}  {'ap_m':>8}  {'fill%':>6}  {'proxy_ap':>9}  KGE")
print('─' * 70)

for sar_path in sorted(GEE_DIR.glob('SAR_area_*.csv')):
    stem = sar_path.stem                    # SAR_area_Manikdoh_3345
    key  = stem[len('SAR_area_'):]          # Manikdoh_3345
    name = key.rsplit('_', 1)[0]            # Manikdoh

    if name not in records:
        continue

    areas, aoi_area = [], None
    with open(sar_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                area = float(row['area_ha'])
                if aoi_area is None:
                    aoi_area = float(row['area_aoi_ha'])
                if area > 0:
                    areas.append(area)
            except (ValueError, KeyError):
                pass

    if not areas or not aoi_area:
        continue

    median_area = np.median(areas)
    fill_frac   = min(median_area / aoi_area, 1.0)  # cap at 1.0 (classifier can overcount)
    proxy_ap    = records[name]['ap_m'] * np.sqrt(fill_frac)

    records[name]['fill_frac'] = fill_frac
    records[name]['proxy_ap']  = proxy_ap
    records[name]['median_area_ha'] = median_area
    records[name]['aoi_area_ha']    = aoi_area

    print(f"{name:30s}  {records[name]['ap_m']:8.1f}  {fill_frac*100:5.1f}%"
          f"  {proxy_ap:9.1f}  {records[name]['kge']:.3f}")

# Sicily anchors have no SAR pilot CSV — keep them with ap_m only (no proxy_ap)
pilot  = [r for r in records.values() if r['group'] != 'Sicily (PlanetScope)' and 'proxy_ap' in r]
sicily = [r for r in records.values() if r['group'] == 'Sicily (PlanetScope)']

print(f"\nPilot reservoirs with data: {len(pilot)}")

# ── Figure: 3 panels ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

panels = [
    ('ap_m',      'HydroLAKES A/P (m)  [max extent]',      True),
    ('proxy_ap',  'Proxy Median A/P (m)  [A/P × √fill]',    True),
    ('fill_frac', 'Median Fill Fraction (SAR area / AOI)',   False),
]

for ax, (x_key, xlabel, add_sicily) in zip(axes, panels):
    # Plot pilot points
    for group in COLORS:
        if group == 'Sicily (PlanetScope)':
            continue
        pts = [r for r in pilot if r['group'] == group and x_key in r]
        if not pts:
            continue
        x = [p[x_key] for p in pts]
        y = [p['kge']  for p in pts]
        ax.scatter(x, y, color=COLORS[group], marker=MARKERS[group],
                   s=80, zorder=5, label=group, edgecolors='white', linewidths=0.5)
        for p in pts:
            ax.annotate(p['name'].replace('_', ' '),
                        (p[x_key], p['kge']),
                        xytext=(4, 3), textcoords='offset points',
                        fontsize=6.5, color=COLORS[group], alpha=0.85)

    # Add Sicily anchors only on A/P panels
    if add_sicily and sicily:
        xs = [p['ap_m'] for p in sicily]
        ys = [p['kge']  for p in sicily]
        ax.scatter(xs, ys, color=COLORS['Sicily (PlanetScope)'],
                   marker=MARKERS['Sicily (PlanetScope)'],
                   s=100, zorder=6, label='Sicily (PlanetScope)',
                   edgecolors='white', linewidths=0.5)
        for p in sicily:
            ax.annotate(p['name'], (p['ap_m'], p['kge']),
                        xytext=(4, 3), textcoords='offset points',
                        fontsize=6.5, color=COLORS['Sicily (PlanetScope)'], alpha=0.85)

    # Regression on pilot only
    pts_all = [r for r in pilot if x_key in r]
    if len(pts_all) >= 4:
        xa = np.array([p[x_key] for p in pts_all])
        ya = np.array([p['kge']  for p in pts_all])
        r_val, p_val = pearsonr(xa, ya)
        slope, intercept, *_ = linregress(xa, ya)
        x_fit = np.linspace(xa.min() * 0.9, xa.max() * 1.05, 200)
        ax.plot(x_fit, slope * x_fit + intercept, 'k--', lw=1.5, alpha=0.6, zorder=3)
        ax.text(0.97, 0.04,
                f'r = {r_val:.3f}  R² = {r_val**2:.3f}\np = {p_val:.3f}  N = {len(pts_all)}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', alpha=0.9))

    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('KGE (SAR vs JRC area)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)

axes[0].set_title('Max-extent A/P (HydroLAKES)', fontsize=11)
axes[1].set_title('Proxy Median A/P', fontsize=11)
axes[2].set_title('Fill Fraction', fontsize=11)

fig.suptitle('Diagnosing A/P metric: max-extent vs operational extent', fontsize=12)
fig.tight_layout()
out = OUT_DIR / 'AP_median_diagnostic.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nFigura salva: {out}')

# ── Correlation summary ────────────────────────────────────────
print(f"\n{'Metric':35s}  {'r':>7}  {'R²':>7}  {'p':>8}")
print('─' * 60)
for x_key, label in [('ap_m','HydroLAKES A/P'), ('proxy_ap','Proxy Median A/P'), ('fill_frac','Fill Fraction')]:
    pts = [r for r in pilot if x_key in r]
    if len(pts) < 4:
        continue
    xa = np.array([p[x_key] for p in pts])
    ya = np.array([p['kge']  for p in pts])
    r_val, p_val = pearsonr(xa, ya)
    print(f"{label:35s}  {r_val:7.3f}  {r_val**2:7.3f}  {p_val:8.4f}")
