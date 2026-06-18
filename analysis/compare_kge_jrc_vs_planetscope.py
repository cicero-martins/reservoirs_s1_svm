"""
Cross-method comparison for 4 validated Sicilian reservoirs.

Three SAR/optical series compared against PlanetScope KGE (ground truth):
  SAR_manual  — existing manually-trained SVM (area_{name}_2014-25.csv)
  SAR_auto    — WorldCover auto-trained SVM (SAR_auto_{name}_sicily.csv from GEE)
  JRC         — monthly optical area (JRC_area_{name}_sicily.csv from GEE)

KGE computed for all pairs:
  KGE(SAR_manual vs JRC)    — is JRC a good substitute for PlanetScope?
  KGE(SAR_auto   vs JRC)    — does auto-training match manual in the pilot metric?
  KGE(SAR_manual vs SAR_auto) — do the two classifiers agree on the same reservoir?
  KGE(SAR_auto   vs PlanetScope)  [proxy, via KGE(auto vs manual) × KGE(manual vs PS)]

Run after downloading from Google Drive → validation_data/GROWL_SAR_pilot/:
  SAR_auto_{Ancipa|Rosamarina|Poma|Pozzillo}_sicily.csv
  JRC_area_{Ancipa|Rosamarina|Poma|Pozzillo}_sicily.csv
"""
import csv, re, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────
BASE    = Path('F:/reservoirs_s1_svm/validation_data')
SC_DIR  = BASE / 'morphometric_analysis/shoreline_compactness'
GEE_DIR = BASE / 'GROWL_SAR_pilot'

VALID_FRAC   = 0.90
SAR_MIN_FRAC = 0.02
N_MIN        = 12

RESERVOIRS = ['Ancipa', 'Rosamarina', 'Poma', 'Pozzillo']
COLORS = {
    'Ancipa':     '#e41a1c',
    'Rosamarina': '#377eb8',
    'Poma':       '#4daf4a',
    'Pozzillo':   '#ff7f00',
}

# ── KGE ───────────────────────────────────────────────────────────
def kge(sim, obs):
    if len(sim) < 4 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r, _ = pearsonr(sim, obs)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def kge_pair(a_monthly, b_monthly, normalize=True):
    common = sorted(set(a_monthly) & set(b_monthly))
    if len(common) < N_MIN:
        return np.nan, len(common)
    s = np.array([a_monthly[m] for m in common])
    o = np.array([b_monthly[m] for m in common])
    if normalize:
        s = s / s.max() if s.max() > 0 else s
        o = o / o.max() if o.max() > 0 else o
    return kge(s, o), len(common)

def load_gee_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return [row for row in csv.DictReader(f)
                if row.get('system:index') != 'system:index']

# ── Load PlanetScope KGE ──────────────────────────────────────────
kge_planet = {}
with open(SC_DIR / 'AP_all_reservoirs.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row.get('validated') == 'True' and row.get('KGE'):
            kge_planet[row['reservoir']] = {
                'kge':  float(row['KGE']),
                'ap_m': float(row['AP_m']),
            }

# ── Check GEE exports ─────────────────────────────────────────────
missing = []
for n in RESERVOIRS:
    for kind in ['SAR_auto', 'JRC_area']:
        p = GEE_DIR / f'{kind}_{n}_sicily.csv'
        if not p.exists():
            missing.append(str(p.name))
if missing:
    print('⚠  Missing GEE exports:')
    for m in missing:
        print(f'   {m}')
    print('\nRun exportSicilianJRC.js in GEE Code Editor first.')
    raise SystemExit

# ── Load and process each reservoir ──────────────────────────────
all_series = {}   # name → {sar_manual, sar_auto, jrc} monthly dicts

for name in RESERVOIRS:
    ap_m  = kge_planet.get(name, {}).get('ap_m', None)

    # --- SAR manual (existing, biweekly → monthly mean) ---
    sar_man_path = SC_DIR / f'area_{name.lower()}_2014-25.csv'
    with open(sar_man_path, encoding='utf-8-sig') as f:
        man_rows = list(csv.DictReader(f))

    all_man = [float(r['areaLago']) for r in man_rows]
    aoi_est = max(all_man) * 1.05

    sar_man = {}
    for row in man_rows:
        raw = row.get('date', '').strip()
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', raw):
            m_, d_, y_ = raw.split('/')
            date = f'{y_}-{m_.zfill(2)}-{d_.zfill(2)}'
        else:
            date = raw[:10]
        if not date or date < '2014-01-01':
            continue
        try:
            area = float(row['areaLago'])
            if area >= SAR_MIN_FRAC * aoi_est:
                sar_man.setdefault(date[:7], []).append(area)
        except (ValueError, KeyError):
            pass
    sar_man_monthly = {m: np.mean(v) for m, v in sar_man.items()}

    # --- SAR auto (WorldCover, from GEE export) ---
    auto_rows = load_gee_csv(GEE_DIR / f'SAR_auto_{name}_sicily.csv')
    aoi_ha_auto = float(auto_rows[0]['area_aoi_ha']) if auto_rows else aoi_est
    n_land  = int(auto_rows[0].get('n_land_pts', 0)) if auto_rows else 0
    n_water = int(auto_rows[0].get('n_water_pts', 0)) if auto_rows else 0

    sar_auto = {}
    for row in auto_rows:
        date = row.get('date', '')[:10]
        if not date or date < '2014-01-01':
            continue
        try:
            area = float(row['area_ha'])
            if area >= SAR_MIN_FRAC * aoi_ha_auto:
                sar_auto.setdefault(date[:7], []).append(area)
        except (ValueError, KeyError):
            pass
    sar_auto_monthly = {m: np.mean(v) for m, v in sar_auto.items()}

    # --- JRC monthly ---
    jrc_rows = load_gee_csv(GEE_DIR / f'JRC_area_{name}_sicily.csv')
    jrc_monthly = {}
    for row in jrc_rows:
        date = row.get('date', '')[:10]
        if not date or date < '2014-01-01':
            continue
        try:
            vf   = float(row.get('valid_frac', 0))
            area = float(row['jrc_area_ha'])
            if vf >= VALID_FRAC:
                jrc_monthly[date[:7]] = area
        except (ValueError, KeyError):
            pass

    all_series[name] = {
        'sar_man':  sar_man_monthly,
        'sar_auto': sar_auto_monthly,
        'jrc':      jrc_monthly,
        'ap_m':     ap_m,
        'n_land':   n_land,
        'n_water':  n_water,
        'aoi_ha':   aoi_ha_auto,
    }

    print(f'\n{name}  (A/P={ap_m:.0f}m, n_land={n_land}, n_water={n_water})')
    print(f'  SAR_manual : {len(sar_man_monthly)} months')
    print(f'  SAR_auto   : {len(sar_auto_monthly)} months')
    print(f'  JRC valid  : {len(jrc_monthly)} months  (vf≥{VALID_FRAC})')

# ── Compute all KGE pairs ─────────────────────────────────────────
print(f'\n{"":=<70}')
print(f'{"Reservoir":12s}  {"KGE(Man/PS)":>11}  {"KGE(Man/JRC)":>12}  '
      f'{"KGE(Auto/JRC)":>13}  {"KGE(Auto/Man)":>13}')
print('─' * 65)

results = []
for name in RESERVOIRS:
    d    = all_series[name]
    ap_m = d['ap_m']
    kge_ps  = kge_planet.get(name, {}).get('kge', np.nan)

    kge_man_jrc,   n_mj  = kge_pair(d['sar_man'],  d['jrc'])
    kge_auto_jrc,  n_aj  = kge_pair(d['sar_auto'], d['jrc'])
    kge_auto_man,  n_am  = kge_pair(d['sar_auto'], d['sar_man'])

    print(f'{name:12s}  {kge_ps:>11.3f}  {kge_man_jrc:>12.3f}  '
          f'{kge_auto_jrc:>13.3f}  {kge_auto_man:>13.3f}')
    results.append({
        'name':          name,
        'ap_m':          ap_m,
        'kge_man_ps':    round(kge_ps,       4),
        'kge_man_jrc':   round(kge_man_jrc,  4) if not np.isnan(kge_man_jrc)  else '',
        'kge_auto_jrc':  round(kge_auto_jrc, 4) if not np.isnan(kge_auto_jrc) else '',
        'kge_auto_man':  round(kge_auto_man, 4) if not np.isnan(kge_auto_man) else '',
        'n_man_jrc':     n_mj,
        'n_auto_jrc':    n_aj,
        'n_auto_man':    n_am,
        'n_land_auto':   d['n_land'],
    })

# ── Save CSV ──────────────────────────────────────────────────────
csv_out = SC_DIR / 'sicilian_kge_comparison.csv'
fields  = ['name','ap_m','kge_man_ps','kge_man_jrc','kge_auto_jrc','kge_auto_man',
           'n_man_jrc','n_auto_jrc','n_auto_man','n_land_auto']
with open(csv_out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(results)
print(f'\nCSV: {csv_out}')

# ── Figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.32)

# Row 0: time series per reservoir (4 panels)
for idx, name in enumerate(RESERVOIRS):
    ax  = fig.add_subplot(gs[0, idx])
    d   = all_series[name]
    all_m = sorted(set(d['sar_man']) | set(d['sar_auto']) | set(d['jrc']))
    x   = np.arange(len(all_m))

    ax.plot(x, [d['sar_man'].get(m, np.nan)  for m in all_m],
            color=COLORS[name], lw=0.9, label='SAR manual', alpha=0.9)
    ax.plot(x, [d['sar_auto'].get(m, np.nan) for m in all_m],
            color='darkorange', lw=0.8, ls='-.', label='SAR auto', alpha=0.85)
    ax.plot(x, [d['jrc'].get(m, np.nan)      for m in all_m],
            color='navy', lw=0.9, ls='--', label='JRC', alpha=0.75)

    yr_ticks  = [i for i, m in enumerate(all_m) if m.endswith('-01')]
    yr_labels = [m[:4] for i, m in enumerate(all_m) if m.endswith('-01')]
    ax.set_xticks(yr_ticks[::2]); ax.set_xticklabels(yr_labels[::2], fontsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel('Area (ha)', fontsize=7)
    ax.set_title(f'{name}  A/P={d["ap_m"]:.0f}m\nn_land(auto)={d["n_land"]}', fontsize=8)
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=6.5, loc='lower right')

# Row 1 left: KGE(Man/JRC) vs KGE(Man/PS) — is JRC a good reference?
ax1 = fig.add_subplot(gs[1, :2])
valid_mj = [r for r in results if r['kge_man_jrc'] != '']
if valid_mj:
    for r in valid_mj:
        ax1.scatter(r['kge_man_ps'], float(r['kge_man_jrc']),
                    color=COLORS[r['name']], s=130, zorder=5,
                    edgecolors='white', linewidths=0.8)
        ax1.annotate(r['name'], (r['kge_man_ps'], float(r['kge_man_jrc'])),
                     xytext=(5, 4), textcoords='offset points',
                     fontsize=9, color=COLORS[r['name']])
    lims = [0, 1.05]
    ax1.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='1:1')
    if len(valid_mj) >= 3:
        x_ = [r['kge_man_ps'] for r in valid_mj]
        y_ = [float(r['kge_man_jrc']) for r in valid_mj]
        rmse = np.sqrt(np.mean((np.array(x_) - np.array(y_))**2))
        bias = np.mean(np.array(y_) - np.array(x_))
        r_v, _ = pearsonr(x_, y_)
        ax1.text(0.05, 0.92,
                 f'r = {r_v:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:+.3f}',
                 transform=ax1.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8'))
ax1.set_xlim(0, 1.05); ax1.set_ylim(0, 1.05)
ax1.set_xlabel('KGE (SAR_manual vs PlanetScope)', fontsize=10)
ax1.set_ylabel('KGE (SAR_manual vs JRC)', fontsize=10)
ax1.set_title('Is JRC a credible substitute for PlanetScope?\n(same SAR classifier)', fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Row 1 right: KGE(Auto/JRC) vs KGE(Man/JRC) — does auto-training match manual?
ax2 = fig.add_subplot(gs[1, 2:])
valid_am = [r for r in results if r['kge_auto_jrc'] != '' and r['kge_man_jrc'] != '']
if valid_am:
    for r in valid_am:
        ax2.scatter(float(r['kge_man_jrc']), float(r['kge_auto_jrc']),
                    color=COLORS[r['name']], s=130, zorder=5,
                    edgecolors='white', linewidths=0.8)
        ax2.annotate(r['name'], (float(r['kge_man_jrc']), float(r['kge_auto_jrc'])),
                     xytext=(5, 4), textcoords='offset points',
                     fontsize=9, color=COLORS[r['name']])
    ax2.plot([0, 1.05], [0, 1.05], 'k--', lw=1, alpha=0.4, label='1:1')
    if len(valid_am) >= 3:
        x_ = [float(r['kge_man_jrc']) for r in valid_am]
        y_ = [float(r['kge_auto_jrc']) for r in valid_am]
        rmse = np.sqrt(np.mean((np.array(x_) - np.array(y_))**2))
        bias = np.mean(np.array(y_) - np.array(x_))
        r_v, _ = pearsonr(x_, y_)
        ax2.text(0.05, 0.92,
                 f'r = {r_v:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:+.3f}',
                 transform=ax2.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8'))
ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)
ax2.set_xlabel('KGE (SAR_manual vs JRC)', fontsize=10)
ax2.set_ylabel('KGE (SAR_auto vs JRC)', fontsize=10)
ax2.set_title('Does WorldCover auto-training match manual training?\n(same JRC reference)', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Row 2: A/P vs KGE — all methods overlaid
ax3 = fig.add_subplot(gs[2, :])
for r in results:
    ap = r['ap_m']
    c  = COLORS[r['name']]
    ax3.scatter(ap, r['kge_man_ps'],  color=c, marker='s', s=110, zorder=5,
                edgecolors='white', linewidths=0.8)
    if r['kge_man_jrc']  != '': ax3.scatter(ap, float(r['kge_man_jrc']),
                                             color=c, marker='o', s=90, zorder=4,
                                             edgecolors='white', linewidths=0.8)
    if r['kge_auto_jrc'] != '': ax3.scatter(ap, float(r['kge_auto_jrc']),
                                             color=c, marker='^', s=90, zorder=4,
                                             edgecolors='white', linewidths=0.8)
    # Label once
    ax3.annotate(r['name'], (ap, r['kge_man_ps']),
                 xytext=(5, 4), textcoords='offset points',
                 fontsize=9, color=c)

# Legend for marker shapes
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], marker='s', color='w', markerfacecolor='grey', ms=9, label='SAR_manual vs PlanetScope'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey', ms=9, label='SAR_manual vs JRC'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='grey', ms=9, label='SAR_auto vs JRC'),
]
ax3.legend(handles=legend_els, fontsize=9, loc='lower right')
ax3.set_xlabel('Shoreline Compactness — A/P (m)', fontsize=11)
ax3.set_ylabel('KGE', fontsize=11)
ax3.set_title('A/P vs KGE across all methods — Sicily validation reservoirs', fontsize=11)
ax3.set_ylim(0, 1.1); ax3.grid(True, alpha=0.3)

fig.suptitle('Sicily: SAR_manual / SAR_auto / JRC — cross-method KGE comparison\n'
             '(validates JRC as reference AND WorldCover auto-training for global pilot)',
             fontsize=12, y=1.01)

out = SC_DIR / 'KGE_JRC_vs_PlanetScope.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f'Figura: {out}')

# ── Interpretation summary ────────────────────────────────────────
print(f'\n{"="*65}')
print('INTERPRETATION:')
print('  KGE(Man/JRC) ≈ KGE(Man/PS)  → JRC is a credible substitute')
print('  KGE(Auto/JRC) ≈ KGE(Man/JRC) → WorldCover training is equivalent to manual')
print('  KGE(Auto/Man) ≈ 1.0          → both classifiers agree on the same reservoir')
