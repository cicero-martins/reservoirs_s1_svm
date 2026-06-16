"""
Post-processing: A/P × SAR quality — global GROWL pilot.
Run AFTER downloading GEE exports from Drive to GEE_DIR.

Input files (in GEE_DIR):
  SAR_area_{name}_{resId}.csv  — SAR biweekly area (GEE export)
  JRC_area_{name}_{resId}.csv  — JRC monthly optical area (GEE export)

Output:
  pilot_kge_results.csv   — KGE, A/P, N_months per reservoir
  AP_vs_KGE_global.png    — scatter plot A/P × KGE
"""
import csv, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr, linregress

# ── Paths ─────────────────────────────────────────────────────────
BASE    = Path('F:/reservoirs_s1_svm/validation_data')
GEE_DIR = BASE / 'GROWL_SAR_pilot'    # downloaded from Google Drive
AP_CSV  = BASE / 'morphometric_analysis/shoreline_compactness/AP_all_reservoirs.csv'
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'
OUT_DIR = BASE / 'morphometric_analysis/shoreline_compactness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MIN_MONTHS = 12    # minimum overlapping months for reliable KGE
VALID_FRAC   = 0.90  # minimum JRC valid-pixel fraction (cloud-free) to include a month

# Reservoirs excluded from KGE analysis with documented physical reasons
EXCLUDED = {
    # HydroLAKES Lake_type = 1 (natural lakes — method targets managed reservoirs)
    'Panam':            'HydroLAKES Lake_type=1 (natural lake); also near-constant area (wind variance)',
    'Kakki':            'HydroLAKES Lake_type=1 (natural lake)',
    'Ry_de_Rome_BE':    'HydroLAKES Lake_type=1 (natural lake); very small volume (0.95 Mm³)',
    'OShannassy_AU':    'HydroLAKES Lake_type=1 (natural lake); very small volume (1.09 Mm³)',
    # Reservoirs with physical limitations incompatible with SAR-JRC comparison
    'Leech_US':         'minimal regulation; seasonal ice + multi-orbit S1 mixing (n_land=3)',
    'Winnibigoshish_US':'minimal regulation; seasonal ice + systematic SAR underdetection (n_land=2)',
}

# ── KGE ───────────────────────────────────────────────────────────
def kge(sim, obs):
    if len(sim) < 4 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r, _ = pearsonr(sim, obs)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# ── Load a GEE CSV (handles .geo and system:index columns) ────────
def load_gee_csv(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            # Drop GEE system columns
            rows.append({k: v for k, v in row.items()
                         if k not in ('.geo', 'system:index')})
    return rows

# ── Sicilian anchors (pre-computed vs PlanetScope) ────────────────
anchors = {}
with open(AP_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['validated'] == 'True' and row['KGE']:
            anchors[row['reservoir']] = {
                'name':    row['reservoir'],
                'ap_m':    float(row['AP_m']),
                'kge':     float(row['KGE']),
                'n_months': None,   # validated vs PlanetScope, not monthly
                'group':   'Sicily (PlanetScope)',
                'country': 'Italy',
            }

# ── Country → region mapping ──────────────────────────────────────
REGION = {
    'India': 'Asia', 'Pakistan': 'Asia', 'China': 'Asia',
    'Spain': 'Europe', 'Belgium': 'Europe', 'France': 'Europe',
    'USA': 'N. America', 'Mexico': 'N. America', 'Canada': 'N. America',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'Italy': 'Sicily (PlanetScope)',
}
pilot_country = {}
with open(PILOT_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row.get('note') == 'GEE_process':
            pilot_country[row['RES_ID']] = row.get('Country', '')

# ── Check GEE export availability ─────────────────────────────────
if not GEE_DIR.exists():
    print(f"⚠  GEE export folder not found: {GEE_DIR}")
    print("   Crie a pasta e baixe os arquivos do Google Drive (pasta GROWL_SAR_pilot).")
    raise SystemExit

sar_files = sorted(GEE_DIR.glob('SAR_area_*.csv'))
jrc_files = sorted(GEE_DIR.glob('JRC_area_*.csv'))
print(f"Arquivos encontrados: {len(sar_files)} SAR, {len(jrc_files)} JRC")

if not sar_files:
    print("Nenhum export SAR encontrado. Rode exportGlobalPilot.js no GEE primeiro.")
    raise SystemExit

# ── Pair SAR + JRC files by suffix (name_resId) ──────────────────
def extract_key(path, prefix):
    """e.g. SAR_area_Manikdoh_3345.csv → 'Manikdoh_3345'"""
    stem = path.stem  # SAR_area_Manikdoh_3345
    return stem[len(prefix):]   # Manikdoh_3345

sar_map = {extract_key(f, 'SAR_area_'): f for f in sar_files}
jrc_map = {extract_key(f, 'JRC_area_'): f for f in jrc_files}

# ── Process each paired reservoir ────────────────────────────────
results = []
for key, sar_path in sorted(sar_map.items()):
    jrc_path = jrc_map.get(key)
    parts    = key.rsplit('_', 1)
    res_name = parts[0]
    res_id   = parts[1] if len(parts) == 2 else ''

    if res_name in EXCLUDED:
        print(f"\n[{key}] EXCLUDED — {EXCLUDED[res_name]}")
        continue

    print(f"\n[{key}]")

    # --- SAR: aggregate to monthly mean ---
    sar_rows = load_gee_csv(sar_path)
    sar_by_month = {}
    for row in sar_rows:
        date = row.get('data', row.get('date', ''))[:10]
        if not date or date < '2014-01-01':
            continue
        month = date[:7]
        try:
            area = float(row['area_ha'])
            sar_by_month.setdefault(month, []).append(area)
        except (ValueError, KeyError):
            pass
    sar_monthly = {m: np.mean(v) for m, v in sar_by_month.items()}
    ap_m = None
    if sar_rows:
        try:
            ap_m = float(sar_rows[0]['ap_m'])
        except (ValueError, KeyError):
            pass
    print(f"  SAR: {len(sar_monthly)} months, A/P = {ap_m:.1f} m" if ap_m else
          f"  SAR: {len(sar_monthly)} months")

    # --- JRC: filter by valid_frac ---
    if not jrc_path:
        print("  JRC: no file — skipping")
        continue
    jrc_rows = load_gee_csv(jrc_path)
    jrc_monthly = {}
    for row in jrc_rows:
        date = row.get('date', '')[:10]
        if not date or date < '2014-01-01':
            continue
        month = date[:7]
        try:
            vf   = float(row.get('valid_frac', 0))
            area = float(row['jrc_area_ha'])
            if vf >= VALID_FRAC:
                jrc_monthly[month] = area
        except (ValueError, KeyError):
            pass
    print(f"  JRC: {len(jrc_monthly)} valid months (valid_frac ≥ {VALID_FRAC})")

    # --- Overlapping months ---
    common = sorted(set(sar_monthly) & set(jrc_monthly))
    print(f"  Overlap: {len(common)} months")
    if len(common) < N_MIN_MONTHS:
        print(f"  ⚠ Insufficient overlap (< {N_MIN_MONTHS}) — skipping KGE")
        continue

    sim = np.array([sar_monthly[m] for m in common])
    obs = np.array([jrc_monthly[m] for m in common])

    # --- Normalize by max within overlap period ---
    sim_n = sim / sim.max() if sim.max() > 0 else sim
    obs_n = obs / obs.max() if obs.max() > 0 else obs

    kge_val = kge(sim_n, obs_n)

    country = pilot_country.get(res_id, '')
    region  = REGION.get(country, 'Other')

    print(f"  KGE = {kge_val:.3f}  (A/P = {ap_m:.1f} m, N = {len(common)} months)")
    results.append({
        'name':     res_name,
        'group':    region,
        'country':  country,
        'growl_id': res_id,
        'ap_m':     ap_m,
        'kge':      round(kge_val, 4),
        'n_months': len(common),
        'sar_months': len(sar_monthly),
        'jrc_months': len(jrc_monthly),
        'period_start': common[0],
        'period_end':   common[-1],
    })

# ── Combine with Sicilian anchors ─────────────────────────────────
all_points = list(anchors.values()) + results

if not all_points:
    print("\nNenhum resultado para plotar. Aguarde os exports GEE.")
    raise SystemExit

# ── Save results CSV ──────────────────────────────────────────────
results_path = OUT_DIR / 'pilot_kge_results.csv'
fields = ['name','group','country','growl_id','ap_m','kge','n_months',
          'sar_months','jrc_months','period_start','period_end']
all_rows = []
for p in all_points:
    all_rows.append({f: p.get(f, '') for f in fields})
with open(results_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(all_rows)
print(f"\nResultados salvos: {results_path}")

# ── Figure: A/P vs KGE ───────────────────────────────────────────
COLORS = {
    'Sicily (PlanetScope)': '#e41a1c',  # red
    'Asia':       '#377eb8',            # blue
    'Europe':     '#4daf4a',            # green
    'N. America': '#ff7f00',            # orange
    'Oceania':    '#984ea3',            # purple
    'Other':      '#999999',            # grey
}
MARKERS = {
    'Sicily (PlanetScope)': 's',  # square
    'Asia':       'o',
    'Europe':     'D',
    'N. America': '^',
    'Oceania':    'P',
    'Other':      'x',
}

valid = [p for p in all_points if p.get('ap_m') and p.get('kge') is not None
         and not (isinstance(p['kge'], float) and np.isnan(p['kge']))]

ap  = np.array([p['ap_m'] for p in valid])
kge_arr = np.array([p['kge']  for p in valid])

fig, ax = plt.subplots(figsize=(8, 6))

for group in COLORS:
    pts = [p for p in valid if p['group'] == group]
    if not pts:
        continue
    x = [p['ap_m'] for p in pts]
    y = [p['kge']  for p in pts]
    label = group if group == 'Sicily (PlanetScope)' else group
    ax.scatter(x, y, color=COLORS[group], marker=MARKERS[group],
               s=80, zorder=5, label=label,
               edgecolors='white', linewidths=0.5)
    # Annotate name
    for p in pts:
        ax.annotate(p['name'], (p['ap_m'], p['kge']),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7, color=COLORS[group], alpha=0.85)

# Regression over all valid points
if len(valid) >= 4:
    slope, intercept, r, p_val, _ = linregress(ap, kge_arr)
    x_fit = np.linspace(ap.min() * 0.9, ap.max() * 1.05, 200)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', lw=1.5, alpha=0.7, zorder=3)
    r2 = r**2
    ax.text(0.97, 0.07,
            f'KGE = {slope*1000:.2f}×10⁻³·A/P + {intercept:.3f}\nR² = {r2:.3f}  (N = {len(valid)})',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', alpha=0.9))
    print(f"\nRegression: KGE = {slope:.6f} × A/P + {intercept:.4f}  R² = {r2:.3f}")

ax.set_xlabel('Shoreline Compactness — A/P (m)', fontsize=11)
ax.set_ylabel('KGE (SAR vs Optical Area)', fontsize=11)
ax.set_title('A/P Ratio vs SAR Classification Quality — Global Pilot', fontsize=12)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=min(0.5, kge_arr.min() - 0.05))

fig.tight_layout()
fig_path = OUT_DIR / 'AP_vs_KGE_global.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figura salva: {fig_path}")

# ── Status summary ────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Âncoras sicilianas: {len(anchors)}")
print(f"Piloto global com KGE: {len(results)}")
print(f"Total na figura: {len(valid)}")
if len(results) > 0:
    print(f"\nKGE global — min={min(r['kge'] for r in results):.3f}  "
          f"max={max(r['kge'] for r in results):.3f}  "
          f"mean={np.mean([r['kge'] for r in results]):.3f}")
