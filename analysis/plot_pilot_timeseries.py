"""
Diagnostic: SAR vs JRC area time series + annual KGE heatmap.

Runs on currently downloaded exports (even old ones) to assess:
  1. JRC optical area credibility (visual inspection per reservoir)
  2. Annual KGE stability (does quality vary by year?)
  3. Root cause of low KGE (SAR constant? Bias? Decorrelation?)

Output:
  timeseries_SAR_vs_JRC_{name}.png  — per-reservoir area comparison
  annual_KGE_heatmap.png            — KGE by reservoir × year
  pilot_annual_kge.csv              — tabular annual KGE
"""
import csv, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────
BASE    = Path('F:/reservoirs_s1_svm/validation_data')
GEE_DIR = BASE / 'GROWL_SAR_pilot'
OUT_DIR = BASE / 'morphometric_analysis/shoreline_compactness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_FRAC   = 0.90
N_MIN_MONTHS = 4    # minimum per year to compute annual KGE

EXCLUDED = {
    'Panam','Kakki','Ry_de_Rome_BE','OShannassy_AU','Leech_US','Winnibigoshish_US',
    'Upper_Coliban_AU','Minilla_ES',
}
SAR_MIN_FRAC = 0.02  # exclude SAR obs < 2% of AOI area (misclassification guard)

# ── KGE ───────────────────────────────────────────────────────────
def kge(sim, obs):
    if len(sim) < 4 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r, _ = pearsonr(sim, obs)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def load_gee_csv(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append({k: v for k, v in row.items()
                         if k not in ('.geo', 'system:index')})
    return rows

# ── Collect all reservoir data ─────────────────────────────────────
sar_files = sorted(GEE_DIR.glob('SAR_area_*.csv'))
reservoirs = {}

for sar_path in sar_files:
    key  = sar_path.stem[len('SAR_area_'):]
    name = key.rsplit('_', 1)[0]
    if name in EXCLUDED:
        continue
    jrc_path = GEE_DIR / f'JRC_area_{key}.csv'
    if not jrc_path.exists():
        continue

    sar_rows = load_gee_csv(sar_path)
    jrc_rows = load_gee_csv(jrc_path)

    ap_m    = float(sar_rows[0]['ap_m'])    if sar_rows else np.nan
    aoi_ha  = float(sar_rows[0]['area_aoi_ha']) if sar_rows else np.nan
    n_land  = int(sar_rows[0].get('n_land_pts', 0)) if sar_rows else 0
    n_water = int(sar_rows[0].get('n_water_pts', 0)) if sar_rows else 0

    # --- SAR: biweekly → monthly mean (filter near-zero misclassification events) ---
    sar_by_month = {}
    for row in sar_rows:
        date = row.get('date', '')[:10]
        if not date or date < '2014-01-01':
            continue
        month = date[:7]
        try:
            area = float(row['area_ha'])
            if not np.isnan(aoi_ha) and area < SAR_MIN_FRAC * aoi_ha:
                continue
            sar_by_month.setdefault(month, []).append(area)
        except (ValueError, KeyError):
            pass
    sar_monthly = {m: np.mean(v) for m, v in sar_by_month.items()}

    # --- JRC: filter by valid_frac ---
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

    reservoirs[name] = {
        'ap_m': ap_m,
        'aoi_ha': aoi_ha,
        'n_land': n_land,
        'n_water': n_water,
        'sar_monthly': sar_monthly,
        'jrc_monthly': jrc_monthly,
    }

# ── Figure 1: SAR vs JRC time series (4×5 grid) ──────────────────
names_sorted = sorted(reservoirs.keys(), key=lambda n: reservoirs[n]['ap_m'])
n_res = len(names_sorted)
ncols = 3
nrows = (n_res + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2), sharex=False)
axes = axes.flatten()

all_annual_kge = {}  # name → {year: kge}

for idx, name in enumerate(names_sorted):
    ax  = axes[idx]
    res = reservoirs[name]
    sar = res['sar_monthly']
    jrc = res['jrc_monthly']

    # Build aligned arrays for plotting
    all_months = sorted(set(sar) | set(jrc))
    sar_arr = np.array([sar.get(m, np.nan) for m in all_months])
    jrc_arr = np.array([jrc.get(m, np.nan) for m in all_months])
    x = np.arange(len(all_months))

    # Common months for KGE
    common = sorted(set(sar) & set(jrc))
    sim_c = np.array([sar[m] for m in common])
    obs_c = np.array([jrc[m] for m in common])
    kge_val = np.nan
    if len(common) >= 12:
        sim_n = sim_c / sim_c.max() if sim_c.max() > 0 else sim_c
        obs_n = obs_c / obs_c.max() if obs_c.max() > 0 else obs_c
        kge_val = kge(sim_n, obs_n)

    # Annual KGE
    annual_kge = {}
    years = sorted(set(m[:4] for m in common))
    for yr in years:
        yr_months = [m for m in common if m[:4] == yr]
        if len(yr_months) < N_MIN_MONTHS:
            continue
        s_yr = np.array([sar[m] for m in yr_months])
        o_yr = np.array([jrc[m] for m in yr_months])
        # normalize within each year separately
        s_n = s_yr / s_yr.max() if s_yr.max() > 0 else s_yr
        o_n = o_yr / o_yr.max() if o_yr.max() > 0 else o_yr
        annual_kge[yr] = kge(s_n, o_n)
    all_annual_kge[name] = annual_kge

    # Detect constant SAR (std < 1% of mean)
    sar_vals = [v for v in sar_arr if not np.isnan(v)]
    is_const = np.std(sar_vals) < 0.01 * np.mean(sar_vals) if sar_vals else True

    # Plot
    sar_plot = [sar_arr[i] if not np.isnan(sar_arr[i]) else None for i in range(len(x))]
    jrc_plot = [jrc_arr[i] if not np.isnan(jrc_arr[i]) else None for i in range(len(x))]

    ax.plot(x, sar_arr, color='#e41a1c', lw=0.8, label='SAR', alpha=0.85)
    ax.plot(x, jrc_arr, color='#377eb8', lw=0.8, label='JRC', alpha=0.85, ls='--')

    # X ticks: show year labels
    year_ticks = [i for i, m in enumerate(all_months) if m.endswith('-01')]
    year_labels = [m[:4] for i, m in enumerate(all_months) if m.endswith('-01')]
    ax.set_xticks(year_ticks[::2])
    ax.set_xticklabels(year_labels[::2], fontsize=7, rotation=45)

    # Title with diagnostics
    kge_str = f'KGE={kge_val:.2f}' if not np.isnan(kge_val) else 'KGE=n/a'
    const_str = ' [SAR CONST]' if is_const else ''
    ax.set_title(f'{name}\nA/P={res["ap_m"]:.0f}m  {kge_str}  land={res["n_land"]}{const_str}',
                 fontsize=8, pad=3,
                 color='darkred' if (is_const or (not np.isnan(kge_val) and kge_val < 0)) else 'black')

    ax.set_ylabel('Area (ha)', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.25)

    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')

# Hide unused axes
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('SAR (red) vs JRC optical (blue dashed) — biweekly→monthly area time series\n'
             'NOTE: Current SAR exports have n_land=0–7 (not WorldCover-retrained)',
             fontsize=11, y=1.01)
fig.tight_layout()
out1 = OUT_DIR / 'timeseries_SAR_vs_JRC_all.png'
fig.savefig(out1, dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved: {out1}')

# ── Figure 2: Annual KGE heatmap ─────────────────────────────────
all_years = sorted(set(yr for ann in all_annual_kge.values() for yr in ann))
# Sort reservoirs by overall median annual KGE (ascending)
res_order = sorted(all_annual_kge.keys(),
                   key=lambda n: np.nanmedian(list(all_annual_kge[n].values())) if all_annual_kge[n] else -99)

matrix = np.full((len(res_order), len(all_years)), np.nan)
for i, name in enumerate(res_order):
    for j, yr in enumerate(all_years):
        if yr in all_annual_kge[name]:
            matrix[i, j] = all_annual_kge[name][yr]

fig2, ax2 = plt.subplots(figsize=(14, 7))
cmap = plt.cm.RdYlGn
cmap.set_bad('lightgrey')
norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im = ax2.imshow(matrix, aspect='auto', cmap=cmap, norm=norm,
                interpolation='nearest')

# Annotate cells
for i in range(len(res_order)):
    for j in range(len(all_years)):
        v = matrix[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                     color='black' if abs(v) < 0.7 else 'white')

ax2.set_xticks(range(len(all_years)))
ax2.set_xticklabels(all_years, fontsize=9, rotation=45)
ax2.set_yticks(range(len(res_order)))
ax2.set_yticklabels([
    f'{n}  (land={reservoirs[n]["n_land"]}, A/P={reservoirs[n]["ap_m"]:.0f}m)'
    for n in res_order
], fontsize=8)
plt.colorbar(im, ax=ax2, label='Annual KGE (normalized within year)')
ax2.set_title('Annual KGE — SAR vs JRC optical area (per calendar year, ≥4 overlapping months)\n'
              'Grey = insufficient valid JRC months  |  Red=poor  Green=good',
              fontsize=10)
fig2.tight_layout()
out2 = OUT_DIR / 'annual_KGE_heatmap.png'
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')

# ── Save annual KGE CSV ───────────────────────────────────────────
csv_path = OUT_DIR / 'pilot_annual_kge.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['reservoir'] + all_years)
    for name in res_order:
        row = [name] + [f'{all_annual_kge[name].get(yr, ""):.4f}'
                        if all_annual_kge[name].get(yr) is not None
                        else '' for yr in all_years]
        w.writerow(row)
print(f'Saved: {csv_path}')

# ── Diagnostic summary ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Reservoir':<28}  {'A/P':>6}  {'n_land':>7}  {'Global KGE':>10}  {'Annual median':>13}")
print('─'*65)
for name in sorted(reservoirs.keys(), key=lambda n: reservoirs[n]['ap_m'], reverse=True):
    res = reservoirs[name]
    sar = res['sar_monthly']
    jrc = res['jrc_monthly']
    common = sorted(set(sar) & set(jrc))
    kge_val = np.nan
    if len(common) >= 12:
        s = np.array([sar[m] for m in common])
        o = np.array([jrc[m] for m in common])
        s_n = s/s.max() if s.max() > 0 else s
        o_n = o/o.max() if o.max() > 0 else o
        kge_val = kge(s_n, o_n)
    annual_vals = list(all_annual_kge.get(name, {}).values())
    ann_med = np.nanmedian(annual_vals) if annual_vals else np.nan
    k_str   = f'{kge_val:.3f}' if not np.isnan(kge_val) else '   n/a'
    a_str   = f'{ann_med:.3f}' if not np.isnan(ann_med) else '   n/a'
    flag = ' ← SAR constant' if np.std(list(sar.values())) < 0.01 * np.mean(list(sar.values())) else ''
    print(f'{name:<28}  {res["ap_m"]:>6.0f}  {res["n_land"]:>7}  {k_str:>10}  {a_str:>13}{flag}')
