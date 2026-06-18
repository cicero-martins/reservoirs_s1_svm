"""Schwatke-style hypsometric reconstruction for Pozzillo reservoir.
Gauge data downloaded directly from AEGIS/CFD API (elementId=58946).
"""
import sys
import numpy as np
import pandas as pd
import xlrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = Path('analysis/schwatke_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DT = 5  # days

# ---------------------------------------------------------------------------
# 1. Gauge (from AEGIS CFD API — sub-hourly → daily mean)
# ---------------------------------------------------------------------------
WL_PATH = 'analysis/schwatke_output/gauge_downloads/pozzillo_wl.csv'
df_wl = pd.read_csv(WL_PATH)
df_wl['time'] = pd.to_datetime(df_wl['time'])
df_wl['wl_m'] = pd.to_numeric(df_wl['wl_m'], errors='coerce')
wl = (df_wl.dropna(subset=['wl_m'])
           .set_index('time')['wl_m']
           .resample('D').mean()
           .reset_index())
wl.columns = ['date', 'wl_m']
wl = wl.dropna().sort_values('date').reset_index(drop=True)
print(f"Gauge: {len(wl)} daily records  "
      f"{wl.date.min().date()} → {wl.date.max().date()}  "
      f"WL {wl.wl_m.min():.2f}–{wl.wl_m.max():.2f} m")

# ---------------------------------------------------------------------------
# 2. SAR surface areas (GEE app output, 2022–2026, overlapping with gauge)
# ---------------------------------------------------------------------------
area = pd.read_csv(
    'validation_data/statistics/area_statistics/ee-chart_pozzillo2022-26.csv')
area = area.rename(columns={'data': 'date'})
area['date'] = pd.to_datetime(area['date'])
area['A_m2'] = area['areaLago'] * 1e4
print(f"SAR: {len(area)} obs  "
      f"{area.date.min().date()} → {area.date.max().date()}  "
      f"area {area.areaLago.min():.1f}–{area.areaLago.max():.1f} ha")

# PlanetScope and SVM validation subsets (2024)
planet = pd.read_csv('validation_data/statistics/area_statistics/pozzilloPlanet.csv')
planet['date'] = pd.to_datetime(planet['data'], dayfirst=False, errors='coerce')
planet = planet.dropna(subset=['date'])
svm_val = pd.read_csv('validation_data/statistics/area_statistics/pozzilloSVM.csv')
svm_val['date'] = pd.to_datetime(svm_val['data'], dayfirst=False, errors='coerce')
svm_val = svm_val.dropna(subset=['date'])
print(f"PlanetScope: {len(planet)} obs  SVM validation: {len(svm_val)} obs")

# ---------------------------------------------------------------------------
# 3. Match pairs (±MAX_DT days)
# ---------------------------------------------------------------------------
pairs = []
for _, row in area.iterrows():
    delta = (wl['date'] - row['date']).dt.days.abs()
    idx = delta.idxmin()
    if delta[idx] <= MAX_DT:
        pairs.append({
            'date':    row['date'],
            'area_ha': row['areaLago'],
            'A_m2':    row['A_m2'],
            'wl_m':    wl.loc[idx, 'wl_m'],
            'dt_days': delta[idx],
        })
pairs_df = pd.DataFrame(pairs).dropna().reset_index(drop=True)
pairs_df['year'] = pd.to_datetime(pairs_df['date']).dt.year
print(f"Pairs (all): N={len(pairs_df)}")

# Sentinel-1C became operational in late 2024/early 2025 and introduces a systematic
# area overestimation (~+200 ha) relative to S1A due to different backscatter statistics.
# Restrict to 2022–2023 (confirmed S1A period) to avoid any transition-period contamination.
pairs_df = pairs_df[pairs_df['year'] <= 2023].reset_index(drop=True)
print(f"Pairs (S1A only, ≤2023): N={len(pairs_df)}  "
      f"area {pairs_df.area_ha.min():.1f}–{pairs_df.area_ha.max():.1f} ha  "
      f"WL {pairs_df.wl_m.min():.2f}–{pairs_df.wl_m.max():.2f} m")

# ---------------------------------------------------------------------------
# 4. Hypsometric fit  A = a * (h - h0)^b
# ---------------------------------------------------------------------------
def hyps_model(h, a, b, h0):
    return a * np.maximum(h - h0, 0.0) ** b

h_obs = pairs_df['wl_m'].values
A_obs = pairs_df['A_m2'].values

h0_upper = float(pairs_df['wl_m'].min()) - 0.01
h0_guess = h0_upper - 5.0

popt, _ = curve_fit(
    hyps_model, h_obs, A_obs,
    p0=[1e6, 1.5, h0_guess],
    bounds=([0, 0.2, 328.0], [1e9, 6.0, h0_upper]),
    maxfev=30000,
)
a_fit, b_fit, h0_fit = popt
A_pred = hyps_model(h_obs, *popt)
r2   = 1.0 - np.sum((A_obs - A_pred) ** 2) / np.sum((A_obs - A_obs.mean()) ** 2)
rmse = np.sqrt(np.mean((A_obs - A_pred) ** 2)) / 1e4
print(f"\nHypsometric fit: A = {a_fit:.2f} * (h - {h0_fit:.3f})^{b_fit:.4f}")
print(f"  R² = {r2:.4f}   RMSE = {rmse:.1f} ha")

# ---------------------------------------------------------------------------
# 5. Design AEV from Pozzillo.xls
#    col 2 = quote (m), col 3 = aree (km²), col 5 = volumi (Mm³)
# ---------------------------------------------------------------------------
wb = xlrd.open_workbook(
    'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Pozzillo.xls')
ws = wb.sheet_by_name('Foglio1')
aev_rows = []
for i in range(1, ws.nrows):
    r = ws.row_values(i)
    try:
        h_val = float(r[2])
        a_km2 = float(r[3])
        v_mm3 = float(r[5])
        if h_val > 100 and a_km2 >= 0 and v_mm3 >= 0:
            aev_rows.append({'h': h_val, 'A_km2': a_km2,
                             'A_m2': a_km2 * 1e6, 'V_Mm3': v_mm3})
    except (ValueError, TypeError):
        pass
aev = pd.DataFrame(aev_rows).sort_values('h').reset_index(drop=True)
print(f"\nDesign AEV: {len(aev)} rows  "
      f"h={aev.h.min():.1f}–{aev.h.max():.1f} m  "
      f"V={aev.V_Mm3.min():.2f}–{aev.V_Mm3.max():.2f} Mm³")

V_design = interp1d(aev['h'], aev['V_Mm3'],
                    kind='linear', bounds_error=False, fill_value='extrapolate')
A_design = interp1d(aev['h'], aev['A_m2'],
                    kind='linear', bounds_error=False, fill_value='extrapolate')

# ---------------------------------------------------------------------------
# 6. Integrate SAR volume, anchor to design AEV at h_ref
# ---------------------------------------------------------------------------
h_ref = float(pairs_df['wl_m'].min())
h_grid = np.arange(h0_fit + 0.01, aev['h'].max() + 1.0, 0.01)
A_grid = hyps_model(h_grid, *popt)
dh = np.diff(h_grid)
V_int = np.concatenate([[0.0],
                         np.cumsum(0.5 * (A_grid[:-1] + A_grid[1:]) * dh)])
idx_ref = int(np.searchsorted(h_grid, h_ref))
V_int += float(V_design(h_ref)) * 1e6 - V_int[idx_ref]
V_sar = interp1d(h_grid, V_int / 1e6,
                 kind='linear', bounds_error=False, fill_value=np.nan)

# ---------------------------------------------------------------------------
# 7. Level shift and sedimentation summary
# ---------------------------------------------------------------------------
h_design_of_A = interp1d(aev['A_m2'], aev['h'],
                          kind='linear', bounds_error=False, fill_value=np.nan)
shift_vals = [row['wl_m'] - float(h_design_of_A(row['A_m2']))
              for _, row in pairs_df.iterrows()
              if np.isfinite(float(h_design_of_A(row['A_m2'])))]
mean_shift = np.nanmean(shift_vals)
print(f"\nMean level shift: {mean_shift:+.2f} m")

h_full = aev['h'].max()
dV_full = float(V_sar(h_full)) - float(V_design(h_full))
print(f"Volume at h={h_full:.1f} m: "
      f"design={float(V_design(h_full)):.2f}  "
      f"SAR={float(V_sar(h_full)):.2f}  "
      f"ΔV={dV_full:+.2f} Mm³ ({dV_full/float(V_design(h_full))*100:+.1f}%)")

print(f"\n{'Level':>8}  {'V_design':>10}  {'V_SAR':>10}  {'ΔV':>10}  {'ΔV%':>8}")
for ht in [338, 340, 344, 348, 352, 356, 360, 364, 366.5]:
    if ht < h_ref:
        continue
    vd, vs = float(V_design(ht)), float(V_sar(ht))
    dv = vs - vd
    pct = dv / vd * 100 if vd > 0 else np.nan
    print(f"{ht:>8.1f}  {vd:>10.3f}  {vs:>10.3f}  {dv:>+10.3f}  {pct:>+7.1f}%")

# ---------------------------------------------------------------------------
# 8. Volume time series + AdB comparison
# ---------------------------------------------------------------------------
wl_valid = wl[wl['wl_m'].between(h_ref, aev['h'].max())].copy()
wl_valid['V_design_Mm3'] = V_design(wl_valid['wl_m'].values)
wl_valid['V_sar_Mm3']    = V_sar(wl_valid['wl_m'].values)

# Load official AdB volumes (monthly)
adib = pd.read_csv('validation_data/statistics/volume_statistics/pozzillo_adib.csv')
adib['date'] = pd.to_datetime(adib['date'], dayfirst=False, errors='coerce')
adib = adib.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
adib = adib.rename(columns={'volume_adib': 'V_adib'})

# Monthly gauge-derived volumes for comparison with AdB
wl_monthly = (wl_valid.set_index('date')[['V_design_Mm3','V_sar_Mm3']]
              .resample('MS').mean()
              .reset_index())
merged = pd.merge(wl_monthly, adib[['date','V_adib']], on='date', how='inner')
print(f"\n=== AdB comparison ({len(merged)} monthly pairs, "
      f"{merged.date.min().date()}–{merged.date.max().date()}) ===")
for col, lbl in [('V_design_Mm3','Design AEV'), ('V_sar_Mm3','SAR hypsometry')]:
    sub = merged[['V_adib', col]].dropna()
    obs, sim = sub['V_adib'].values, sub[col].values
    rmse = np.sqrt(np.mean((sim - obs)**2))
    bias = (sim - obs).mean()
    r2v  = 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)
    r_p  = np.corrcoef(obs, sim)[0, 1]
    print(f"  {lbl:20s}: R²={r2v:.4f}  r={r_p:.4f}  "
          f"RMSE={rmse:.2f} Mm³  bias={bias:+.2f} Mm³")

# ---------------------------------------------------------------------------
# 9. PlanetScope vs SVM validation (2024 subset)
# ---------------------------------------------------------------------------
# Match SVM to Planet by date (±5 days)
val_pairs = []
for _, pr in planet.iterrows():
    delta = (svm_val['date'] - pr['date']).dt.days.abs()
    idx = delta.idxmin()
    if delta[idx] <= 5:
        val_pairs.append({'date': pr['date'],
                          'area_planet': pr['area'],
                          'area_svm': svm_val.loc[idx, 'area']})
val_df = pd.DataFrame(val_pairs).dropna()
if len(val_df) >= 3:
    obs_p = val_df['area_planet'].values
    sim_s = val_df['area_svm'].values
    r2_val = 1 - np.sum((obs_p - sim_s)**2) / np.sum((obs_p - obs_p.mean())**2)
    rmse_val = np.sqrt(np.mean((sim_s - obs_p)**2))
    bias_val = (sim_s - obs_p).mean()
    print(f"\nSVM vs PlanetScope ({len(val_df)} pairs): "
          f"R²={r2_val:.4f}  RMSE={rmse_val:.1f} ha  bias={bias_val:+.1f} ha")

# ---------------------------------------------------------------------------
# 10. Figure — 3 panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
h_plot = np.linspace(h0_fit + 0.1, aev['h'].max(), 400)
# Load ALL pairs (including 2025-2026) to show the discontinuity
all_pairs = pd.read_csv('analysis/schwatke_output/pozzillo_hyps_pairs.csv')
all_pairs['date'] = pd.to_datetime(all_pairs['date'])

# Rebuild pairs from area file for display, split by year group
area_all = pd.read_csv(
    'validation_data/statistics/area_statistics/ee-chart_pozzillo2022-26.csv')
area_all = area_all.rename(columns={'data': 'date'})
area_all['date'] = pd.to_datetime(area_all['date'])
area_all['A_m2'] = area_all['areaLago'] * 1e4
area_all['year'] = area_all['date'].dt.year

def _build_pairs(subset):
    out = []
    for _, row in subset.iterrows():
        delta = (wl['date'] - row['date']).dt.days.abs()
        idx = delta.idxmin()
        if delta[idx] <= MAX_DT:
            out.append({'area_ha': row['areaLago'], 'wl_m': wl.loc[idx, 'wl_m']})
    return pd.DataFrame(out).dropna()

p_used    = _build_pairs(area_all[area_all['year'] <= 2023])   # fit pairs
p_excl_24 = _build_pairs(area_all[area_all['year'] == 2024])  # 2024 transition (excluded)
p25       = _build_pairs(area_all[area_all['year'] >= 2025])  # S1C 2025-26 (excluded)

ax.scatter(p_used['area_ha'], p_used['wl_m'],
           s=18, alpha=0.65, color='steelblue', zorder=3, label='S1A pairs used in fit (2022–2023)')
ax.scatter(p_excl_24['area_ha'], p_excl_24['wl_m'],
           s=18, alpha=0.5, color='goldenrod', zorder=3, marker='s',
           label='S1A 2024 (excluded — transition period)')
if len(p25):
    ax.scatter(p25['area_ha'], p25['wl_m'],
               s=18, alpha=0.5, color='tomato', zorder=3, marker='^',
               label='S1A+C pairs (2025–2026, excluded)')
ax.plot(hyps_model(h_plot, *popt) / 1e4, h_plot,
        'r-', lw=2, label=f'SAR fit S1A (R²={r2:.3f})')
ax.plot(aev['A_m2'] / 1e4, aev['h'], 'k--', lw=1.5, label='Design AEV')
ax.set_xlabel('Surface area (ha)')
ax.set_ylabel('Water level (m a.s.l.)')
ax.set_title('(a) Hypsometric curve — Pozzillo')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1]
h_range = np.linspace(h_ref, aev['h'].max(), 400)
ax.plot(V_sar(h_range), h_range, 'r-', lw=2, label='SAR-derived')
ax.plot(V_design(h_range), h_range, 'k--', lw=1.5, label='Design AEV')
ax.axhline(h_ref, color='gray', lw=0.8, ls=':', label=f'h_ref={h_ref:.1f} m')
ax.set_xlabel('Volume (Mm³)')
ax.set_ylabel('Water level (m a.s.l.)')
ax.set_title('(b) Volume–level curves')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(adib['date'], adib['V_adib'],
        'o-', ms=3, lw=1.0, color='gray', alpha=0.5, label='AdB official (monthly)')
ax.plot(wl_valid['date'], wl_valid['V_design_Mm3'],
        'k--', lw=1.2, alpha=0.85, label='Design AEV (gauge)')
ax.plot(wl_valid['date'], wl_valid['V_sar_Mm3'],
        'r-', lw=1.6, label='SAR hypsometry (gauge, S1A)')
ax.axvspan(wl_valid['date'].min(), wl_valid['date'].max(),
           alpha=0.06, color='steelblue')
ax.set_xlabel('Date')
ax.set_ylabel('Volume (Mm³)')
ax.set_title('(c) Storage time series vs AdB official')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

fig.suptitle('Pozzillo — Schwatke MVP hypsometric reconstruction', fontsize=12)
fig.tight_layout()
out_path = OUT_DIR / 'pozzillo_schwatke_mvp.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure: {out_path}")
plt.close(fig)

pairs_df.to_csv(OUT_DIR / 'pozzillo_hyps_pairs.csv', index=False)
wl_valid[['date', 'wl_m', 'V_design_Mm3', 'V_sar_Mm3']].to_csv(
    OUT_DIR / 'pozzillo_volume_timeseries.csv', index=False)
print("CSVs saved.")
