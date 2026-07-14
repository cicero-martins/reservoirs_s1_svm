"""
garcia_validation_deep.py  (E4 — deepened field ground-truth validation)

Extends garcia_survey_comparison.py with three deeper analyses of the Garcia
satellite DEM_B (2022-2026) against the Dec-2025 echosounder survey:

  1. Error by elevation band  -- bias / RMSE / MAE per 1 m stratum of the
     SAR-observable range, so we see WHERE the DEM is accurate vs biased
     (not just one aggregate number).
  2. AEV error across the whole curve  -- area & volume error (DEM - survey)
     at every 0.5 m level, plus integrated volume error and % error.
  3. Volume-change (sedimentation) validation  -- does the SAR DEM capture the
     storage LOSS that the field survey reveals vs the 1960s design curve?
     ΔV_true = design-survey  vs  ΔV_sar = design-DEM  ->  capture ratio.

Reuses the CACHED gridded survey rasters produced by garcia_survey_comparison.py
(survey_dem_Garcia.tif = full sonar+terrain; survey_terrain_Garcia.tif = exposed
shore only) so no re-gridding of the 194k survey points is needed. Run
garcia_survey_comparison.py first if those caches are missing.

Outputs: analysis/schwatke_output/garcia_survey/
  garcia_error_by_band.csv, garcia_aev_error.csv, garcia_volume_change.csv,
  garcia_validation_deep.png
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from scipy.interpolate import interp1d

sys.stdout.reconfigure(encoding='utf-8')

REPO      = pathlib.Path('.')
OUT_DIR   = REPO / 'analysis' / 'schwatke_output' / 'garcia_survey'
DEM_B     = REPO / 'analysis' / 'schwatke_output' / 'dem_Garcia_B.tif'
SURVEY_DEM  = OUT_DIR / 'survey_dem_Garcia.tif'       # full (sonar + terrain), dist-filtered
SURVEY_TERR = OUT_DIR / 'survey_terrain_Garcia.tif'   # exposed shore only, dist-filtered
DESIGN_XLS  = 'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Garcia.xls'

PIXEL_HA  = 0.01     # 10 m x 10 m
BAND_M    = 1.0      # elevation-band width for stratified error

for p in (DEM_B, SURVEY_DEM, SURVEY_TERR):
    if not p.exists():
        sys.exit(f"Missing {p}. Run analysis/garcia_survey_comparison.py first "
                 f"(it grids the survey points and writes the cached rasters).")

# ── Load rasters (all share the DEM_B grid) ────────────────────────────────────
def read(path):
    with rasterio.open(path) as s:
        return s.read(1).astype(np.float64)

dem_b    = read(DEM_B)
srv_full = read(SURVEY_DEM)     # for AEV (has the deep sonar zone)
srv_terr = read(SURVEY_TERR)    # for pixel/per-band error (exposed shore)

lake_mask = ~np.isnan(dem_b)
floor_B   = float(np.nanmin(dem_b[lake_mask]))
max_B     = float(np.nanmax(dem_b[lake_mask]))
print(f"DEM_B: floor {floor_B:.2f}  max {max_B:.2f} m  ({lake_mask.sum()} px)")

# ── Design AEV curve (col2=quota, col3=area_km2, col4=vol_Mm3) ─────────────────
curve = pd.read_excel(DESIGN_XLS, sheet_name=0, header=None)[[2, 3, 4]]
curve = curve.apply(pd.to_numeric, errors='coerce').dropna()
curve.columns = ['quota', 'area_km2', 'vol_Mm3']
curve = curve.sort_values('quota').reset_index(drop=True)
h2a_design = interp1d(curve.quota, curve.area_km2 * 100, bounds_error=False, fill_value='extrapolate')
h2v_design = interp1d(curve.quota, curve.vol_Mm3,        bounds_error=False, fill_value='extrapolate')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ERROR BY ELEVATION BAND  (pixel comparison stratified by survey elevation)
# ═══════════════════════════════════════════════════════════════════════════════
# SAR-observable range: dem_b > floor+1 (exclude floor-assigned core),
# survey terrain within [floor_B, max_B].
shallow = (lake_mask & ~np.isnan(srv_terr) &
           (dem_b > floor_B + 1.0) &
           (srv_terr >= floor_B) & (srv_terr <= max_B))

d_sh = dem_b[shallow]
s_sh = srv_terr[shallow]
diff = d_sh - s_sh   # satellite - survey  (negative = satellite lower)

# overall (sanity check vs recorded RMSE 2.44)
rmse_all = float(np.sqrt(np.mean(diff**2)))
print(f"\nOverall (SAR-observable): n={shallow.sum()}  bias={diff.mean():+.3f}  "
      f"RMSE={rmse_all:.3f}  MAE={np.abs(diff).mean():.3f} m")

lo = int(np.floor(floor_B + 1.0))
hi = int(np.ceil(max_B))
band_rows = []
for b0 in np.arange(lo, hi, BAND_M):
    b1 = b0 + BAND_M
    m = (s_sh >= b0) & (s_sh < b1)
    if m.sum() < 20:
        continue
    dd = diff[m]
    band_rows.append({
        'band_lo': round(float(b0), 1), 'band_hi': round(float(b1), 1),
        'n': int(m.sum()),
        'bias_m': round(float(dd.mean()), 3),
        'rmse_m': round(float(np.sqrt(np.mean(dd**2))), 3),
        'mae_m':  round(float(np.abs(dd).mean()), 3),
        'std_m':  round(float(dd.std()), 3),
    })
band_df = pd.DataFrame(band_rows)
band_df.to_csv(OUT_DIR / 'garcia_error_by_band.csv', index=False)
print("\n--- Error by 1 m elevation band (survey elevation) ---")
print(band_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. AEV ERROR ACROSS THE CURVE
# ═══════════════════════════════════════════════════════════════════════════════
levels = np.arange(floor_B, max_B + 1e-6, 0.5)

def compute_aev(elev_grid, mask, levels, pixel_ha=PIXEL_HA):
    """area(h)=pixels below h (ha); volume via trapezoid rel. to first level (Mm3)."""
    areas = np.array([np.sum((elev_grid < h) & mask) * pixel_ha for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols

a_dem, v_dem = compute_aev(dem_b,    lake_mask, levels)
a_srv, v_srv = compute_aev(srv_full, lake_mask, levels)
a_des = h2a_design(levels)
v_des_abs = h2v_design(levels)
v_des = v_des_abs - float(h2v_design(floor_B))   # relative to floor_B like dem/srv

aev = pd.DataFrame({
    'level_m': np.round(levels, 2),
    'area_design_ha': np.round(a_des, 1),
    'area_survey_ha': np.round(a_srv, 1),
    'area_dem_ha':    np.round(a_dem, 1),
    'vol_design_Mm3': np.round(v_des, 4),
    'vol_survey_Mm3': np.round(v_srv, 4),
    'vol_dem_Mm3':    np.round(v_dem, 4),
})
aev['area_err_dem_ha']  = np.round(a_dem - a_srv, 1)   # DEM - survey
aev['vol_err_dem_Mm3']  = np.round(v_dem - v_srv, 4)
vpct = np.divide(v_dem - v_srv, v_srv, out=np.full_like(v_srv, np.nan), where=v_srv > 0) * 100
aev['vol_err_dem_pct']  = np.round(vpct, 1)
aev.to_csv(OUT_DIR / 'garcia_aev_error.csv', index=False)

# integrated (volume-weighted) errors over the observable band
obs = levels >= floor_B + 1.0
vol_mae = float(np.mean(np.abs((v_dem - v_srv)[obs])))
area_mae = float(np.mean(np.abs((a_dem - a_srv)[obs])))
print(f"\n--- AEV error (observable band {floor_B+1:.1f}-{max_B:.1f} m) ---")
print(f"area MAE (DEM vs survey): {area_mae:.1f} ha   vol MAE: {vol_mae:.3f} Mm3")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VOLUME-CHANGE (SEDIMENTATION) VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
# Does DEM_B capture the storage loss the survey reveals vs the design curve?
# All volumes relative to floor_B; compare at reference water levels.
ref_levels = [l for l in (185.0, 187.0, 188.0, max_B) if floor_B < l <= max_B]
chg_rows = []
for h in ref_levels:
    vD = float(np.interp(h, levels, v_des))
    vS = float(np.interp(h, levels, v_srv))
    vM = float(np.interp(h, levels, v_dem))
    aD = float(np.interp(h, levels, a_des))
    aS = float(np.interp(h, levels, a_srv))
    aM = float(np.interp(h, levels, a_dem))
    dV_true = vD - vS          # design -> survey  (real loss)
    dV_sar  = vD - vM          # design -> DEM     (SAR-detected loss)
    chg_rows.append({
        'level_m': round(h, 2),
        'vol_design_Mm3': round(vD, 3), 'vol_survey_Mm3': round(vS, 3), 'vol_dem_Mm3': round(vM, 3),
        'loss_true_Mm3': round(dV_true, 3), 'loss_true_pct': round(dV_true / vD * 100, 1) if vD else np.nan,
        'loss_sar_Mm3':  round(dV_sar, 3),  'loss_sar_pct':  round(dV_sar / vD * 100, 1) if vD else np.nan,
        'capture_ratio': round(dV_sar / dV_true, 2) if dV_true else np.nan,
        'area_design_ha': round(aD, 1), 'area_survey_ha': round(aS, 1), 'area_dem_ha': round(aM, 1),
    })
chg = pd.DataFrame(chg_rows)
chg.to_csv(OUT_DIR / 'garcia_volume_change.csv', index=False)
print("\n--- Volume-change (sedimentation) validation ---")
print(chg[['level_m','vol_design_Mm3','vol_survey_Mm3','vol_dem_Mm3',
           'loss_true_Mm3','loss_sar_Mm3','capture_ratio']].to_string(index=False))
top = chg.iloc[-1]
print(f"\nAt {top.level_m:.1f} m: survey reveals {top.loss_true_Mm3:.2f} Mm3 loss "
      f"({top.loss_true_pct:.0f}% of design); SAR DEM detects {top.loss_sar_Mm3:.2f} Mm3 "
      f"({top.loss_sar_pct:.0f}%) -> captures {top.capture_ratio*100:.0f}% of the true loss.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
fig.suptitle('Garcia — deepened validation: satellite DEM_B vs Dec-2025 echo-sounder survey',
             fontsize=12, fontweight='bold')

# Panel A: error by elevation band
axA = fig.add_subplot(gs[0])
axA.axvline(0, color='gray', lw=0.8, ls=':')
axA.errorbar(band_df.bias_m, (band_df.band_lo + band_df.band_hi) / 2,
             xerr=band_df.rmse_m, fmt='o-', color='C3', capsize=3, lw=1.5, label='bias ± RMSE')
axA.set_xlabel('DEM − survey (m)   (negative = satellite lower)')
axA.set_ylabel('Elevation band (m ASL)')
axA.set_title('Error stratified by elevation')
axA.grid(True, alpha=0.3); axA.legend(fontsize=8)

# Panel B: AEV volume curves
axB = fig.add_subplot(gs[1])
axB.plot(v_des, levels, 'k--', lw=1.5, label='Design (1960s)')
axB.plot(v_srv, levels, 'C2-', lw=2.0, label='Survey Dec-2025')
axB.plot(v_dem, levels, 'C0-', lw=2.0, label='Satellite DEM_B')
axB.set_xlabel('Volume above floor (Mm³)')
axB.set_ylabel('Water level (m ASL)')
axB.set_title('Volume–elevation (AEV)')
axB.grid(True, alpha=0.3); axB.legend(fontsize=8); axB.set_xlim(left=0)

# Panel C: storage-loss capture bars
axC = fig.add_subplot(gs[2])
xs = np.arange(len(chg)); w = 0.38
axC.bar(xs - w/2, chg.loss_true_Mm3, w, color='C2', label='True loss (design−survey)')
axC.bar(xs + w/2, chg.loss_sar_Mm3,  w, color='C0', label='SAR loss (design−DEM)')
for i, r in chg.iterrows():
    axC.text(i, max(r.loss_true_Mm3, r.loss_sar_Mm3) + 0.1,
             f"{r.capture_ratio*100:.0f}%", ha='center', fontsize=8)
axC.set_xticks(xs); axC.set_xticklabels([f"{l:.0f} m" for l in chg.level_m])
axC.set_ylabel('Storage loss vs design (Mm³)')
axC.set_title('Sedimentation-loss capture')
axC.grid(True, alpha=0.3, axis='y'); axC.legend(fontsize=8)

fig.subplots_adjust(top=0.88)
out = OUT_DIR / 'garcia_validation_deep.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
print("CSVs: garcia_error_by_band.csv, garcia_aev_error.csv, garcia_volume_change.csv")
