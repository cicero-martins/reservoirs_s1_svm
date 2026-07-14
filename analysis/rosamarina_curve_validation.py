"""
rosamarina_curve_validation.py  (E4 — Rosamarina 2nd independent reference)

Rosamarina now has an updated centimetric area-volume curve from the official
2025 bathymetric survey, extracted from PDF by extract_rosamarina_curve.py
(validation_data/updated_curves/rosamarina_2025.csv: quota_m, vol_m3, area_m2).
Unlike Poma's curve it carries AREA directly, so no dV/dh derivation is needed.

Same three comparisons as poma_curve_validation.py, in the SAR-observable band
[floor_B+1, max_B]:
  1. Area-elevation A(h): DEM_B vs design (1960s) vs 2025 survey.
  2. Volume-elevation V(h): absolute design vs survey (official change) and
     relative-to-floor increments incl. DEM_B (SAR-comparable).
  3. Change interpretation: official design->survey ΔV and the SAR relative
     deficit/surplus vs each curve.

Rosamarina is a CORE reservoir (has DEM A/B + DAHITI + gauge); adding the 2025
survey curve makes it — with Garcia — one of the two most-validated reservoirs.

Outputs: analysis/schwatke_output/rosamarina_curve/
"""

import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from scipy.interpolate import interp1d

sys.stdout.reconfigure(encoding='utf-8')

REPO       = pathlib.Path('.')
DEM_B      = REPO / 'analysis' / 'schwatke_output' / 'dem_Rosamarina_B.tif'
OUT_DIR    = REPO / 'analysis' / 'schwatke_output' / 'rosamarina_curve'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_XLS = 'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Rosamarina.xls'
UPDATE_CSV = REPO / 'validation_data' / 'updated_curves' / 'rosamarina_2025.csv'

PIXEL_HA = 0.01

# ── Satellite DEM_B AEV (absolute elevation from gauge/DAHITI WL) ──────────────
with rasterio.open(DEM_B) as s:
    dem = s.read(1).astype(np.float64)
mask  = ~np.isnan(dem)
floor = float(np.nanmin(dem[mask]))
top   = float(np.nanmax(dem[mask]))
print(f"DEM_Rosamarina_B: floor {floor:.2f}  max {top:.2f} m  ({mask.sum()} px)")

levels = np.arange(floor, top + 1e-6, 0.5)

def dem_aev(elev, mask, levels, pixel_ha=PIXEL_HA):
    areas = np.array([np.sum((elev < h) & mask) * pixel_ha for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols

a_dem, v_dem_rel = dem_aev(dem, mask, levels)

# ── Design curve (Foglio1: col2=quota, col3=area_ha, col5=vol_Mm3) ────────────
d = pd.read_excel(DESIGN_XLS, sheet_name=0, header=None, engine='xlrd')[[2, 3, 5]]
d = d.apply(pd.to_numeric, errors='coerce').dropna()
d.columns = ['quota', 'area_ha', 'vol_Mm3']
d = d[d.quota > 80].sort_values('quota').reset_index(drop=True)
des_area = interp1d(d.quota, d.area_ha, bounds_error=False, fill_value='extrapolate')
des_vol  = interp1d(d.quota, d.vol_Mm3, bounds_error=False, fill_value='extrapolate')

# ── Updated 2025 survey curve (CSV: quota_m, vol_m3, area_m2) ──────────────────
u = pd.read_csv(UPDATE_CSV)
upd_area = interp1d(u.quota_m, u.area_m2 / 1e4, bounds_error=False, fill_value='extrapolate')  # ha
upd_vol  = interp1d(u.quota_m, u.vol_m3 / 1e6,  bounds_error=False, fill_value='extrapolate')  # Mm3
print(f"Updated 2025 curve: {len(u)} rows  quota {u.quota_m.min():.2f}-{u.quota_m.max():.2f}  "
      f"vol {u.vol_m3.min()/1e6:.2f}-{u.vol_m3.max()/1e6:.2f} Mm3")

# ── Sample all sources on the DEM levels ──────────────────────────────────────
a_des = des_area(levels);  v_des_abs = des_vol(levels)
a_upd = upd_area(levels);  v_upd_abs = upd_vol(levels)
v_des_rel = v_des_abs - float(des_vol(floor))
v_upd_rel = v_upd_abs - float(upd_vol(floor))

aev = pd.DataFrame({
    'level_m': np.round(levels, 2),
    'area_dem_ha': np.round(a_dem, 1), 'area_design_ha': np.round(a_des, 1),
    'area_updated_ha': np.round(a_upd, 1),
    'vol_dem_rel_Mm3': np.round(v_dem_rel, 4), 'vol_design_rel_Mm3': np.round(v_des_rel, 4),
    'vol_updated_rel_Mm3': np.round(v_upd_rel, 4),
    'vol_design_abs_Mm3': np.round(v_des_abs, 3), 'vol_updated_abs_Mm3': np.round(v_upd_abs, 3),
})
aev.to_csv(OUT_DIR / 'rosamarina_aev_comparison.csv', index=False)

# ── Change interpretation at reference levels (observable band) ────────────────
ref_levels = [l for l in (155.0, 160.0, 163.0, 165.0, top) if floor + 1.0 <= l <= top]
rows = []
for h in ref_levels:
    vda, vua = float(des_vol(h)), float(upd_vol(h))
    vdr = float(np.interp(h, levels, v_des_rel))
    vur = float(np.interp(h, levels, v_upd_rel))
    vmr = float(np.interp(h, levels, v_dem_rel))
    rows.append({
        'level_m': round(h, 2),
        'vol_design_abs_Mm3': round(vda, 2), 'vol_updated_abs_Mm3': round(vua, 2),
        'official_change_Mm3': round(vua - vda, 2),
        'official_change_pct': round((vua - vda) / vda * 100, 1) if vda else np.nan,
        'sar_minus_design_rel_Mm3': round(vmr - vdr, 2),
        'sar_minus_updated_rel_Mm3': round(vmr - vur, 2),
        'area_dem_ha': round(float(np.interp(h, levels, a_dem)), 1),
        'area_design_ha': round(float(des_area(h)), 1),
        'area_updated_ha': round(float(upd_area(h)), 1),
    })
chg = pd.DataFrame(rows)
chg.to_csv(OUT_DIR / 'rosamarina_change_summary.csv', index=False)

print("\n--- Official change (2025 survey vs design) & SAR relative deficit ---")
print(chg[['level_m','vol_design_abs_Mm3','vol_updated_abs_Mm3','official_change_Mm3',
           'official_change_pct','sar_minus_design_rel_Mm3','sar_minus_updated_rel_Mm3']].to_string(index=False))
t = chg.iloc[-1]
sign = 'LOSS' if t.official_change_Mm3 < 0 else 'gain'
print(f"\nAt {t.level_m:.1f} m: 2025 survey vs design = {t.official_change_Mm3:+.2f} Mm3 "
      f"({t.official_change_pct:+.1f}%) -> official {sign}.")
print(f"SAR DEM_B increment vs updated survey (rel. to floor): "
      f"{t.sar_minus_updated_rel_Mm3:+.2f} Mm3.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
fig.suptitle('Rosamarina — satellite DEM_B vs design (1960s) and 2025 survey curve',
             fontsize=12, fontweight='bold')
band = levels >= floor + 1.0

axA = fig.add_subplot(gs[0])
axA.plot(a_des[band], levels[band], 'k--', lw=1.5, label='Design (1960s)')
axA.plot(a_upd[band], levels[band], 'C2-', lw=2.0, label='2025 survey')
axA.plot(a_dem[band], levels[band], 'C0-', lw=2.0, label='Satellite DEM_B')
axA.set_xlabel('Area (ha)'); axA.set_ylabel('Water level (m ASL)')
axA.set_title('Area–elevation'); axA.grid(True, alpha=0.3); axA.legend(fontsize=8); axA.set_xlim(left=0)

axB = fig.add_subplot(gs[1])
axB.plot(v_des_abs, levels, 'k--', lw=1.5, label='Design (1960s)')
axB.plot(v_upd_abs, levels, 'C2-', lw=2.0, label='2025 survey')
axB.fill_betweenx(levels, v_upd_abs, v_des_abs, color='firebrick', alpha=0.15)
axB.set_xlabel('Absolute volume (Mm³)'); axB.set_ylabel('Water level (m ASL)')
axB.set_title('Official storage: design vs 2025\n(gap = capacity change)')
axB.grid(True, alpha=0.3); axB.legend(fontsize=8); axB.set_xlim(left=0)

axC = fig.add_subplot(gs[2])
axC.plot(v_des_rel[band], levels[band], 'k--', lw=1.5, label='Design (rel.)')
axC.plot(v_upd_rel[band], levels[band], 'C2-', lw=2.0, label='2025 survey (rel.)')
axC.plot(v_dem_rel[band], levels[band], 'C0-', lw=2.0, label='DEM_B (rel.)')
axC.set_xlabel('Volume above floor (Mm³)'); axC.set_ylabel('Water level (m ASL)')
axC.set_title('Storage increment (rel. to DEM floor)')
axC.grid(True, alpha=0.3); axC.legend(fontsize=8); axC.set_xlim(left=0)

fig.subplots_adjust(top=0.86)
out = OUT_DIR / 'rosamarina_curve_validation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
print("CSVs: rosamarina_aev_comparison.csv, rosamarina_change_summary.csv")
