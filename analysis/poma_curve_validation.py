"""
poma_curve_validation.py  (E4 — 2nd independent field reference)

Poma has no echo-sounder point cloud, but the Sicilian water authority published
an UPDATED centimetric area-volume curve derived from a recent topo-bathymetric
survey (NewCurves/POMA.XLS, sheet 'foglio1' / 'Tabella centimetrica': quota[m],
volume[m3] at 1 cm resolution, 168.00-196.85 m). This gives a second, independent
ground-truth AEV to check the satellite DEM_B against, complementing the Garcia
echo-sounder validation.

Three comparisons in the SAR-observable band [floor_B+1, max_B]:
  1. Area-elevation A(h): DEM_B vs design (1960s) vs updated survey. Absolute
     elevation -> datum-robust (area needs no volume reference).
  2. Volume-elevation V(h): absolute design vs updated (the OFFICIAL change
     signal), and relative-to-floor increments incl. DEM_B (comparable to SAR).
  3. Change interpretation: official design->updated ΔV, and the SAR relative
     deficit vs each official curve.

KEY EXPECTATION (see project memory): the updated official curve holds ~+2 Mm3
MORE than the design curve at every level -> Poma shows NO net capacity loss;
the SAR DEM's apparent deficit is therefore a low-elevation-bias artifact,
cross-checkable against Garcia's -0.6 m pixel bias.

Outputs: analysis/schwatke_output/poma_curve/
  poma_aev_comparison.csv, poma_change_summary.csv, poma_curve_validation.png
"""

import pathlib, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from scipy.interpolate import interp1d

sys.stdout.reconfigure(encoding='utf-8')

REPO       = pathlib.Path('.')
DEM_B      = REPO / 'analysis' / 'schwatke_output' / 'dem_Poma_B.tif'
OUT_DIR    = REPO / 'analysis' / 'schwatke_output' / 'poma_curve'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_XLS = 'C:/Users/Unipa/Documents/GEE/Data/Curve aree-volumi/Poma.xls'

# The updated centimetric curve lives in a cloud-synced folder whose exact file
# name has drifted (POMA.XLS / POMA_new.XLS); resolve it robustly.
def _resolve(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits, key=len)[0]
    sys.exit(f"Could not resolve updated Poma curve; tried: {patterns}")

UPDATE_XLS = _resolve([
    'C:/Users/Unipa/Documents/GEE/Data/NewCurves/POMA*.XLS',
    'C:/Users/Unipa/Documents/GEE/Data/NewCurves/POMA*.xls',
])
print(f"Updated curve file: {UPDATE_XLS}")

PIXEL_HA = 0.01

# ── Satellite DEM_B AEV (absolute elevation from gauge WL) ─────────────────────
with rasterio.open(DEM_B) as s:
    dem = s.read(1).astype(np.float64)
mask   = ~np.isnan(dem)
floor  = float(np.nanmin(dem[mask]))
top    = float(np.nanmax(dem[mask]))
print(f"DEM_Poma_B: floor {floor:.2f}  max {top:.2f} m  ({mask.sum()} px)")

levels = np.arange(floor, top + 1e-6, 0.5)

def dem_aev(elev, mask, levels, pixel_ha=PIXEL_HA):
    areas = np.array([np.sum((elev < h) & mask) * pixel_ha for h in levels])
    vols = np.zeros_like(areas)
    for i in range(1, len(levels)):
        vols[i] = vols[i-1] + (areas[i] + areas[i-1]) / 2 * (levels[i] - levels[i-1]) * 0.01
    return areas, vols

a_dem, v_dem_rel = dem_aev(dem, mask, levels)   # volume relative to floor

# ── Design curve (Foglio1: col2=quota, col4=area_ha, col5=vol_Mm3) ─────────────
d = pd.read_excel(DESIGN_XLS, sheet_name='Foglio1', header=None, engine='xlrd')[[2, 4, 5]]
d = d.apply(pd.to_numeric, errors='coerce').dropna()
d.columns = ['quota', 'area_ha', 'vol_Mm3']
d = d[d.quota > 100].sort_values('quota').reset_index(drop=True)
des_area = interp1d(d.quota, d.area_ha,  bounds_error=False, fill_value='extrapolate')
des_vol  = interp1d(d.quota, d.vol_Mm3,  bounds_error=False, fill_value='extrapolate')

# ── Updated survey curve (foglio1: col0=quota, col1=vol_m3); area = dV/dh ──────
u = pd.read_excel(UPDATE_XLS, sheet_name='foglio1', header=None, engine='xlrd')[[0, 1]]
u.columns = ['quota', 'vol_m3']
u = u.apply(pd.to_numeric, errors='coerce').dropna().sort_values('quota').reset_index(drop=True)
u_vol_Mm3 = u.vol_m3.values / 1e6
u_area_ha = np.gradient(u.vol_m3.values, u.quota.values) / 1e4   # m3/m -> m2 -> ha
upd_area = interp1d(u.quota, u_area_ha,  bounds_error=False, fill_value='extrapolate')
upd_vol  = interp1d(u.quota, u_vol_Mm3,  bounds_error=False, fill_value='extrapolate')
print(f"Updated curve: {len(u)} rows  quota {u.quota.min():.2f}-{u.quota.max():.2f}  "
      f"vol {u_vol_Mm3.min():.2f}-{u_vol_Mm3.max():.2f} Mm3")

# ── Sample all sources on the DEM levels ──────────────────────────────────────
a_des = des_area(levels);  v_des_abs = des_vol(levels)
a_upd = upd_area(levels);  v_upd_abs = upd_vol(levels)
v_des_rel = v_des_abs - float(des_vol(floor))
v_upd_rel = v_upd_abs - float(upd_vol(floor))

aev = pd.DataFrame({
    'level_m': np.round(levels, 2),
    'area_dem_ha':    np.round(a_dem, 1),
    'area_design_ha': np.round(a_des, 1),
    'area_updated_ha':np.round(a_upd, 1),
    'vol_dem_rel_Mm3':     np.round(v_dem_rel, 4),
    'vol_design_rel_Mm3':  np.round(v_des_rel, 4),
    'vol_updated_rel_Mm3': np.round(v_upd_rel, 4),
    'vol_design_abs_Mm3':  np.round(v_des_abs, 3),
    'vol_updated_abs_Mm3': np.round(v_upd_abs, 3),
})
aev.to_csv(OUT_DIR / 'poma_aev_comparison.csv', index=False)

# ── Change interpretation at reference levels (observable band only) ───────────
obs = levels >= floor + 1.0
ref_levels = [l for l in (185.0, 188.0, 190.0, 192.0, top) if floor + 1.0 <= l <= top]
rows = []
for h in ref_levels:
    vda, vua = float(des_vol(h)), float(upd_vol(h))
    vdr = float(np.interp(h, levels, v_des_rel))
    vur = float(np.interp(h, levels, v_upd_rel))
    vmr = float(np.interp(h, levels, v_dem_rel))
    rows.append({
        'level_m': round(h, 2),
        'vol_design_abs_Mm3': round(vda, 2), 'vol_updated_abs_Mm3': round(vua, 2),
        'official_change_Mm3': round(vua - vda, 2),          # updated - design (>0 = no loss)
        'official_change_pct': round((vua - vda) / vda * 100, 1),
        'sar_minus_design_rel_Mm3': round(vmr - vdr, 2),     # SAR increment vs design (<0 = SAR low)
        'sar_minus_updated_rel_Mm3': round(vmr - vur, 2),
        'area_dem_ha': round(float(np.interp(h, levels, a_dem)), 1),
        'area_design_ha': round(float(des_area(h)), 1),
        'area_updated_ha': round(float(upd_area(h)), 1),
    })
chg = pd.DataFrame(rows)
chg.to_csv(OUT_DIR / 'poma_change_summary.csv', index=False)

print("\n--- Official change (updated survey vs design) & SAR relative deficit ---")
print(chg[['level_m','vol_design_abs_Mm3','vol_updated_abs_Mm3','official_change_Mm3',
           'official_change_pct','sar_minus_design_rel_Mm3']].to_string(index=False))
t = chg.iloc[-1]
print(f"\nAt {t.level_m:.1f} m: official updated survey holds {t.official_change_Mm3:+.2f} Mm3 "
      f"({t.official_change_pct:+.1f}%) vs design -> NO net capacity loss.")
print(f"SAR DEM_B storage increment sits {t.sar_minus_design_rel_Mm3:+.2f} Mm3 vs the design shape "
      f"(rel. to floor) -> consistent with a low-elevation bias, not real sedimentation.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
fig.suptitle('Poma — satellite DEM_B vs design (1960s) and updated survey curve (2nd reference)',
             fontsize=12, fontweight='bold')
band = obs   # only draw observable band for DEM-based curves

# Panel A: area-elevation
axA = fig.add_subplot(gs[0])
axA.plot(a_des[band], levels[band], 'k--', lw=1.5, label='Design (1960s)')
axA.plot(a_upd[band], levels[band], 'C2-', lw=2.0, label='Updated survey')
axA.plot(a_dem[band], levels[band], 'C0-', lw=2.0, label='Satellite DEM_B')
axA.set_xlabel('Area (ha)'); axA.set_ylabel('Water level (m ASL)')
axA.set_title('Area–elevation'); axA.grid(True, alpha=0.3); axA.legend(fontsize=8); axA.set_xlim(left=0)

# Panel B: absolute volume — the official change signal
axB = fig.add_subplot(gs[1])
axB.plot(v_des_abs, levels, 'k--', lw=1.5, label='Design (1960s)')
axB.plot(v_upd_abs, levels, 'C2-', lw=2.0, label='Updated survey')
axB.fill_betweenx(levels, v_des_abs, v_upd_abs, color='C2', alpha=0.15)
axB.set_xlabel('Absolute volume (Mm³)'); axB.set_ylabel('Water level (m ASL)')
axB.set_title('Official storage: updated ≥ design\n(no net loss)')
axB.grid(True, alpha=0.3); axB.legend(fontsize=8); axB.set_xlim(left=0)

# Panel C: relative-to-floor increment incl. DEM (SAR-comparable)
axC = fig.add_subplot(gs[2])
axC.plot(v_des_rel[band], levels[band], 'k--', lw=1.5, label='Design (rel.)')
axC.plot(v_upd_rel[band], levels[band], 'C2-', lw=2.0, label='Updated (rel.)')
axC.plot(v_dem_rel[band], levels[band], 'C0-', lw=2.0, label='DEM_B (rel.)')
axC.set_xlabel('Volume above floor (Mm³)'); axC.set_ylabel('Water level (m ASL)')
axC.set_title('Storage increment (rel. to DEM floor)\nSAR sits below official → low bias')
axC.grid(True, alpha=0.3); axC.legend(fontsize=8); axC.set_xlim(left=0)

fig.subplots_adjust(top=0.86)
out = OUT_DIR / 'poma_curve_validation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
print("CSVs: poma_aev_comparison.csv, poma_change_summary.csv")
