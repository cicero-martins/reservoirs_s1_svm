"""
build_poma_swot_only_dem.py (2026-07-29)

Builds Poma's Period-B bathymetric DEM using ONLY SWOT water levels -- no
gauge anywhere in the chain, not even for calibration -- to test the paper's
core claim (full-remote-sensing reconstruction, gauge-free) at the level of
the SPATIAL DEM product itself, not just the aggregate volume (which
fullrs_wl_ladder.py already validates via an independent scalar hypsometric
fit per source). Uses the same 24 area-outlier-clean masks as the
gauge-based densified reconstruction.

Per-date water level comes from a power-law curve A=a(h-h0)^b FIT ON GENUINE
SAR-area/SWOT-WL coincident pairs (continuous SAR series matched to all ~75
raw SWOT observations across the mission, not just the 24 export-window
dates), then INVERTED using each mask's own pixel-derived area -- NOT from
nearest-neighbour snapping to the closest raw SWOT pass. The latter would
assign the SAME stale SWOT reading to several different mask dates whenever
SWOT's sparser revisit leaves more than one SAR acquisition inside the match
tolerance (e.g. 2026-04-19 through 2026-05-02 all snapping to one 193.23 m
pass here), which is a real level even though it isn't that specific date's
level. Curve-inversion instead uses the area actually observed that day,
consistent with how the gauge-free 'model' fallback already works elsewhere
in the pipeline, just calibrated against SWOT instead of the gauge.

Compares directly against dem_Poma_B_densified.tif (the gauge-based version)
via a difference map, and reports RMSE/bias between the two.
"""
import pathlib, sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Poma'
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
BAD_DATES = {'2026-02-12', '2026-03-26'}

cfg = m.CONFIGS[RES]
swot = m.load_swot_corrected(cfg, pathlib.Path('validation_data/SWOT/Poma_swot.csv'), RES)

# Continuous SAR area series (near-every-acquisition coverage) matched to
# EVERY raw SWOT observation (+/-3 days) -- genuine coincident pairs, used
# ONLY to fit the curve, independent of which masks were selected/exported.
cont_area = pd.read_csv(cfg['sar_csv'], parse_dates=['date']).sort_values('date')
cont_area = cont_area.groupby('date')['area_ha'].mean()

swot_pairs = []
for dt, wl in swot.items():
    near = cont_area[(cont_area.index >= dt - pd.Timedelta(days=3)) &
                     (cont_area.index <= dt + pd.Timedelta(days=3))]
    if len(near):
        idx = (near.index - dt).to_series().abs().values.argmin()
        swot_pairs.append({'wl_m': wl, 'area_ha': float(near.iloc[idx])})
swot_pairs = pd.DataFrame(swot_pairs)
print(f'{len(swot_pairs)} genuine SAR-area/SWOT-WL coincident pairs '
      f'(of {len(swot)} raw SWOT observations)')

a, h0, b = m.fit_hyps_model(swot_pairs, cfg['h0_bound_lo'])
print(f'SWOT-calibrated model: A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')

df = pd.read_csv(OUT_DIR / 'poma_densify_prototype_pairs.csv', parse_dates=['date'])
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
df = df[~df['date_str'].isin(BAD_DATES)].copy()

df['swot_wl'] = df['area_ha'].apply(lambda area: m.invert_power_law(area, a, h0, b))
df = df.dropna(subset=['swot_wl'])
print(f'Building SWOT-only (curve-inverted) DEM from {len(df)} masks '
      f'(WL {df["swot_wl"].min():.1f}-{df["swot_wl"].max():.1f} m)')

raw_arrays, wls, meta = [], [], None
for _, row in df.iterrows():
    with rasterio.open(MASK_DIR / f'mask_{RES}_{row["date_str"]}.tif') as src:
        arr = src.read(1).astype(np.float32)
        if meta is None:
            meta = src.meta.copy()
    raw_arrays.append(arr)
    wls.append(row['swot_wl'])

dem_swot = m.build_dem_from_arrays(raw_arrays, wls)

out_meta = meta.copy()
out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
out_tif = OUT_DIR / f'dem_{RES}_B_swotonly.tif'
with rasterio.open(out_tif, 'w', **out_meta) as dst:
    dst.write(dem_swot[np.newaxis, :, :])
print(f'Saved {out_tif}  WL range {min(wls):.1f}-{max(wls):.1f} m  n_masks={len(wls)}')

with rasterio.open(OUT_DIR / f'dem_{RES}_B_densified.tif') as src:
    dem_gauge = src.read(1)

diff = dem_swot - dem_gauge
valid = np.isfinite(diff)
print(f'\nSWOT-only vs gauge-based DEM (n={valid.sum()} overlapping px):')
print(f'  bias (mean diff) = {np.nanmean(diff):.3f} m')
print(f'  RMSE             = {np.sqrt(np.nanmean(diff**2)):.3f} m')
print(f'  std              = {np.nanstd(diff):.3f} m')
print(f'  |diff|>1m: {(np.abs(diff[valid]) > 1).sum()} px ({100*(np.abs(diff[valid])>1).mean():.1f}%)')
print(f'  |diff|>2m: {(np.abs(diff[valid]) > 2).sum()} px ({100*(np.abs(diff[valid])>2).mean():.1f}%)')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
vmin = np.nanmin([dem_gauge, dem_swot]); vmax = np.nanmax([dem_gauge, dem_swot])
im0 = axes[0].imshow(dem_gauge, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0].set_title('Gauge-based (n=24)')
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='m')
im1 = axes[1].imshow(dem_swot, cmap='terrain', vmin=vmin, vmax=vmax)
axes[1].set_title(f'SWOT-only, curve-inverted (n={len(wls)})')
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='m')
dmax = np.nanmax(np.abs(diff))
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
axes[2].set_title('Difference (SWOT-only - gauge)')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='m')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out_png = OUT_DIR / f'poma_swotonly_vs_gauge.png'
plt.savefig(out_png, dpi=150)
print(f'\nSaved {out_png}')
