"""
densify_poma_compare.py — side-by-side comparison for the densification prototype.

Rebuilds BOTH the original-10-mask and the densified-26-mask Poma Period-B DEM
through the identical build_dem_from_arrays code path (so any difference is
attributable only to mask count/density, not to algorithm-version drift versus
the production dem_Poma_B.tif), then plots them side by side plus a difference
map and an elevation histogram (a proxy for step/terrace structure: fewer,
taller spikes = coarser terracing; a smoother, more continuous histogram =
finer resolution).
"""
import pathlib, sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')

df = pd.read_csv(OUT_DIR / 'poma_densify_prototype_pairs.csv', parse_dates=['date'])
df = df.dropna(subset=['wl_m'])
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

# Cross-check each mask's own pixel-derived area against the trusted continuous
# SAR_area_Poma.csv series (Paper 1's SAR-area pipeline) using the SAME >60%
# rolling/neighbour-deviation spike threshold already used to clean that series
# (select_mask_dates.py::remove_outliers). Flags 2026-02-12 and 2026-03-26 --
# both ORIGINAL (already-in-production) masks whose windowed-export classification
# collapsed to ~73 ha (bounding box shrinks to a fraction of the reservoir) while
# the continuous series and the gauge-implied trend both show a much larger,
# smoothly-varying extent that date. Excluded from BOTH reconstructions below so
# the comparison isn't contaminated by a known, pre-existing mask defect.
cont = pd.read_csv(m.CONFIGS['Poma']['sar_csv'], parse_dates=['date']).sort_values('date')
cont = cont.groupby('date')['area_ha'].mean()


def mask_area_ha(date_str):
    with rasterio.open(MASK_DIR / f'mask_Poma_{date_str}.tif') as src:
        arr = src.read(1)
        return (arr == 1).sum() * src.res[0] * src.res[1] / 10000


bad_dates = []
for _, row in df.iterrows():
    mask_ha = mask_area_ha(row['date_str'])
    cont_val = cont.reindex([row['date']], method='nearest', tolerance=pd.Timedelta(days=1))
    if not len(cont_val.dropna()):
        continue
    cont_ha = float(cont_val.iloc[0])
    if abs(mask_ha - cont_ha) / cont_ha > 0.60:
        bad_dates.append(row['date_str'])
print(f'Outlier masks flagged (>60% deviation from continuous series): {bad_dates}')
df = df[~df['date_str'].isin(bad_dates)]


def build(sub):
    raw_arrays, wls = [], []
    for _, row in sub.iterrows():
        tif_path = MASK_DIR / f'mask_Poma_{row["date_str"]}.tif'
        with rasterio.open(tif_path) as src:
            raw_arrays.append(src.read(1).astype(np.float32))
    wls = sub['wl_m'].tolist()
    return m.build_dem_from_arrays(raw_arrays, wls)


dem_orig = build(df[~df['is_new']])
dem_dense = build(df)

print(f'Original  (n={(~df["is_new"]).sum()}):  '
      f'range {np.nanmin(dem_orig):.1f}-{np.nanmax(dem_orig):.1f} m')
print(f'Densified (n={len(df)}):  '
      f'range {np.nanmin(dem_dense):.1f}-{np.nanmax(dem_dense):.1f} m')

diff = dem_dense - dem_orig
print(f'Diff stats: mean={np.nanmean(diff):.3f}  std={np.nanstd(diff):.3f}  '
      f'|diff|>1m: {(np.abs(diff) > 1).sum()} px  |diff|>2m: {(np.abs(diff) > 2).sum()} px')

# Terracing proxy: unique elevation levels used, and histogram shape
u_orig = np.unique(np.round(dem_orig[np.isfinite(dem_orig)], 2))
u_dense = np.unique(np.round(dem_dense[np.isfinite(dense := dem_dense)], 2))
print(f'Distinct elevation values -- original: {len(u_orig)}, densified: {len(u_dense)}')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin = np.nanmin([dem_orig, dem_dense])
vmax = np.nanmax([dem_orig, dem_dense])

im0 = axes[0].imshow(dem_orig, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0].set_title(f'Original (10 masks)')
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='m')

im1 = axes[1].imshow(dem_dense, cmap='terrain', vmin=vmin, vmax=vmax)
axes[1].set_title(f'Densified (26 masks)')
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='m')

dmax = np.nanmax(np.abs(diff))
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
axes[2].set_title('Difference (densified - original)')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='m')

axes[3].hist(dem_orig[np.isfinite(dem_orig)], bins=80, alpha=0.5, label=f'Original (n=10)', color='tab:orange')
axes[3].hist(dem_dense[np.isfinite(dem_dense)], bins=80, alpha=0.5, label=f'Densified (n=26)', color='tab:blue')
axes[3].set_xlabel('Elevation (m)'); axes[3].set_ylabel('Pixel count')
axes[3].set_title('Elevation histogram (terracing proxy)')
axes[3].legend()

for ax in axes[:3]:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out_png = OUT_DIR / 'poma_densify_compare.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
