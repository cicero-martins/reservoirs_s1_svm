"""
densify_rosamarina_compare.py — side-by-side comparison, reservoir 2.

Builds the original (8 masks -- 2 of the production 10 have no gauge/SWOT
match within tolerance, a pre-existing production gap, not something this
prototype introduced) vs densified (34 masks, mixed gauge+SWOT) Rosamarina
Period-B DEM through the identical build_dem_from_arrays code path.
"""
import pathlib, sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Rosamarina'
MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')

df = pd.read_csv(OUT_DIR / 'rosamarina_densify_prototype_pairs.csv')
df = df.dropna(subset=['wl_m'])


def build(sub):
    raw_arrays, wls = [], []
    for _, row in sub.iterrows():
        with rasterio.open(MASK_DIR / f'mask_{RES}_{row["date"]}.tif') as src:
            raw_arrays.append(src.read(1).astype(np.float32))
        wls.append(row['wl_m'])
    return m.build_dem_from_arrays(raw_arrays, wls)


dem_orig = build(df[~df['is_new']])
dem_dense = build(df)

print(f'Original  (n={(~df["is_new"]).sum()}):  range {np.nanmin(dem_orig):.1f}-{np.nanmax(dem_orig):.1f} m')
print(f'Densified (n={len(df)}):  range {np.nanmin(dem_dense):.1f}-{np.nanmax(dem_dense):.1f} m')

diff = dem_dense - dem_orig
print(f'Diff stats: mean={np.nanmean(diff):.3f}  std={np.nanstd(diff):.3f}  '
      f'|diff|>1m: {(np.abs(diff) > 1).sum()} px  |diff|>2m: {(np.abs(diff) > 2).sum()} px')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin = np.nanmin([dem_orig, dem_dense]); vmax = np.nanmax([dem_orig, dem_dense])

im0 = axes[0].imshow(dem_orig, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0].set_title('Original (8 masks)')
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='m')

im1 = axes[1].imshow(dem_dense, cmap='terrain', vmin=vmin, vmax=vmax)
axes[1].set_title('Densified (34 masks, gauge+SWOT)')
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='m')

dmax = np.nanmax(np.abs(diff))
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
axes[2].set_title('Difference (densified - original)')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='m')

axes[3].hist(dem_orig[np.isfinite(dem_orig)], bins=80, alpha=0.5, label='Original (n=8)', color='tab:orange')
axes[3].hist(dem_dense[np.isfinite(dem_dense)], bins=80, alpha=0.5, label='Densified (n=34)', color='tab:blue')
axes[3].set_xlabel('Elevation (m)'); axes[3].set_ylabel('Pixel count')
axes[3].set_title('Elevation histogram (terracing proxy)')
axes[3].legend()

for ax in axes[:3]:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out_png = OUT_DIR / 'rosamarina_densify_compare.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')

out_meta = None
with rasterio.open(MASK_DIR / f'mask_{RES}_{df.iloc[0]["date"]}.tif') as src:
    out_meta = src.meta.copy()
out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
out_tif = OUT_DIR / f'dem_{RES}_B_densified.tif'
with rasterio.open(out_tif, 'w', **out_meta) as dst:
    dst.write(dem_dense[np.newaxis, :, :])
print(f'Saved {out_tif}')
