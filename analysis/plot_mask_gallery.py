"""
plot_mask_gallery.py (2026-07-31)

Visual QA: for a reservoir, plot every mask in its current usable pool (same
pool build_frs_dem.py stacks into the DEM -- densified pool where one exists,
else mask_wl_pairs' production dates) as a small thumbnail, sorted by
assigned water level, so a spurious/misclassified mask (e.g. wind-roughening,
a bad NDWI threshold) is visible directly rather than only inferred from an
area-vs-level monotonicity number. Triggered by finding Nicoletti's
2026-03-14 mask breaking monotonicity in the production calibration set
itself (not just the densified pool), unflagged by the existing
area-outlier check.

Run:
    python analysis/plot_mask_gallery.py            # all 9
    python analysis/plot_mask_gallery.py Nicoletti   # one reservoir
"""
import pathlib, sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_frs_dem import mask_pool, RESERVOIRS

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT = pathlib.Path('analysis/schwatke_output/mask_gallery')
OUT.mkdir(parents=True, exist_ok=True)


def gallery_one(name):
    # mask_pool()'s own wl_m is the production-assigned level (gauge, SWOT
    # fallback, or curve-inversion for densified-only dates) -- the same
    # level each mask is actually stacked at when building the DEM.
    pool = mask_pool(name).sort_values('wl_m').reset_index(drop=True)

    n = len(pool)
    ncols = 8
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))
    axes = np.atleast_2d(axes)
    for i, row in pool.iterrows():
        ax = axes.flat[i]
        fp = MASK_DIR / f'mask_{name}_{row.date}.tif'
        if not fp.exists():
            ax.axis('off')
            continue
        with rasterio.open(fp) as src:
            arr = src.read(1)
        ax.imshow(arr, cmap='Blues', vmin=0, vmax=1)
        lbl = f"{row.date}\n{row.area_ha:.1f} ha"
        if pd.notna(row.wl_m):
            lbl += f"  {row.wl_m:.2f} m"
        ax.set_title(lbl, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes.flat[j].axis('off')

    fig.suptitle(f'{name}: mask pool in level order (n={n})', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_fp = OUT / f'mask_gallery_{name}.png'
    fig.savefig(out_fp, dpi=130)
    plt.close(fig)
    print(f'{name}: saved {out_fp} ({n} masks)')


if __name__ == '__main__':
    names = sys.argv[1:] or RESERVOIRS
    for n in names:
        gallery_one(n)
