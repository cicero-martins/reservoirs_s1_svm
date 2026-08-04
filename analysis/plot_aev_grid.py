"""
plot_aev_grid.py

Publication figure: reconstructed (SAR) volume-elevation curve against the
design curve and, where available, the updated official survey/curve, for
all 9 reservoirs in one grid, ordered by A/P. All volumes relative to each
reservoir's own DEM floor, matching Table tab:capacity in Results.

Reuses consolidate_bathymetry.py's design/updated curve loaders directly
(importing it re-runs its own consolidation as a side effect -- harmless,
just reprints its own table) so the curves definitions can never drift
from what generated the numbers already reported in the Results text.

Output: manuscript_paper2/figures/aev_grid.png
"""
import sys, pathlib
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import consolidate_bathymetry as cb   # noqa: E402  (side effect: reprints its own table)

OUT_DIR = cb.OUT
ORDER = [
    ('Olivo', 50.7), ('Ancipa', 90.5), ('Nicoletti', 119.7), ('Castello', 126.7),
    ('Garcia', 167.7), ('Arancio', 182.2), ('Rosamarina', 187.4), ('Poma', 190.1),
    ('Pozzillo', 240.5),
]


def sar_vol_curve(dem, mask, floor, top, n=60):
    hs = np.linspace(floor, top, n)
    vols = np.array([cb.vol_exact(dem, mask, floor, h) for h in hs])
    return hs, vols


# 5x2 rather than 3x3: at 3x3 each panel was too small to read in print. The tenth
# slot carries the legend shared by all nine panels, so no panel loses plot area to
# an in-panel legend box repeating the same entries nine times.
fig, axes = plt.subplots(5, 2, figsize=(13, 17))
for ax, (name, ap) in zip(axes.flat, ORDER):
    cfg = cb.RES[name]
    dem_path = OUT_DIR / f'dem_{name}_B.tif'
    with rasterio.open(dem_path) as s:
        dem = s.read(1).astype(np.float64)
    mask = ~np.isnan(dem)
    floor, top = float(np.nanmin(dem[mask])), float(np.nanmax(dem[mask]))

    hs, v_sar = sar_vol_curve(dem, mask, floor, top)
    v_sar_rel = v_sar  # already relative to floor by construction

    des_vol = cb.load_design_vol(name, cfg)
    v_des_abs = des_vol(hs)
    v_des_rel = v_des_abs - float(des_vol(floor))

    ax.plot(v_sar_rel, hs, '-', color='#1565c0', lw=2, label='SAR (this study)')
    ax.plot(v_des_rel, hs, '--', color='0.4', lw=1.5, label='Design curve')

    if cfg.get('updated') == 'garcia_echo':
        srv_path = OUT_DIR / 'garcia_survey' / 'survey_dem_Garcia.tif'
        with rasterio.open(srv_path) as s:
            srv = s.read(1).astype(np.float64)
        srv_mask = np.isfinite(srv)
        hs_g, v_srv = sar_vol_curve(srv, srv_mask, floor, top)
        ax.plot(v_srv, hs_g, ':', color='#2e7d32', lw=2, label='Echo-sounder survey')
    elif cfg.get('updated'):
        upd_vol = cb.load_updated_vol(cfg['updated'])
        if upd_vol is not None:
            v_upd_abs = upd_vol(hs)
            v_upd_rel = v_upd_abs - float(upd_vol(floor))
            ax.plot(v_upd_rel, hs, ':', color='#2e7d32', lw=2, label='Updated curve')

    ax.set_title(f'{name} (A/P {ap:.0f})', fontsize=10)
    ax.set_xlabel('Volume above floor (Mm³)', fontsize=8)
    ax.set_ylabel('Elevation (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

fig.suptitle('Reconstructed volume-elevation curves vs design and updated references', fontsize=13)
_h, _l = axes.flat[0].get_legend_handles_labels()
axes.flat[9].axis('off')
axes.flat[9].legend(_h, _l, loc='center', fontsize=13, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = pathlib.Path('manuscript_paper2/figures/aev_grid.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f'Saved {out}')
