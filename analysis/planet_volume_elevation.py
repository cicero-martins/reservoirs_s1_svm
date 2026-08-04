"""
planet_volume_elevation.py (2026-08-04)

Cross-sensor validation in volume-elevation space: the PlanetScope reconstruction, the
Sentinel-1 reconstruction and the original design curve, for the four reservoirs
PlanetScope covers.

Replaces the earlier operational time-series check (planet_operational_check.py). That
version converted each reconstruction to a storage curve and inverted it, area -> volume,
to drive a multi-year volume series. The inversion proved fragile: these sparse
level-slice DEMs end on a wide flat shelf, so area grows over a very thin elevation
range at the top and the inverse mapping collapses there. At Ancipa it plateaued near
11 Mm3 even though that DEM genuinely holds 23.05 Mm3, and linear extrapolation beyond
the observed area range diverged to 220 Mm3. Volume-elevation is the reconstruction's
native space and needs no inversion, so the comparison is direct and the observability
limit of each sensor is visible as the extent of its own curve.

Same convention as Fig. aevgrid: volume relative to a common floor, the design curve
shown for reference, each reconstruction drawn only over the band it actually observes.

Output: analysis/schwatke_output/planet/planet_volume_elevation.{csv,png}
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path('.')
sys.path.insert(0, str(REPO / 'tool'))
sys.path.insert(0, str(REPO / 'analysis'))
import bathymetry as bt   # noqa: E402

OUT = REPO / 'analysis' / 'schwatke_output' / 'planet'
SITES = ['Ancipa', 'Pozzillo', 'Poma', 'Rosamarina']


def curve(elev, mask, pixel_ha, levels, floor):
    """Volume (Mm3) above `floor` at each level, by exact per-pixel water column."""
    pix_m2 = pixel_ha * 1e4
    return np.array([np.sum(np.clip(h - elev, 0.0, h - floor)[mask]) * pix_m2 / 1e6
                     for h in levels])


rows = []
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
for ax, site in zip(axes.flat, SITES):
    pl_fp = OUT / f'dem_{site}_Planet.tif'
    if not pl_fp.exists():
        ax.axis('off'); continue
    with rasterio.open(pl_fp) as src:
        pl = src.read(1).astype(float)
        pl_pix_ha = abs(src.transform.a * src.transform.e) / 1e4
    pl_mask = np.isfinite(pl)

    sar = bt.load_dem(site, 'B')
    s_arr, s_mask = sar['arr'], sar['mask']

    # Common floor so the three curves share one vertical datum, then each sensor is
    # drawn only up to its OWN highest observed level: the point of the figure is partly
    # that those ceilings differ.
    floor = min(float(np.nanmin(pl[pl_mask])), float(np.nanmin(s_arr[s_mask])))
    pl_top = float(np.nanmax(pl[pl_mask]))
    s_top = float(np.nanmax(s_arr[s_mask]))

    lv_pl = np.linspace(floor, pl_top, 160)
    lv_sar = np.linspace(floor, s_top, 160)
    v_pl = curve(pl, pl_mask, pl_pix_ha, lv_pl, floor)
    v_sar = curve(s_arr, s_mask, bt.PIXEL_HA, lv_sar, floor)

    dc = bt.design_curve(site)
    lv_des = np.linspace(floor, max(pl_top, s_top), 200)
    v_des = (dc[1](lv_des) - float(dc[1](floor))) if dc is not None else None

    upd = bt.updated_curve(site)
    v_upd = None
    if upd is not None and upd[1] is not None:
        v_upd = upd[1](lv_des) - float(upd[1](floor))

    if v_des is not None:
        ax.plot(v_des, lv_des, color='#444444', lw=1.6, ls='--', label='Design curve')
    if v_upd is not None:
        ax.plot(v_upd, lv_des, color='#2e7d32', lw=1.6, ls='-.', label='Updated curve')
    ax.plot(v_sar, lv_sar, color='#6a1b9a', lw=2.4, label='SAR reconstruction')
    ax.plot(v_pl, lv_pl, color='#0277bd', lw=2.4, ls='--', label='PlanetScope reconstruction')

    ax.set_title(f"{site} (A/P {bt.RESERVOIRS[site]['ap']:.0f} m)", fontsize=12, loc='left')
    ax.set_xlabel('Volume above common floor (Mm$^3$)', fontsize=10)
    ax.set_ylabel('Water level (m ASL)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)

    rows.append(dict(reservoir=site, ap_m=bt.RESERVOIRS[site]['ap'],
                     floor_m=round(floor, 2),
                     sar_top_m=round(s_top, 2), planet_top_m=round(pl_top, 2),
                     sar_vol_Mm3=round(float(v_sar[-1]), 2),
                     planet_vol_Mm3=round(float(v_pl[-1]), 2),
                     design_vol_at_sar_top_Mm3=(
                         round(float(np.interp(s_top, lv_des, v_des)), 2)
                         if v_des is not None else np.nan)))

# Union of handles across panels, not just the first: Ancipa and Pozzillo have no
# updated curve, so taking the legend from one panel would silently drop that entry.
_seen, _h, _l = set(), [], []
for _ax in axes.flat:
    for _hh, _ll in zip(*_ax.get_legend_handles_labels()):
        if _ll not in _seen:
            _seen.add(_ll); _h.append(_hh); _l.append(_ll)
fig.legend(_h, _l, loc='lower center', ncol=4, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / 'planet_volume_elevation.png', dpi=170, bbox_inches='tight')
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'planet_volume_elevation.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print(f"\nSaved {OUT/'planet_volume_elevation.png'}")
