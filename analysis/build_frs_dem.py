"""
build_frs_dem.py (2026-07-31)

Generalizes build_poma_swot_only_dem.py to all 9 reservoirs: builds a "full
remote sensing" (FRS) Period-B bathymetric DEM using ONLY SWOT water levels,
no gauge anywhere in the chain, by fitting a power-law hypsometric curve
A=a(h-h0)^b on genuine SAR-area/SWOT-WL coincident pairs (the full, continuous
SAR series matched to every raw SWOT observation across the mission, not just
the mask-export dates), then inverting that curve at each usable mask's own
observed area to assign it a level, and stacking exactly as the gauge-based
reconstruction does (schwatke_bathymetry_3d.build_dem_from_arrays).

Mask pool per reservoir: the densified pool (outlier==False rows of
{name}_densify_prototype_pairs.csv) where one exists (8/9 reservoirs); for
Ancipa (revisit-limited -- all 9 Sentinel-1 scenes already in its production
set, no larger pool to densify) falls back to mask_wl_pairs_Ancipa.csv's own
period-B rows.

Run:
    python analysis/build_frs_dem.py            # all 9
    python analysis/build_frs_dem.py Rosamarina  # one reservoir
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')
SWOT_DIR = pathlib.Path('validation_data/SWOT')

RESERVOIRS = ['Olivo', 'Ancipa', 'Nicoletti', 'Castello', 'Garcia',
              'Arancio', 'Rosamarina', 'Poma', 'Pozzillo']


# Extra, manually-confirmed bad dates not captured by either outlier column.
# Two sources, both confirmed by the 2026-07-31 per-reservoir visual mask-
# gallery audit (analysis/plot_mask_gallery.py) and now also applied to the
# *production* pool via finalize_reservoir.py + schwatke_bathymetry_3d.py
# phase1()/phase2():
#  (1) the 2026-01-19/02-12/03-26 regional wind-roughening event (found
#      2026-07-29) -- doesn't move total area enough to trip the
#      area_outlier/dev_pct>60% check at every reservoir (Poma's 2026-01-19 in
#      particular has a corrupted *spatial pattern* whose *total* pixel count
#      still lands in a plausible range).
#  (2) reservoir-specific bad masks found by directly eyeballing every mask in
#      the pool, sorted by assigned level, one gallery per reservoir: Garcia
#      (2025-12-14, 2026-01-07, 2026-05-01/02/13), Arancio (2026-04-01),
#      Olivo (2026-03-26), Nicoletti (2026-03-14), Rosamarina (2026-03-15).
# Pozzillo's 2026-02-12 and Garcia's 2026-02-12 turned out to already be
# caught by the area_outlier column once finalize_reservoir.py was fixed to
# check it (it previously only excluded the manually-passed CLI dates) --
# kept here too for redundancy since this dict is also read directly, not
# just via finalize_reservoir.py. Ancipa is handled separately (see
# export_ancipa_orbit124.py): its single bad date, 2025-01-24, is a genuine
# near-dam radar-shadow gap in orbit 117 alone, fixed by adding orbit 124
# (DESCENDING) rather than by exclusion.
EXTRA_BAD_DATES = {
    'Arancio':    {'2026-01-19', '2026-02-12', '2026-03-26', '2026-04-01'},
    'Castello':   {'2026-02-13'},
    'Garcia':     {'2026-02-12', '2026-03-26', '2026-01-07', '2025-12-14',
                    '2026-05-01', '2026-05-02', '2026-05-13'},
    'Nicoletti':  {'2026-01-19', '2026-02-12', '2026-03-14', '2025-10-15'},
    'Olivo':      {'2026-01-19', '2026-02-12', '2026-03-26'},
    'Poma':       {'2026-01-19', '2026-02-12', '2026-03-26'},
    'Pozzillo':   {'2026-01-19', '2026-01-31', '2026-02-12'},
    'Rosamarina': {'2026-03-15'},
    # Ancipa: single bad date (dual-orbit confirmed near-dam radar-shadow gap,
    # see export_ancipa_orbit124.py), not the wind event.
    'Ancipa':     {'2025-01-24'},
}


def mask_pool(name):
    dens_fp = OUT_DIR / f'{name.lower()}_densify_prototype_pairs.csv'
    if dens_fp.exists():
        df = pd.read_csv(dens_fp)
        df['date'] = df['date'].astype(str)
        # Same magnitude floor as finalize_reservoir.py: at near-zero areas
        # (Ancipa's extreme-drawdown dates, <5 ha) dev_pct blows up from
        # sub-hectare noise, not real corruption -- don't drop those.
        for col in ('area_outlier', 'outlier'):
            if col in df.columns:
                df = df[~(df[col] & (df['continuous_ha'].fillna(0) > 5))]
        df = df[~df['date'].isin(EXTRA_BAD_DATES.get(name, set()))]
        df = df.dropna(subset=['wl_m'])
        return df[['date', 'area_ha', 'wl_m']]
    pairs = pd.read_csv(OUT_DIR / f'mask_wl_pairs_{name}.csv')
    pairs = pairs[pairs.period == 'B'].copy()
    pairs['date'] = pairs['date'].astype(str)
    return pairs[['date', 'area_ha', 'wl_m']]


def fit_swot_curve(name):
    """Power-law A=a(h-h0)^b fit on genuine SAR-area/SWOT-WL coincident pairs
    (+/-3 d match) -- same method as build_poma_swot_only_dem.py."""
    cfg = m.CONFIGS[name]
    swot = m.load_swot_corrected(cfg, SWOT_DIR / f'{name}_swot.csv', name)
    cont_area = pd.read_csv(cfg['sar_csv'], parse_dates=['date']).sort_values('date')
    cont_area = cont_area.groupby('date')['area_ha'].mean()

    pairs = []
    for dt, wl in swot.items():
        near = cont_area[(cont_area.index >= dt - pd.Timedelta(days=3)) &
                          (cont_area.index <= dt + pd.Timedelta(days=3))]
        if len(near):
            idx = (near.index - dt).to_series().abs().values.argmin()
            pairs.append({'wl_m': wl, 'area_ha': float(near.iloc[idx])})
    pairs = pd.DataFrame(pairs)
    if len(pairs) < 6:
        return None, pairs
    fit = m.fit_hyps_model(pairs, cfg['h0_bound_lo'])
    return fit, pairs


def build_one(name):
    print(f'\n=== {name} ===')
    fit, pairs = fit_swot_curve(name)
    if fit is None:
        print(f'  SKIP: only {len(pairs)} genuine SAR-area/SWOT-WL pairs (<6)')
        return
    a, h0, b = fit
    print(f'  {len(pairs)} genuine SAR-area/SWOT-WL pairs; A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')

    pool = mask_pool(name).copy()
    pool['swot_wl'] = pool.area_ha.apply(lambda ar: m.invert_power_law(ar, a, h0, b))
    pool = pool.dropna(subset=['swot_wl'])
    print(f'  Building FRS DEM from {len(pool)} masks '
          f'(WL {pool.swot_wl.min():.1f}-{pool.swot_wl.max():.1f} m)')

    raw_arrays, wls, meta = [], [], None
    for _, row in pool.iterrows():
        fp = MASK_DIR / f'mask_{name}_{row.date}.tif'
        if not fp.exists():
            continue
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
            elif arr.shape != (meta['height'], meta['width']):
                print(f'  skip {row.date}: grid shape {arr.shape} != '
                      f"reference {(meta['height'], meta['width'])} (different export batch)")
                continue
        raw_arrays.append(arr)
        wls.append(row.swot_wl)
    if len(raw_arrays) < 4:
        print(f'  SKIP: only {len(raw_arrays)} mask files found on disk')
        return

    dem_frs = m.build_dem_from_arrays(raw_arrays, wls)
    out_meta = meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    out_tif = OUT_DIR / f'dem_{name}_B_swotonly.tif'
    with rasterio.open(out_tif, 'w', **out_meta) as dst:
        dst.write(dem_frs[np.newaxis, :, :])
    print(f'  Saved {out_tif}  n_masks={len(wls)}')

    ref_fp = OUT_DIR / f'dem_{name}_B_densified.tif'
    if not ref_fp.exists():
        ref_fp = OUT_DIR / f'dem_{name}_B.tif'
    if not ref_fp.exists():
        print('  (no reference DEM to compare against)')
        return
    with rasterio.open(ref_fp) as src:
        dem_ref = src.read(1)
    if dem_ref.shape != dem_frs.shape:
        print(f'  (grid mismatch vs {ref_fp.name}: {dem_ref.shape} vs {dem_frs.shape} -- skip diff)')
        return
    diff = dem_frs - dem_ref
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        print('  (reference DEM has no overlapping pixels)')
        return
    print(f'  FRS vs {ref_fp.name} (n={valid.sum()} px): '
          f'bias={np.nanmean(diff):+.3f} m  RMSE={np.sqrt(np.nanmean(diff**2)):.3f} m  '
          f'std={np.nanstd(diff):.3f} m')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reservoir', nargs='?', default=None)
    args = ap.parse_args()
    names = [args.reservoir] if args.reservoir else RESERVOIRS
    for n in names:
        build_one(n)
