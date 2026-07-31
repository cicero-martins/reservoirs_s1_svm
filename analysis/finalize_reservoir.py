"""
finalize_reservoir.py (2026-07-29)

For a given reservoir: exclude confirmed-bad dates from its densify_prototype_pairs
CSV, reselect 10 WL-stratified production B dates from the clean pool (mirroring
the Poma/Rosamarina fix), write them into selected_mask_dates.json, and build the
full-clean-pool densified DEM (dem_{res}_B_densified.tif). Does NOT itself rebuild
the production dem_{res}_B.tif -- run schwatke_bathymetry_3d.phase1()/phase2()
afterward (once for all reservoirs) to do that.

Run:
    python analysis/finalize_reservoir.py Pozzillo 2026-01-19 2026-02-12
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

MASK_DIR = pathlib.Path('raw_data/GEE_SicilyMasks')
OUT_DIR = pathlib.Path('analysis/schwatke_output')


def main(res, bad_dates):
    bad = set(bad_dates)
    df = pd.read_csv(OUT_DIR / f'{res.lower()}_densify_prototype_pairs.csv')
    df['date'] = df['date'].astype(str)
    # Also respect the existing area-outlier flag (dev_pct>60% vs the continuous
    # series) -- finalize_reservoir.py previously only excluded the manually-passed
    # bad_dates, silently keeping already-flagged corrupted dates (found via
    # Pozzillo 2026-02-12: area_outlier=True, dev_pct=405.6%, same regional
    # wind-roughening event as the manually-flagged dates elsewhere). Restricted to
    # continuous_ha > 5 ha: at near-zero denominators (Ancipa's extreme-drawdown
    # dates, <3 ha) dev_pct blows up to 60-240% from sub-hectare noise, not real
    # corruption -- both orbits' tiny values there roughly agree, and these are
    # exactly the deepest, most informative masks for capacity reconstruction.
    outlier_col = 'area_outlier' if 'area_outlier' in df.columns else (
        'outlier' if 'outlier' in df.columns else None)
    if outlier_col:
        is_outlier = df[outlier_col].fillna(False) & (df['continuous_ha'].fillna(0) > 5)
    else:
        is_outlier = pd.Series(False, index=df.index)
    clean = df[~df['date'].isin(bad) & ~is_outlier].dropna(subset=['wl_m']).copy()
    print(f'{res}: {len(df)} total candidates, {len(clean)} clean after excluding '
          f'{sorted(bad)} + {int(is_outlier.sum())} pre-flagged area-outlier date(s)')

    # Reselect 10 WL-stratified production dates from the clean pool.
    wl_lo, wl_hi = np.percentile(clean['wl_m'], [5, 95])
    targets = np.linspace(wl_lo, wl_hi, 10)
    used, picked = set(), []
    for t in targets:
        cand = clean[~clean['date'].isin(used)]
        idx = (cand['wl_m'] - t).abs().idxmin()
        row = cand.loc[idx]
        used.add(row['date'])
        picked.append(row)
    picked = pd.DataFrame(picked).sort_values('date')
    print(picked[['date', 'area_ha', 'wl_m', 'source', 'is_new']].to_string(index=False))

    dj_path = m.DATES_JSON
    dj = json.loads(dj_path.read_text())
    dj[res]['B'] = [{'date': r['date'], 'area_ha': round(float(r['area_ha']), 2), 'pct': -1}
                     for _, r in picked.iterrows()]
    dj_path.write_text(json.dumps(dj, indent=2))
    print(f'Updated {dj_path} ({res}.B -> {len(picked)} dates)')

    # Full-clean-pool densified DEM. Skip masks from a different export batch
    # (different AOI polygon/buffer -> different grid shape) rather than crash
    # np.stack -- same guard as build_frs_dem.py's mask_pool loop.
    raw_arrays, wls, meta = [], [], None
    for _, row in clean.iterrows():
        with rasterio.open(MASK_DIR / f'mask_{res}_{row["date"]}.tif') as src:
            arr = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
            elif arr.shape != (meta['height'], meta['width']):
                print(f'  skip {row["date"]}: grid shape {arr.shape} != '
                      f"reference {(meta['height'], meta['width'])} (different export batch)")
                continue
        raw_arrays.append(arr)
        wls.append(row['wl_m'])
    dem = m.build_dem_from_arrays(raw_arrays, wls)
    out_meta = meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    out_tif = OUT_DIR / f'dem_{res}_B_densified.tif'
    with rasterio.open(out_tif, 'w', **out_meta) as dst:
        dst.write(dem[np.newaxis, :, :])
    print(f'Saved {out_tif}  WL range {min(wls):.1f}-{max(wls):.1f} m  n_masks={len(wls)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reservoir')
    ap.add_argument('bad_dates', nargs='*')
    args = ap.parse_args()
    main(args.reservoir, args.bad_dates)
