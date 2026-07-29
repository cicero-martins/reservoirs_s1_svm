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
    clean = df[~df['date'].isin(bad)].dropna(subset=['wl_m']).copy()
    print(f'{res}: {len(df)} total candidates, {len(clean)} clean after excluding {sorted(bad)}')

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

    # Full-clean-pool densified DEM.
    raw_arrays, wls, meta = [], [], None
    for _, row in clean.iterrows():
        with rasterio.open(MASK_DIR / f'mask_{res}_{row["date"]}.tif') as src:
            arr = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
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
