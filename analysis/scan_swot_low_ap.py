"""
scan_swot_low_ap.py  (Fase 4 -- worldwide transferability demo, Paper 2)

Checks SWOT altimetry coverage for the low-A/P GDW candidates that DAHITI missed
(DAHITI is nadir-track altimetry -- narrow reservoirs are exactly the case it
misses; SWOT's wide swath is the whole point of using it here instead).

Key discovery (verified against GRDL.zip): the SWOT Prior Lake Database's
`p_res_id` field IS the GDW_ID/GRAND_ID directly -- no separate lake_id lookup
table needed. For each candidate we search+download a handful of
SWOT_L2_HR_LakeSP_prior_2.0 granules covering its coordinates (2023-07 onward),
filter each granule's features to p_res_id == GDW_ID, and collect the water-
surface-elevation (wse) values found to estimate a usable water-level range.

Run:
    python analysis/scan_swot_low_ap.py
"""
import pathlib
import sys

import geopandas as gpd
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
import earthaccess

OUT_DIR = pathlib.Path('scratch_swot_scan')
OUT_DIR.mkdir(exist_ok=True)
N_GRANULES_PER_CANDIDATE = 6
BUF_DEG = 0.15


def load_candidates(n=20):
    df = pd.read_csv('analysis/gdw_new_candidates.csv')
    low = df[df['ap_class'] == 'Low'].sort_values('area_ha', ascending=False)
    return low.head(n)


def scan_one(gdw_id, name, country, lat, lon):
    results = earthaccess.search_data(
        short_name='SWOT_L2_HR_LakeSP_prior_2.0',
        bounding_box=(lon - BUF_DEG, lat - BUF_DEG, lon + BUF_DEG, lat + BUF_DEG),
        temporal=('2023-07-01', '2026-06-01'),
        count=N_GRANULES_PER_CANDIDATE,
    )
    if not results:
        return dict(GDW_ID=gdw_id, name=name, country=country, n_granules=0,
                    n_obs=0, wl_range_m=None)

    wl_vals = []
    for g in results:
        try:
            files = earthaccess.download([g], str(OUT_DIR))
        except Exception:
            continue
        for f in files:
            try:
                gdf = gpd.read_file(f'zip://{f}' if str(f).endswith('.zip') else f)
            except Exception:
                continue
            hit = gdf[gdf['p_res_id'] == gdw_id]
            hit = hit[(hit['wse'] > -1e6) & hit['wse'].notna()]
            wl_vals += hit['wse'].tolist()

    if not wl_vals:
        return dict(GDW_ID=gdw_id, name=name, country=country,
                    n_granules=len(results), n_obs=0, wl_range_m=None)
    return dict(GDW_ID=gdw_id, name=name, country=country, n_granules=len(results),
                n_obs=len(wl_vals), wl_range_m=round(max(wl_vals) - min(wl_vals), 2))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cands = load_candidates(n)
    earthaccess.login(strategy='netrc')
    rows = []
    for _, r in cands.iterrows():
        print(f"[{r['name']}, {r['COUNTRY']}] GDW_ID={r['GDW_ID']}  "
              f"area={r['area_ha']:.0f}ha  ap={r['ap_m']:.0f}m ...", flush=True)
        res = scan_one(int(r['GDW_ID']), r['name'], r['COUNTRY'], r['lat'], r['lon'])
        print(f"  -> granules={res['n_granules']}  obs={res['n_obs']}  "
              f"wl_range_m={res['wl_range_m']}")
        rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv('analysis/swot_low_ap_scan.csv', index=False)
    print('\nSaved analysis/swot_low_ap_scan.csv')
    print(out.sort_values('wl_range_m', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
