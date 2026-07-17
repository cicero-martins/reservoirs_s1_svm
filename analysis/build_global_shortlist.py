"""
build_global_shortlist.py  (Fase 4 -- worldwide transferability demo, Paper 2)

Joins three already-local datasets to select a small, defensible global sample of
reservoirs spanning low/medium/high shoreline A/P, with a GRDL AEV reference curve
to validate against and DAHITI altimetry to reconstruct with.

Start from the SCARCEST resource (DAHITI altimetry coverage, ~110 reservoirs already
GDW_ID-matched in validation_data/DAHITI/*.csv) rather than from a pre-filtered GDW
candidate pool (analysis/gdw_new_candidates.csv is a DIFFERENT, disjoint screen built
for an earlier purpose -- zero overlap with the DAHITI set, confirmed empirically).
Attributes (area, A/P, YEAR_DAM, country) are looked up directly in the full GDW v1.0
reservoir polygons (raw_data/GDW2024/GDW_v1_0_shp.zip). A/P is computed the same way
as analysis/screen_gdw_candidates.py: polygon area/perimeter in an equal-area (World
Mollweide) projection.

Output: analysis/global_shortlist_candidates.csv (all matches, for review) and a
suggested final shortlist spanning A/P classes and countries -- a proposal to review
BEFORE spending GEE export quota, not an automatic final answer.

Run:
    python analysis/build_global_shortlist.py
"""
import pathlib
import zipfile

import geopandas as gpd
import pandas as pd

REPO = pathlib.Path('.')
DAHITI_FILES = [
    REPO / 'validation_data' / 'DAHITI' / 'dahiti_reservoir_scan_results.csv',
    REPO / 'validation_data' / 'DAHITI' / 'dahiti_scan_medium.csv',
]
GDW_SHP_ZIP = REPO / 'raw_data' / 'GDW2024' / 'GDW_v1_0_shp.zip'
GDW_SHP_INNER = 'GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
GRDL_ZIP = REPO / 'validation_data' / 'GRDL.zip'
OUT_CSV = REPO / 'analysis' / 'global_shortlist_candidates.csv'

EQUAL_AREA_CRS = 'ESRI:54009'  # World Mollweide, metres


def load_dahiti():
    frames = [pd.read_csv(f) for f in DAHITI_FILES if f.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['gdw_id']).copy()
    df['GDW_ID'] = df['gdw_id'].astype(int)
    df = df.sort_values('wl_range_m', ascending=False).drop_duplicates('GDW_ID')
    return df[['GDW_ID', 'wl_range_m', 'wl_date_start', 'wl_date_end', 'continent']]


def grdl_ids_available():
    with zipfile.ZipFile(GRDL_ZIP) as z:
        names = z.namelist()
    return {int(pathlib.Path(n).stem) for n in names if pathlib.Path(n).stem.isdigit()}


def ap_class(ap_m):
    if ap_m < 120:
        return 'Low'
    if ap_m < 250:
        return 'Med'
    return 'High'


def main():
    dahiti = load_dahiti()
    grdl_ids = grdl_ids_available()
    print(f'DAHITI-matched GDW reservoirs: {len(dahiti)}  |  GRDL curves available: {len(grdl_ids)}')

    print('Reading GDW v1.0 reservoir polygons (one-time, ~large file)...')
    gdf = gpd.read_file(f'zip://{GDW_SHP_ZIP}!{GDW_SHP_INNER}')
    gdf = gdf[gdf['GDW_ID'].isin(dahiti['GDW_ID'])].copy()
    print(f'  matched in GDW polygons: {len(gdf)}')

    gdf_eq = gdf.to_crs(EQUAL_AREA_CRS)
    gdf['area_m2'] = gdf_eq.geometry.area
    gdf['perim_m'] = gdf_eq.geometry.length
    gdf['ap_m'] = gdf['area_m2'] / gdf['perim_m'].replace(0, pd.NA)
    gdf['area_ha'] = gdf['area_m2'] / 1e4
    gdf['ap_class'] = gdf['ap_m'].apply(ap_class)
    gdf['has_grdl'] = gdf['GDW_ID'].isin(grdl_ids)

    merged = gdf.merge(dahiti, on='GDW_ID', how='left')
    keep_cols = ['GDW_ID', 'RES_NAME', 'COUNTRY', 'YEAR_DAM', 'area_ha', 'ap_m',
                 'ap_class', 'LONG_DAM', 'LAT_DAM', 'wl_range_m', 'has_grdl']
    merged = merged[keep_cols].sort_values(['has_grdl', 'ap_class', 'wl_range_m'],
                                            ascending=[False, True, False])
    merged.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}\n')

    with_grdl = merged[merged['has_grdl']]
    print(f'Candidates with BOTH DAHITI altimetry AND a GRDL curve: {len(with_grdl)}\n')
    print(f"{'GDW_ID':>7} {'name':28s} {'country':16s} {'ap_class':6s} {'ap_m':>7} "
          f"{'YEAR_DAM':>8} {'area_ha':>9} {'wl_range_m':>10}")
    for _, r in with_grdl.iterrows():
        print(f"{r['GDW_ID']:>7} {str(r['RES_NAME'])[:28]:28s} {str(r['COUNTRY'])[:16]:16s} "
              f"{r['ap_class']:6s} {r['ap_m']:7.0f} {str(r['YEAR_DAM']):>8} "
              f"{r['area_ha']:9.0f} {r['wl_range_m']:10.2f}")

    print('\n=== Suggested shortlist (best wl_range_m per A/P class x country) ===')
    picked = (with_grdl.sort_values('wl_range_m', ascending=False)
              .drop_duplicates(subset=['ap_class', 'COUNTRY'])
              .groupby('ap_class', group_keys=False).head(4))
    picked = picked.sort_values(['ap_class', 'wl_range_m'], ascending=[True, False])
    for _, r in picked.iterrows():
        print(f"{r['GDW_ID']:>7} {str(r['RES_NAME'])[:28]:28s} {str(r['COUNTRY'])[:16]:16s} "
              f"{r['ap_class']:6s} {r['ap_m']:7.0f} {str(r['YEAR_DAM']):>8} "
              f"{r['area_ha']:9.0f} {r['wl_range_m']:10.2f}")


if __name__ == '__main__':
    main()
