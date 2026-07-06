"""
screen_gdw_candidates.py

Screen the Global Dam Watch (GDW v1.0) reservoir polygons for NEW pilot candidates
that fill the geographic/biome blanks of the current 42 (Europe/US over-represented;
tropical, Asia, South America, sub-Saharan Africa, Middle East thin or empty).

Filters (match the existing pilot's envelope):
  - surface area 250–980 ha  (2.5–9.8 km²; pilot band is 240–969 ha)
  - dam built ≤ 2012          (stable, pre-Sentinel-1, so the 2014-21 series is clean)
  - has a reservoir name       (traceable / not a random pond)
  - inside one of the TARGET_REGIONS bounding boxes (the blanks)

For each survivor it computes an A/P estimate from the GDW polygon (equal-area proj,
metres) so we can also spread candidates across the A/P axis, and prints the best few
per region. NOTHING is committed — this only proposes; JRC coverage is verified later
at export time (getLakePoly needs JRC water within 10 km).

Reads:  raw_data/GDW2024/GDW_v1_0_shp.zip  (extracted to scratch by the caller)
Output: analysis/gdw_new_candidates.csv  (+ printed shortlist)
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')

SHP = sys.argv[1] if len(sys.argv) > 1 else None
if SHP is None:
    sys.exit('usage: screen_gdw_candidates.py <path-to-GDW_reservoirs_v1_0.shp>')

AREA_MIN_HA, AREA_MAX_HA = 250, 980
YEAR_MAX = 2012

# Bounding boxes over the currently EMPTY/THIN regions, with the biome we want them for.
# (lon_min, lon_max, lat_min, lat_max, target_biome)
TARGET_REGIONS = {
    'SE_Asia_tropical':   (95, 128, -10, 23,  'Tropical (Af/Am/Aw)'),
    'West_Africa_trop':   (-18, 16,  4, 15,   'Tropical (Aw/Am)'),
    'East_Africa':        (28, 42, -12, 15,   'Tropical highland/Aw'),
    'Tropical_S_America': (-75, -45, -12, 11, 'Tropical (Af/Am/Aw)'),
    'Andes_S_America':    (-78, -64, -40, -12,'Arid highland / temperate'),
    'South_Cone':         (-73, -53, -42, -25,'Temperate / semi-arid'),
    'China':              (100, 122, 22, 42,  'Humid subtropical/continental'),
    'India_N_Central':    (72, 88, 20, 32,    'Semi-arid / subtropical'),
    'Middle_East_arid':   (35, 60, 25, 40,    'Arid / semi-arid (BW/BS)'),
    'Central_Asia':       (55, 80, 37, 50,    'Cold arid (BSk/BWk)'),
    'Mexico_CAmerica':    (-112, -83, 14, 30, 'Semi-arid / tropical'),
    'Canada_boreal':      (-125, -60, 46, 58, 'Continental/boreal (Dfb/Dfc)'),
}


def region_of(lon, lat):
    for name, (x0, x1, y0, y1, _) in TARGET_REGIONS.items():
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return name
    return None


print('Reading GDW reservoirs …')
g = gpd.read_file(SHP, columns=['GDW_ID', 'RES_NAME', 'DAM_NAME', 'COUNTRY', 'RIVER',
                                'YEAR_DAM', 'AREA_SKM', 'MAIN_USE', 'ELEV_MASL',
                                'QUALITY'])
print(f'  {len(g)} reservoirs total')

# LONG_DAM/LAT_DAM are 0 for most rows → use the polygon CENTROID for coords + region.
cen = g.geometry.centroid
g['lon'] = cen.x.values
g['lat'] = cen.y.values
# name: RES_NAME, else DAM_NAME (only 2093 have RES_NAME; DAM_NAME broadens the pool)
nm = g['RES_NAME'].astype('string').str.strip()
dm = g['DAM_NAME'].astype('string').str.strip()
g['name'] = nm.where(nm.notna() & (nm != ''), dm)

g['area_ha'] = g['AREA_SKM'] * 100.0
g = g[(g['area_ha'] >= AREA_MIN_HA) & (g['area_ha'] <= AREA_MAX_HA)]
g = g[g['YEAR_DAM'].notna() & (g['YEAR_DAM'] <= YEAR_MAX) & (g['YEAR_DAM'] > 0)]
g = g[g['name'].notna() & (g['name'].astype(str).str.strip() != '')]
g['region'] = [region_of(lo, la) for lo, la in zip(g['lon'], g['lat'])]
g = g[g['region'].notna()].copy()
print(f'  {len(g)} in size/year/name band AND inside a target region\n')

# A/P estimate from the GDW polygon (equal-area metres). A/P = area / perimeter.
gm = g.to_crs(6933)   # World Cylindrical Equal Area (m)
g['ap_m'] = (gm.geometry.area / gm.geometry.length).values
g['perim_km'] = (gm.geometry.length / 1000).values


def ap_class(ap):
    return 'Low' if ap < 120 else ('Med' if ap < 250 else 'High')


g['ap_class'] = g['ap_m'].map(ap_class)

out = g[['GDW_ID', 'name', 'COUNTRY', 'RIVER', 'YEAR_DAM', 'area_ha', 'ap_m',
         'ap_class', 'perim_km', 'ELEV_MASL', 'MAIN_USE', 'QUALITY', 'region',
         'lat', 'lon']].copy()
out = out.sort_values(['region', 'ap_m']).reset_index(drop=True)
out.to_csv('analysis/gdw_new_candidates.csv', index=False)
print(f'Saved {len(out)} candidates -> analysis/gdw_new_candidates.csv\n')

# Shortlist: per region, aim for A/P spread — pick up to 3 (a Low, a Med, a High if present),
# preferring larger area (better JRC signal) and older dams within each A/P class.
print(f'{"region":<20}{"biome target":<30}{"n":>4}   A/P-class avail')
print('-' * 78)
picks = []
for reg, (_, _, _, _, biome) in TARGET_REGIONS.items():
    sub = out[out['region'] == reg]
    if sub.empty:
        print(f'{reg:<20}{biome:<30}{0:>4}   —')
        continue
    avail = sub['ap_class'].value_counts().to_dict()
    print(f'{reg:<20}{biome:<30}{len(sub):>4}   {avail}')
    for cls in ['Low', 'Med', 'High']:
        c = sub[sub['ap_class'] == cls].sort_values('area_ha', ascending=False)
        if not c.empty:
            picks.append(c.iloc[0])

pick = pd.DataFrame(picks).drop_duplicates('GDW_ID').reset_index(drop=True)
pick.to_csv('analysis/gdw_shortlist.csv', index=False)
print(f'\n=== SHORTLIST ({len(pick)}) — one per (region × A/P class) ===')
show = pick[['name', 'COUNTRY', 'region', 'YEAR_DAM', 'area_ha', 'ap_m', 'ap_class',
             'lat', 'lon']].copy()
show['area_ha'] = show['area_ha'].round(0)
show['ap_m'] = show['ap_m'].round(0)
show['lat'] = show['lat'].round(3)
show['lon'] = show['lon'].round(3)
print(show.to_string(index=False))
print(f'\n-> analysis/gdw_shortlist.csv  (edit/curate, then I wire into the export script)')
