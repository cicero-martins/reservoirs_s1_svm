"""
Systematic DAHITI scan — find ALL Reservoir-type targets with useful WL coverage.

Strategy (inverse of dahiti_longlist_screening.py):
  1. Query DAHITI by sub-continental bounding boxes → collect ALL targets
  2. Keep only type='Reservoir' with water_level_altimetry='public'
  3. Download WL for each → count obs in S1 era (2014+), compute WL range
  4. Output ranked list for GEE area cross-referencing

GEE cross-reference (manual next step):
  For each target in the output, look up the nearest GDW polygon by coordinates
  to get area_km2. Then filter to the target range (500–10,000 ha = 5–100 km²).

Run from project root:
  python analysis/dahiti_reservoir_scan.py
"""

import csv, time, math, sys
import requests, urllib3
from pathlib import Path

urllib3.disable_warnings()
sys.stdout.reconfigure(encoding='utf-8')

API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
OUT_DIR  = Path('validation_data/DAHITI')
WL_DIR   = OUT_DIR / 'dahiti_scan_wl'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WL_DIR.mkdir(parents=True, exist_ok=True)

# Minimum WL observations in 2014+ to be included in output
MIN_OBS_2014 = 10

S = requests.Session()
S.verify = False

def api_post(endpoint, params):
    p = dict(params)
    p['api_key'] = API_KEY
    if endpoint.startswith('download-'):
        p.setdefault('format', 'json')
    try:
        r = S.post(f'{BASE_URL}/{endpoint}/', data=p, timeout=40)
        return r.json()
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── Sub-continental bounding boxes ─────────────────────────────────────────
# Subdivide to avoid potential per-query result limits.
# Each box ~20°×20° or smaller in dense regions.
REGIONS = [
    # Africa
    ('AF-NW',  18,  38, -18,  10),   # NW Africa (Morocco → Nigeria)
    ('AF-NE',  -5,  38,  10,  50),   # NE Africa (Ethiopia, Sudan, Egypt)
    ('AF-SW', -35,  18, -20,  25),   # SW Africa (Angola → South Africa)
    ('AF-SE', -35,  -5,  25,  50),   # SE Africa (Mozambique, Zimbabwe, Madagascar)
    # Europe
    ('EU-W',   35,  70, -10,  20),   # W Europe (Portugal → Germany)
    ('EU-E',   35,  70,  20,  60),   # E Europe (Poland → Russia/Turkey)
    # Middle East / Caucasus / Central Asia
    ('ME',     20,  50,  25,  75),   # ME + Caucasus + Central Asia
    # South Asia
    ('SA-IND', -5,  38,  60,  90),   # Pakistan, India, Sri Lanka, Bangladesh
    # East / SE Asia
    ('EA-CN',  18,  55,  90, 125),   # China, Vietnam, Thailand, Laos, Myanmar
    ('EA-SEA', -12, 25,  95, 145),   # Indonesia, Malaysia, Philippines, PNG
    # North America
    ('NA-W',   15,  55,-130, -95),   # Western US + Mexico + W Canada
    ('NA-E',   15,  55, -95, -55),   # Eastern US + E Canada + Caribbean
    # South America
    ('SA-N',  -15,  15, -85, -34),   # Venezuela, Colombia, Ecuador, NE Brazil
    ('SA-S',  -56, -15, -80, -34),   # Argentina, Chile, Uruguay, S Brazil
    # Oceania
    ('OC-AU', -45,  -9, 110, 155),   # Australia + Papua New Guinea
    ('OC-PAC',-50,   0, 155, 180),   # NZ + Pacific islands
]

# ── Step 1: collect all Reservoir targets ─────────────────────────────────
print("=== DAHITI continental scan — collecting Reservoir targets ===\n")
all_targets = {}  # dahiti_id → target dict

for region_name, min_lat, max_lat, min_lon, max_lon in REGIONS:
    r = api_post('list-targets', {
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_lon': min_lon, 'max_lon': max_lon,
    })
    targets = r.get('data') or []
    reservoirs = [t for t in targets if t.get('type') == 'Reservoir']
    for t in targets:
        did = str(t['dahiti_id'])
        if did not in all_targets:
            all_targets[did] = t
    print(f"  {region_name:12s}: {len(targets):4d} total  |  {len(reservoirs):4d} Reservoir")
    time.sleep(0.3)

reservoirs = {k: v for k, v in all_targets.items()
              if v.get('type') == 'Reservoir'}
print(f"\nTotal unique targets : {len(all_targets)}")
print(f"Type=Reservoir       : {len(reservoirs)}")

# Note: skip pre-filtering by data_access — not reliable in list-targets response.
# Just attempt WL download for all Reservoir-type targets; empty response = skip.
print(f"WL altimetry=public  : (not pre-filtered; will attempt download for all)")

# ── Step 2: download WL for all Reservoir targets ─────────────────────────
print(f"\n=== Downloading WL for {len(reservoirs)} Reservoir targets ===")
print(f"(only those with WL 2014+ >= {MIN_OBS_2014} will be kept)\n")

results = []
n_kept = 0

for i, (did, t) in enumerate(sorted(reservoirs.items(), key=lambda x: x[0])):
    name = t.get('target_name', '')
    lat  = float(t.get('latitude', 0))
    lon  = float(t.get('longitude', 0))

    r2 = api_post('download-water-level', {'dahiti_id': did})
    data_wl = r2.get('data') or []
    time.sleep(0.2)

    if not data_wl:
        continue

    dates = sorted(str(d.get('datetime', ''))[:10] for d in data_wl if d.get('datetime'))
    d2014 = [dt for dt in dates if dt >= '2014-01-01']
    if len(d2014) < MIN_OBS_2014:
        continue

    wse_vals = []
    for d in data_wl:
        try:
            wse_vals.append(float(d['wse']))
        except (ValueError, KeyError):
            pass

    wl_range = round(max(wse_vals) - min(wse_vals), 2) if wse_vals else 0.0

    row = {
        'dahiti_id':     did,
        'name':          name,
        'lat':           round(lat, 4),
        'lon':           round(lon, 4),
        'wl_total':      len(data_wl),
        'wl_2014':       len(d2014),
        'wl_min':        round(min(wse_vals), 2) if wse_vals else '',
        'wl_max':        round(max(wse_vals), 2) if wse_vals else '',
        'wl_range_m':    wl_range,
        'wl_date_start': dates[0]  if dates else '',
        'wl_date_end':   dates[-1] if dates else '',
        'area_km2_gee':  '',    # to be filled by GEE lookup
        'gdw_id':        '',    # to be filled by GEE lookup
        'continent':     '',    # to be filled manually / GEE
        'biome':         '',
        'note':          '',
    }
    results.append(row)
    n_kept += 1

    # Save WL CSV
    safe = name.replace(' ', '_').replace('/', '_').replace(',', '')[:40]
    out = WL_DIR / f"{did}_{safe}_wl.csv"
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data_wl[0].keys()))
        w.writeheader(); w.writerows(data_wl)

    if i % 10 == 0 or n_kept <= 5:
        print(f"  [{did:>6}] {name:35s}  WL2014={len(d2014):4d}  "
              f"range={wl_range:6.2f}m  {dates[0][:7]}–{dates[-1][:7]}")

# ── Step 3: save and summarise ─────────────────────────────────────────────
results.sort(key=lambda x: (-x['wl_2014'], x['name']))

out_csv = OUT_DIR / 'dahiti_reservoir_scan_results.csv'
fields  = ['dahiti_id', 'name', 'lat', 'lon',
           'wl_total', 'wl_2014', 'wl_min', 'wl_max', 'wl_range_m',
           'wl_date_start', 'wl_date_end',
           'area_km2_gee', 'gdw_id', 'continent', 'biome', 'note']
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(results)

print(f"\n{'='*70}")
print(f"Reservoirs with WL 2014+ >= {MIN_OBS_2014}: {n_kept}")
print(f"Saved: {out_csv}")
print(f"WL CSVs: {WL_DIR}  ({n_kept} files)")
print(f"\nNext step: paste coordinates into GEE to get area_km2 from GDW,")
print(f"then filter to target range 5–100 km² (500–10,000 ha).")
