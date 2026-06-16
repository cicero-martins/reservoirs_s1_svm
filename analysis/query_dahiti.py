"""
Query DAHITI API v2 for each pilot + Sicilian reservoir.
Uses POST requests + requests library (matching official DAHITI script style).
Steps per reservoir:
  1. list-targets with bounding box → find DAHITI ID + check data_access
  2. download-surface-area (if public) → save CSV
  3. download-water-level (fallback) → save CSV
"""
import json, time, csv, math
import requests
from pathlib import Path

API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
BASE     = Path('F:/reservoirs_s1_svm/validation_data')
OUT_DIR  = BASE / 'DAHITI'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.verify = False  # bypass DAHITI SSL cert chain issue on Windows
import urllib3; urllib3.disable_warnings()

def api_post(endpoint, params):
    params['api_key'] = API_KEY
    try:
        r = SESSION.post(f'{BASE_URL}/{endpoint}/', data=params, timeout=30)
        return r.json()
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── Reservoir list ─────────────────────────────────────────────────
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'
with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot_rows = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

reservoirs = []
for r in pilot_rows:
    reservoirs.append({
        'name': r['Name'], 'lat': float(r['Latitude']),
        'lon': float(r['Longitude']), 'group': 'pilot',
        'growl_id': r['RES_ID'], 'country': r.get('Country', ''),
    })
for item in [
    ('Poma',       37.9945, 13.090, 0.942),
    ('Rosamarina', 37.9435, 13.640, 0.889),
    ('Pozzillo',   37.700,  14.530, 0.969),
    ('Ancipa',     37.830,  14.573, 0.808),
]:
    reservoirs.append({'name': item[0], 'lat': item[1], 'lon': item[2],
                       'group': 'anchor', 'kge_planet': item[3],
                       'growl_id': '', 'country': 'Italy'})

# ── Query loop ─────────────────────────────────────────────────────
BBOX_DEG  = 0.2   # ±0.2° bounding box (~22 km)
DIST_MAX  = 15.0  # km — ignore matches farther than this
results   = []

print(f"\n{'Name':30s}  {'Country':12s}  {'DAHITI ID':>10}  {'Type':10s}  {'Dist':>6}  "
      f"{'SA pts':>7}  {'WL pts':>7}  {'2014+':>6}")
print('─'*100)

for res in reservoirs:
    lat, lon = res['lat'], res['lon']

    # Step 1: list-targets in small bounding box
    r1 = api_post('list-targets', {
        'min_lat': lat - BBOX_DEG, 'max_lat': lat + BBOX_DEG,
        'min_lon': lon - BBOX_DEG, 'max_lon': lon + BBOX_DEG,
    })
    targets = r1.get('data') or []

    # Find closest match
    best = None
    best_dist = 9999
    for t in targets:
        d = haversine(lat, lon, float(t['latitude']), float(t['longitude']))
        if d < best_dist:
            best_dist = d
            best = t

    if best is None or best_dist > DIST_MAX:
        n_found = len(targets)
        print(f"{res['name']:30s}  {res['country']:12s}  {'—':>10}  "
              f"{'no match':10s}  {best_dist if best else 0:6.1f}  "
              f"{'—':>7}  {'—':>7}  ({n_found} targets in box)")
        results.append({**res, 'dahiti_id': '', 'dahiti_name': '', 'dist_km': best_dist,
                        'sa_total': 0, 'sa_2014': 0, 'wl_total': 0, 'wl_2014': 0})
        continue

    dahiti_id   = best['dahiti_id']
    dahiti_name = best['target_name']
    ttype       = best.get('type', '?')
    da          = best.get('data_access') or {}

    # Step 2: surface area
    sa_total = sa_2014 = 0
    if da.get('surface_area') == 'public':
        r2 = api_post('download-surface-area', {'dahiti_id': dahiti_id})
        data_sa = r2.get('data') or []
        sa_total = len(data_sa)
        d2014 = [d for d in data_sa if str(d.get('date', '')) >= '2014-01-01']
        sa_2014 = len(d2014)
        if sa_total:
            out = OUT_DIR / f"{dahiti_id}_{res['name'].replace(' ','_')}_SA.csv"
            with open(out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data_sa[0].keys()))
                w.writeheader(); w.writerows(data_sa)

    # Step 3: water level (fallback / supplement)
    wl_total = wl_2014 = 0
    if da.get('water_level_altimetry') == 'public':
        r3 = api_post('download-water-level', {'dahiti_id': dahiti_id})
        data_wl = r3.get('data') or []
        wl_total = len(data_wl)
        wl_2014 = len([d for d in data_wl if str(d.get('date', '')) >= '2014-01-01'])
        if wl_total:
            out = OUT_DIR / f"{dahiti_id}_{res['name'].replace(' ','_')}_WL.csv"
            with open(out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(data_wl[0].keys()))
                w.writeheader(); w.writerows(data_wl)

    flag = '✓SA' if sa_2014 > 0 else ('✓WL' if wl_2014 > 0 else '—')
    print(f"{res['name']:30s}  {res['country']:12s}  {str(dahiti_id):>10}  "
          f"{ttype:10s}  {best_dist:6.1f}  {sa_total:7d}  {wl_total:7d}  "
          f"{max(sa_2014, wl_2014):6d}  {flag}")

    results.append({
        'name':        res['name'], 'group': res['group'],
        'country':     res.get('country', ''), 'growl_id': res.get('growl_id', ''),
        'lat':         lat, 'lon': lon, 'kge_planet': res.get('kge_planet', ''),
        'dahiti_id':   dahiti_id,   'dahiti_name': dahiti_name,
        'dist_km':     round(best_dist, 2), 'type': ttype,
        'sa_total':    sa_total,    'sa_2014': sa_2014,
        'wl_total':    wl_total,    'wl_2014': wl_2014,
    })
    time.sleep(0.3)

# ── Summary ────────────────────────────────────────────────────────
summary_path = OUT_DIR / 'dahiti_coverage_summary.csv'
fields = ['name','group','country','growl_id','lat','lon','kge_planet',
          'dahiti_id','dahiti_name','dist_km','type',
          'sa_total','sa_2014','wl_total','wl_2014']
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(results)

sa_ok = [r for r in results if r.get('sa_2014', 0) > 0]
wl_ok = [r for r in results if r.get('wl_2014', 0) > 0]
print(f"\n{'='*60}")
print(f"Consultados: {len(results)}  |  SA 2014+: {len(sa_ok)}  |  WL 2014+: {len(wl_ok)}")
print(f"Salvo: {summary_path}")
