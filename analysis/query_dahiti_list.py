"""
Use list-targets with bounding box to find DAHITI entries for our reservoirs,
then check data_access fields. Also test download-water-level for found targets.
"""
import json, ssl, csv, time
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
OUT_DIR  = Path('F:/reservoirs_s1_svm/validation_data/DAHITI')
OUT_DIR.mkdir(parents=True, exist_ok=True)

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode   = ssl.CERT_NONE

def api_get(endpoint, params):
    params['api_key'] = API_KEY
    qs  = urllib.parse.urlencode(params)
    url = f'{BASE_URL}/{endpoint}/?{qs}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

# ── 1. List all targets in Sicily ─────────────────────────────────
print("=== DAHITI targets in Sicily (lat 37-39, lon 12-15.5) ===")
r = api_get('list-targets', {
    'min_lat': 37.0, 'max_lat': 39.0,
    'min_lon': 12.0, 'max_lon': 15.5
})
sicily_targets = r.get('data', []) or []
print(f"Found {len(sicily_targets)} targets\n")
for t in sicily_targets:
    da = t.get('data_access', {}) or {}
    print(f"  [{t['dahiti_id']:>6}] {t['target_name']:35s}  "
          f"lat={t['latitude']:7.3f}  lon={t['longitude']:7.3f}  "
          f"WL={da.get('water_level_altimetry','null'):6s}  "
          f"SA={da.get('surface_area','null'):6s}  "
          f"type={t.get('type','?')}")

# ── 2. Try download-water-level for Poma + Rosamarina ─────────────
print("\n=== Water level for Poma (42134) and Scalzano/Rosamarina (42122) ===")
for dahiti_id, name in [(42134, 'Poma'), (42122, 'Scalzano/Rosamarina')]:
    r = api_get('download-water-level', {'dahiti_id': dahiti_id})
    if r.get('code') == 200 and r.get('data'):
        data = r['data']
        d2014 = [d for d in data if str(d.get('date','')) >= '2014-01-01']
        print(f"  [{dahiti_id}] {name}: total={len(data)}, 2014+={len(d2014)}")
        for d in d2014[:3]:
            print(f"    {d}")
        # Save
        out = OUT_DIR / f'{dahiti_id}_{name}_WL.csv'
        with open(out, 'w', newline='', encoding='utf-8') as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
        print(f"    Saved: {out}")
    else:
        print(f"  [{dahiti_id}] {name}: no data — {r.get('message','')}")

# ── 3. Check a few global pilot reservoirs more carefully ──────────
print("\n=== Bounding box search for select pilot reservoirs ===")
bbox_tests = [
    ('Ricobayo ES',    41.5, 41.6, -6.1, -5.9),
    ('Elephant Butte', 33.1, 33.2, -107.3, -107.1),
    ('Manikdoh IN',    19.2, 19.4, 73.7, 73.9),
    ('Maroondah AU',  -37.7, -37.6, 145.5, 145.6),
]
for label, min_lat, max_lat, min_lon, max_lon in bbox_tests:
    r = api_get('list-targets', {
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_lon': min_lon, 'max_lon': max_lon
    })
    targets = r.get('data', []) or []
    print(f"\n  {label}: {len(targets)} targets")
    for t in targets:
        da = t.get('data_access', {}) or {}
        print(f"    [{t['dahiti_id']:>6}] {t['target_name']:35s}  "
              f"SA={da.get('surface_area','null'):6s}  WL={da.get('water_level_altimetry','null')}")
    time.sleep(0.3)
