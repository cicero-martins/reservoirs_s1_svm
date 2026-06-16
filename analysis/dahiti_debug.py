"""Debug DAHITI API — print raw response for Poma."""
import requests, urllib3, json
urllib3.disable_warnings()

API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
S = requests.Session(); S.verify = False

def post(endpoint, params):
    params['api_key'] = API_KEY
    r = S.post(f'{BASE_URL}/{endpoint}/', data=params, timeout=30)
    print(f"\n[{endpoint}] HTTP {r.status_code}")
    try:
        j = r.json()
        # Print everything except the data array (might be large)
        summary = {k: v for k, v in j.items() if k != 'data'}
        summary['data_length'] = len(j.get('data') or [])
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if j.get('data'):
            print(f"  First item: {j['data'][0]}")
    except Exception as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw: {r.text[:500]}")
    return r

# Poma = 42134
print("=== list-targets for Poma bbox ===")
post('list-targets', {'min_lat': 37.9, 'max_lat': 38.1, 'min_lon': 12.9, 'max_lon': 13.2})

print("\n=== download-water-level Poma (42134) ===")
post('download-water-level', {'dahiti_id': 42134})

print("\n=== download-volume-variation Poma (42134) ===")
post('download-volume-variation', {'dahiti_id': 42134})

# Also check if 'format' param makes a difference
print("\n=== download-water-level Poma with format=json ===")
post('download-water-level', {'dahiti_id': 42134, 'format': 'json'})
