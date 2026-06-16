"""
DAHITI v2 — targeted test:
1. download-water-level + download-volume-variation for known IDs (Poma 42134, Rosamarina 42122)
2. list-targets with wider bbox (±0.5°) for all 24 global pilot reservoirs
3. For any found target: try all three endpoints (SA, WL, Vol)
"""
import json, time, csv, math, requests, urllib3
from pathlib import Path

urllib3.disable_warnings()
API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
BASE     = Path('F:/reservoirs_s1_svm/validation_data')
OUT_DIR  = BASE / 'DAHITI'
OUT_DIR.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.verify = False

def post(endpoint, params):
    params['api_key'] = API_KEY
    try:
        r = S.post(f'{BASE_URL}/{endpoint}/', data=params, timeout=30)
        return r.json()
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

def save_csv(data, path):
    if not data: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def probe_target(dahiti_id, label):
    """Try all three data endpoints for a known DAHITI ID."""
    results = {}
    for ep, key in [('download-surface-area','wsa'), ('download-water-level','water_level'),
                    ('download-volume-variation','volume')]:
        r = post(ep, {'dahiti_id': dahiti_id})
        data = r.get('data') or []
        n2014 = len([d for d in data if str(d.get('date','')) >= '2014-01-01'])
        results[ep] = {'total': len(data), '2014+': n2014}
        if data:
            out = OUT_DIR / f"{dahiti_id}_{label}_{ep.split('-')[-1]}.csv"
            save_csv(data, out)
    print(f"\n  [{dahiti_id}] {label}")
    print(f"    surface-area:     total={results['download-surface-area']['total']:4d}  2014+={results['download-surface-area']['2014+']:4d}")
    print(f"    water-level:      total={results['download-water-level']['total']:4d}  2014+={results['download-water-level']['2014+']:4d}")
    print(f"    volume-variation: total={results['download-volume-variation']['total']:4d}  2014+={results['download-volume-variation']['2014+']:4d}")
    return results

# ── 1. Known Sicilian IDs ─────────────────────────────────────────
print("=== Sicilian reservoirs (known DAHITI IDs) ===")
probe_target(42134, 'Poma')
probe_target(42122, 'Rosamarina_Scalzano')

# ── 2. Wider search for global pilot reservoirs ───────────────────
print("\n=== Global pilot — wider bbox (±0.5°) ===")
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'
with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

BBOX = 0.5   # degrees
DIST = 25.0  # km max

summary = []
print(f"\n{'Name':30s}  {'DAHITI ID':>10}  {'Type':12s}  {'Dist':>6}  "
      f"{'SA':>4}  {'WL':>5}  {'Vol':>5}")
print('─'*80)

for res in pilot:
    lat, lon = float(res['Latitude']), float(res['Longitude'])
    r = post('list-targets', {
        'min_lat': lat-BBOX, 'max_lat': lat+BBOX,
        'min_lon': lon-BBOX, 'max_lon': lon+BBOX,
    })
    targets = r.get('data') or []

    # Pick closest reservoir/lake (not river)
    best, best_dist = None, 9999
    for t in targets:
        d = haversine(lat, lon, float(t['latitude']), float(t['longitude']))
        ttype = (t.get('type') or '').lower()
        if d < best_dist and any(kw in ttype for kw in ['reservoir','lake','lagoon','pond']):
            best_dist, best = d, t
    # If no reservoir found, take closest of any type
    if best is None:
        for t in targets:
            d = haversine(lat, lon, float(t['latitude']), float(t['longitude']))
            if d < best_dist:
                best_dist, best = d, t

    if best is None or best_dist > DIST:
        print(f"{res['Name']:30s}  {'—':>10}  {'no match':12s}  {best_dist if best else 0:6.1f}  "
              f"  —     —     —   ({len(targets)} in box)")
        summary.append({'name': res['Name'], 'dahiti_id': '', 'dist_km': best_dist or 0,
                        'sa_2014': 0, 'wl_2014': 0, 'vol_2014': 0})
        continue

    did  = best['dahiti_id']
    da   = best.get('data_access') or {}
    ttype = best.get('type','?')

    # Download all available endpoints
    sa_n = wl_n = vol_n = 0
    for ep, key, var in [('download-surface-area','wsa','sa'),
                         ('download-water-level','water_level','wl'),
                         ('download-volume-variation','volume','vol')]:
        r2 = post(ep, {'dahiti_id': did})
        data = r2.get('data') or []
        n14 = len([d for d in data if str(d.get('date','')) >= '2014-01-01'])
        if var == 'sa':  sa_n = n14
        if var == 'wl':  wl_n = n14
        if var == 'vol': vol_n = n14
        if data:
            out = OUT_DIR / f"{did}_{res['Name'].replace(' ','_')}_{var}.csv"
            save_csv(data, out)
        time.sleep(0.1)

    print(f"{res['Name']:30s}  {str(did):>10}  {ttype:12s}  {best_dist:6.1f}  "
          f"{sa_n:4d}  {wl_n:5d}  {vol_n:5d}")
    summary.append({'name': res['Name'], 'dahiti_id': did, 'dahiti_name': best['target_name'],
                    'dist_km': round(best_dist,2), 'sa_2014': sa_n, 'wl_2014': wl_n, 'vol_2014': vol_n})
    time.sleep(0.3)

# Save summary
out_sum = OUT_DIR / 'dahiti_pilot_coverage.csv'
with open(out_sum,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['name','dahiti_id','dahiti_name','dist_km','sa_2014','wl_2014','vol_2014'], extrasaction='ignore')
    w.writeheader(); w.writerows(summary)

wl_ok  = [r for r in summary if r.get('wl_2014',0) > 0]
vol_ok = [r for r in summary if r.get('vol_2014',0) > 0]
print(f"\n{'='*60}")
print(f"Piloto global: {len(summary)} reservatórios")
print(f"  com WL 2014+:  {len(wl_ok)}")
print(f"  com Vol 2014+: {len(vol_ok)}")
print(f"Salvo: {out_sum}")
