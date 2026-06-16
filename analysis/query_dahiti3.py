"""
DAHITI v2 — full coverage check with format=json fix.
For each pilot + Sicilian reservoir:
  1. list-targets bbox → find best match + data_access
  2. download all available endpoints (WL, Vol, SA) with format=json
  3. Save CSVs + print date ranges and point counts
"""
import json, time, csv, math, requests, urllib3
from pathlib import Path

urllib3.disable_warnings()
API_KEY  = '510183B30DFFC0A80C004524BABA85C6EFE0ECD67998D7F9E85EFA06930AA375'
BASE_URL = 'https://dahiti.dgfi.tum.de/api/v2'
BASE     = Path('F:/reservoirs_s1_svm/validation_data')
OUT_DIR  = BASE / 'DAHITI'
OUT_DIR.mkdir(parents=True, exist_ok=True)

S = requests.Session(); S.verify = False

DOWNLOAD_ENDPOINTS = {'download-water-level', 'download-volume-variation',
                      'download-surface-area', 'download-hypsometry',
                      'download-water-level-hypsometry'}

def post(endpoint, params):
    params['api_key'] = API_KEY
    if endpoint in DOWNLOAD_ENDPOINTS:
        params.setdefault('format', 'json')
    try:
        r = S.post(f'{BASE_URL}/{endpoint}/', data=params, timeout=30)
        return r.json()
    except Exception as e:
        return {'code': -1, 'message': str(e), 'data': None}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def save_csv(data, path):
    if not data: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

def download_all(dahiti_id, name_slug):
    """Download WL, Vol, SA for a DAHITI ID. Returns dict with counts and date ranges."""
    res = {}
    endpoints = [
        ('download-water-level',      'wl',  'wse'),
        ('download-volume-variation', 'vol', 'volume'),
        ('download-surface-area',     'sa',  'wsa'),
    ]
    for ep, key, val_field in endpoints:
        r = post(ep, {'dahiti_id': dahiti_id})
        data = r.get('data') or []
        if data:
            # Dates may be in 'datetime' or 'date' field
            dates = [d.get('datetime', d.get('date',''))[:10] for d in data]
            dates.sort()
            n_all = len(data)
            n_14  = sum(1 for d in dates if d >= '2014-01-01')
            date_range = f"{dates[0]} → {dates[-1]}" if dates else ''
            out = OUT_DIR / f"{dahiti_id}_{name_slug}_{key}.csv"
            save_csv(data, out)
        else:
            n_all = n_14 = 0
            date_range = ''
        res[key] = {'total': n_all, 'n2014': n_14, 'range': date_range}
        time.sleep(0.15)
    return res

# ── Reservoir list ─────────────────────────────────────────────────
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'
with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot_rows = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

reservoirs = []
for r in pilot_rows:
    reservoirs.append({'name': r['Name'], 'lat': float(r['Latitude']),
                       'lon': float(r['Longitude']), 'group': 'pilot',
                       'country': r.get('Country','')})
for item in [('Poma',37.9945,13.090,0.942),('Rosamarina',37.9435,13.640,0.889),
             ('Pozzillo',37.700,14.530,0.969),('Ancipa',37.830,14.573,0.808)]:
    reservoirs.append({'name': item[0], 'lat': item[1], 'lon': item[2],
                       'group': 'anchor', 'kge_planet': item[3], 'country': 'Italy'})

# ── Query ────────────────────────────────────────────────────────────
BBOX = 0.5; DIST_MAX = 25.0
results = []

print(f"\n{'Name':28s}  {'ID':>7}  {'Dist':>5}  "
      f"{'WL_tot':>7} {'WL_14':>6}  {'Vol_tot':>7} {'Vol_14':>6}  {'SA_tot':>7} {'SA_14':>6}  Range WL")
print('─'*115)

for res in reservoirs:
    lat, lon = res['lat'], res['lon']
    r1 = post('list-targets', {'min_lat': lat-BBOX, 'max_lat': lat+BBOX,
                                'min_lon': lon-BBOX, 'max_lon': lon+BBOX})
    targets = r1.get('data') or []

    best, best_dist = None, 9999
    for t in targets:
        d = haversine(lat, lon, float(t['latitude']), float(t['longitude']))
        if d < best_dist:
            best_dist, best = d, t

    if best is None or best_dist > DIST_MAX:
        print(f"{res['name']:28s}  {'—':>7}  {best_dist:5.1f}  {'—':>7} {'—':>6}  "
              f"{'—':>7} {'—':>6}  {'—':>7} {'—':>6}  ({len(targets)} in box)")
        results.append({'name': res['name'], 'group': res['group'],
                        'country': res['country'], 'dahiti_id': '',
                        'dist_km': round(best_dist, 1),
                        'wl_total':0,'wl_2014':0,'vol_total':0,'vol_2014':0,'sa_total':0,'sa_2014':0})
        continue

    did  = best['dahiti_id']
    slug = res['name'].replace(' ','_').replace('\\','_')[:20]
    da   = best.get('data_access') or {}

    d = download_all(did, slug)
    wl, vol, sa = d['wl'], d['vol'], d['sa']

    flag = ('✓WL' if wl['n2014']>0 else '') + (' ✓Vol' if vol['n2014']>0 else '')
    print(f"{res['name']:28s}  {str(did):>7}  {best_dist:5.1f}  "
          f"{wl['total']:>7} {wl['n2014']:>6}  {vol['total']:>7} {vol['n2014']:>6}  "
          f"{sa['total']:>7} {sa['n2014']:>6}  {wl['range'][:22] if wl['range'] else '—'}  {flag}")

    results.append({'name': res['name'], 'group': res['group'], 'country': res['country'],
                    'kge_planet': res.get('kge_planet',''),
                    'dahiti_id': did, 'dahiti_name': best['target_name'],
                    'dist_km': round(best_dist,1), 'type': best.get('type',''),
                    'wl_total': wl['total'], 'wl_2014': wl['n2014'],
                    'vol_total': vol['total'], 'vol_2014': vol['n2014'],
                    'sa_total': sa['total'], 'sa_2014': sa['n2014'],
                    'wl_range': wl['range'], 'vol_range': vol['range']})
    time.sleep(0.3)

# ── Summary ───────────────────────────────────────────────────────
fields = ['name','group','country','kge_planet','dahiti_id','dahiti_name','dist_km','type',
          'wl_total','wl_2014','vol_total','vol_2014','sa_total','sa_2014','wl_range','vol_range']
out_sum = OUT_DIR / 'dahiti_pilot_coverage_v2.csv'
with open(out_sum,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(results)

wl_ok  = [r for r in results if r.get('wl_2014',0)>0]
vol_ok = [r for r in results if r.get('vol_2014',0)>0]
print(f"\n{'='*70}")
print(f"Total consultados: {len(results)}")
print(f"  WL 2014+:  {len(wl_ok)}  → {[r['name'] for r in wl_ok]}")
print(f"  Vol 2014+: {len(vol_ok)}  → {[r['name'] for r in vol_ok]}")
print(f"Salvo: {out_sum}")
