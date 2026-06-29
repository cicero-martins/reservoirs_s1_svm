"""Reprocess downloaded WL CSVs to fix statistics (DAHITI v2: datetime/wse fields)."""
import csv, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

WL_DIR  = Path('validation_data/DAHITI/dahiti_longlist_wl')
SRC_CSV = Path('validation_data/DAHITI/dahiti_longlist_screening.csv')

# Load existing results
with open(SRC_CSV, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

# Build WL stats from downloaded CSVs
stats = {}
for wl_file in WL_DIR.glob('*_wl.csv'):
    dahiti_id = wl_file.name.split('_')[0]
    with open(wl_file, encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    if not data:
        continue
    dates = sorted(str(d.get('datetime', ''))[:10] for d in data if d.get('datetime'))
    d2014 = [dt for dt in dates if dt >= '2014-01-01']
    wse_vals = []
    for d in data:
        try:
            wse_vals.append(float(d['wse']))
        except (ValueError, KeyError):
            pass
    stats[dahiti_id] = {
        'wl_total':      len(data),
        'wl_2014':       len(d2014),
        'wl_min':        round(min(wse_vals), 2) if wse_vals else '',
        'wl_max':        round(max(wse_vals), 2) if wse_vals else '',
        'wl_range_m':    round(max(wse_vals) - min(wse_vals), 2) if wse_vals else '',
        'wl_date_start': dates[0]  if dates else '',
        'wl_date_end':   dates[-1] if dates else '',
    }

# Merge stats into rows
for row in rows:
    did = str(row.get('dahiti_id', '')).strip()
    if did and did in stats:
        row.update(stats[did])

# Save corrected CSV
fields = ['name', 'country', 'continent', 'biome', 'area_ha_est', 'in_pilot',
          'lat', 'lon', 'dahiti_id', 'dahiti_name', 'dahiti_type', 'dist_km',
          'wl_total', 'wl_2014', 'sa_total', 'sa_2014',
          'wl_min', 'wl_max', 'wl_range_m', 'wl_date_start', 'wl_date_end']
with open(SRC_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

# Print summary — only DAHITI-matched, sorted by continent
matched = [r for r in rows if r.get('dahiti_id')]
print(f"{'Name':28s}  {'Cont':5s}  {'Biome':22s}  {'Area(ha)':>8s}  "
      f"{'DAHITI':>7s}  {'Type':10s}  {'Dist':>5s}  "
      f"{'WL-tot':>6s}  {'WL-2014':>7s}  {'Range(m)':>9s}  "
      f"{'Period':>22s}  {'Pilot':>5s}")
print('-' * 165)

for r in sorted(matched, key=lambda x: (x['continent'], x['name'])):
    period = (f"{r['wl_date_start'][:7]} -> {r['wl_date_end'][:7]}"
              if r.get('wl_date_start') else '—')
    rng = (f"{r['wl_range_m']} m" if r.get('wl_range_m') != '' else '—')
    pilot_flag = '✓' if str(r['in_pilot']) == 'True' else ' '
    wl_2014 = int(r.get('wl_2014', 0) or 0)
    marker = '★' if wl_2014 >= 10 else (' ' if wl_2014 == 0 else '·')
    print(f"{marker} {r['name']:27s}  {r['continent']:5s}  {r['biome']:22s}  "
          f"{int(r['area_ha_est']):8d}  {r['dahiti_id']:>7s}  "
          f"{r['dahiti_type']:10s}  {float(r['dist_km']):5.1f}  "
          f"{int(r.get('wl_total',0) or 0):6d}  {wl_2014:7d}  "
          f"{rng:>9s}  {period:>22s}  {pilot_flag:>5s}")

wl_ok   = [r for r in matched if int(r.get('wl_2014', 0) or 0) >= 10]
wl_ok_n = [r for r in wl_ok   if str(r['in_pilot']) != 'True']
print(f"\n★ = WL 2014+ >= 10 obs  |  Matched: {len(matched)}  "
      f"|  ★ total: {len(wl_ok)}  ({len(wl_ok_n)} new candidates)")
print(f"Saved: {SRC_CSV}")
