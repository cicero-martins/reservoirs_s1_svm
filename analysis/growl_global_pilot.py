"""
Selects a stratified global pilot sample from GROWL for A/P × SAR-quality study.
Criteria:
  - in-situ storage ratio >= 0.5  (reference is real gauge, not satellite)
  - storage record >= 10 years    (robust seasonal signal)
  - HydroLAKES ID present         (needed for A/P geometry and GEE AOI)
  - GDW/GRanD ID present          (needed for GRDL AEV curve → volume validation)
  - Sentinel-1 coverage (lat -60 to 83)
Output: ranked global summary + pilot CSV
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

CSV = Path('F:/reservoirs_s1_svm/validation_data/GROWL_Metadata.csv')
OUT = Path('F:/reservoirs_s1_svm/validation_data/GROWL_pilot_sample.csv')

with open(CSV, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

def flt(r, key, default=0.0):
    v = r.get(key, '').strip()
    try: return float(v)
    except: return default

# ── Full pipeline requirements ────────────────────────────────────────────────
candidates = []
for r in rows:
    insitu_stor = flt(r, 'In-situ_Storage_Ratio')
    yrs_stor    = flt(r, 'Storage_Record_Length_Years')
    lat, lon    = flt(r, 'Latitude'), flt(r, 'Longitude')
    hylak       = r.get('Hylak_ID', '').strip()
    gdw         = r.get('GDW_ID', '').strip()

    if (insitu_stor >= 0.5        # real gauge data
        and yrs_stor >= 10        # robust record
        and hylak                 # geometry for A/P + GEE AOI
        and gdw                   # GRDL AEV linkage
        and -60 <= lat <= 83      # S1 coverage
    ):
        candidates.append(r)

country_counts = Counter(r['Country'] for r in candidates)
total = len(candidates)

print(f"=== GROWL Full-Pipeline Candidates ===")
print(f"Total eligible (in-situ ≥50%, ≥10yr, HydroLAKES, GRanD, S1 lat): {total}\n")
print(f"{'Country':30s}  {'N':>5}  {'%':>5}")
print('-' * 45)
for country, n in country_counts.most_common():
    print(f"{country:30s}  {n:5d}  {100*n/total:5.1f}%")

# ── Relaxed (no GDW requirement — AEV from literature) ───────────────────────
relaxed = []
for r in rows:
    insitu_stor = flt(r, 'In-situ_Storage_Ratio')
    yrs_stor    = flt(r, 'Storage_Record_Length_Years')
    lat         = flt(r, 'Latitude')
    hylak       = r.get('Hylak_ID', '').strip()
    if (insitu_stor >= 0.5 and yrs_stor >= 10 and hylak and -60 <= lat <= 83):
        relaxed.append(r)

print(f"\n=== Relaxed (no GRanD required) ===")
print(f"Total: {len(relaxed)}")
country_rel = Counter(r['Country'] for r in relaxed)
for country, n in country_rel.most_common(15):
    print(f"  {country:28s}: {n}")

# ── Stratified sample: aim for geographic spread ─────────────────────────────
# Pick top N from each major region, prioritizing longest records
REGIONS = {
    'Europe':       ['Spain', 'France', 'Portugal', 'Italy', 'Greece',
                     'Romania', 'Bulgaria', 'Serbia', 'Croatia', 'Hungary',
                     'Czech Republic', 'Slovakia', 'Austria', 'Germany',
                     'United Kingdom', 'Ireland', 'Norway', 'Sweden',
                     'Finland', 'Switzerland', 'Poland', 'Belgium',
                     'Netherlands', 'Denmark'],
    'N_America':    ['USA', 'Canada', 'Mexico'],
    'S_America':    ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru',
                     'Venezuela', 'Bolivia', 'Ecuador'],
    'Asia':         ['India', 'China', 'Turkey', 'Iran', 'Pakistan',
                     'Japan', 'South Korea', 'Thailand', 'Vietnam',
                     'Bangladesh', 'Nepal', 'Sri Lanka'],
    'Africa':       ['South Africa', 'Morocco', 'Algeria', 'Tunisia',
                     'Egypt', 'Kenya', 'Ethiopia', 'Nigeria', 'Ghana',
                     'Zimbabwe', 'Zambia', 'Tanzania'],
    'Oceania':      ['Australia', 'New Zealand'],
    'Middle_East':  ['Iraq', 'Syria', 'Jordan', 'Lebanon', 'Israel',
                     'Saudi Arabia', 'Yemen', 'Oman', 'UAE'],
    'Cyprus':       ['Cyprus'],
}

# Assign region to each candidate
def get_region(country):
    for region, countries in REGIONS.items():
        if country in countries:
            return region
    return 'Other'

by_region = defaultdict(list)
for r in candidates:
    by_region[get_region(r['Country'])].append(r)

# Sort each region by storage record length (longest first)
for reg in by_region:
    by_region[reg].sort(key=lambda x: -flt(x, 'Storage_Record_Length_Years'))

# Pick up to N_PER_REGION from each (prioritize diversity within region by country)
N_PER_REGION = 6
pilot = []
for reg in sorted(by_region):
    recs = by_region[reg]
    seen_countries = set()
    selected = []
    # First pass: one per country
    for r in recs:
        if r['Country'] not in seen_countries and len(selected) < N_PER_REGION:
            selected.append(r)
            seen_countries.add(r['Country'])
    # Second pass: fill remaining slots
    for r in recs:
        if len(selected) >= N_PER_REGION:
            break
        if r not in selected:
            selected.append(r)
    pilot.extend(selected[:N_PER_REGION])
    print(f"\n{reg} ({len(recs)} eligible → {len(selected[:N_PER_REGION])} selected):")
    for r in selected[:N_PER_REGION]:
        yrs = flt(r, 'Storage_Record_Length_Years')
        print(f"  [{r['RES_ID']:>4}] {r['Name']:35s} {r['Country']:15s}  "
              f"{flt(r,'Latitude'):7.3f} {flt(r,'Longitude'):7.3f}  "
              f"{yrs:5.1f}yr  Hylak={r['Hylak_ID']:>10}  GDW={r['GDW_ID']:>8}")

print(f"\n=== PILOT TOTAL: {len(pilot)} reservoirs ===")

# Add the 4 Sicilian validated reservoirs as anchors
# (already processed — skip GEE for these, use existing KGE values)
sicilian_anchors = [
    {'RES_ID':'SIC1','Name':'Poma','Country':'Italy','Latitude':'37.995',
     'Longitude':'13.090','Hylak_ID':'SICILIA','GDW_ID':'SICILIA',
     'Storage_Record_Length_Years':'10','In-situ_Storage_Ratio':'1',
     'note':'Anchor — KGE=0.942 (PlanetScope)'},
    {'RES_ID':'SIC2','Name':'Rosamarina','Country':'Italy','Latitude':'37.944',
     'Longitude':'13.640','Hylak_ID':'SICILIA','GDW_ID':'SICILIA',
     'Storage_Record_Length_Years':'10','In-situ_Storage_Ratio':'1',
     'note':'Anchor — KGE=0.889 (PlanetScope)'},
    {'RES_ID':'SIC3','Name':'Pozzillo','Country':'Italy','Latitude':'37.700',
     'Longitude':'14.530','Hylak_ID':'SICILIA','GDW_ID':'SICILIA',
     'Storage_Record_Length_Years':'10','In-situ_Storage_Ratio':'1',
     'note':'Anchor — KGE=0.969 (PlanetScope)'},
    {'RES_ID':'SIC4','Name':'Ancipa','Country':'Italy','Latitude':'37.830',
     'Longitude':'14.573','Hylak_ID':'SICILIA','GDW_ID':'SICILIA',
     'Storage_Record_Length_Years':'10','In-situ_Storage_Ratio':'1',
     'note':'Anchor — KGE=0.808 (PlanetScope)'},
]

# Save pilot CSV
all_cols = list(rows[0].keys()) + ['note']
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
    w.writeheader()
    for r in pilot:
        r['note'] = 'GEE_process'
        w.writerow(r)
    for r in sicilian_anchors:
        w.writerow(r)

print(f"\nSalvo: {OUT}  ({len(pilot)} GEE + 4 Sicilian anchors = {len(pilot)+4} total)")
