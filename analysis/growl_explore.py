import csv
from collections import Counter

with open('F:/reservoirs_s1_svm/validation_data/GROWL_Metadata.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

# Basic stats
has_storage  = [r for r in rows if r['Storage_Periods'].strip()]
insitu_stor  = [r for r in has_storage if float(r.get('In-situ_Storage_Ratio','0') or 0) >= 0.5]
insitu_level = [r for r in rows if float(r.get('In-situ_Level_Ratio','0') or 0) >= 0.5]

print(f"Total GROWL:                {len(rows)}")
print(f"  com dados de Storage:     {len(has_storage)}")
print(f"  in-situ Storage ≥50%:     {len(insitu_stor)}")
print(f"  in-situ Level ≥50%:       {len(insitu_level)}")

# Country distribution for in-situ storage records (top 20)
country_counts = Counter(r['Country'] for r in insitu_stor)
print(f"\nTop 20 países — in-situ Storage ≥50%:")
for country, n in country_counts.most_common(20):
    print(f"  {country:30s}: {n}")

# Mediterranean / semi-arid countries of interest
med_countries = ['Spain','France','Portugal','Turkey','Morocco','Algeria','Tunisia',
                 'Greece','Croatia','Serbia','Bulgaria','Romania','Iran','Iraq',
                 'Syria','Lebanon','Jordan','Algeria','Libya','Egypt','Australia',
                 'Chile','Argentina','Mexico','USA','Brazil','India','China']

print(f"\n--- Reservatórios In-Situ Storage ≥50% em países mediterrâneos/similares ---")
med_recs = [r for r in insitu_stor if r['Country'] in med_countries]
print(f"Total: {len(med_recs)}")

# Filter: storage record >= 5 years (more robust)
long_recs = [r for r in med_recs
             if float(r.get('Storage_Record_Length_Years','0') or 0) >= 5]
print(f"  com ≥5 anos de storage:   {len(long_recs)}")

# Show countries of long_recs
print(f"\nDistribuição (≥5 anos storage, países mediterrâneos/similares):")
for country, n in Counter(r['Country'] for r in long_recs).most_common():
    print(f"  {country:30s}: {n}")

# Show records with HydroLAKES match (needed for A/P computation)
with_hydro = [r for r in long_recs if r.get('Hylak_ID','').strip()]
print(f"\n  com HydroLAKES match:     {len(with_hydro)}")
print(f"  sem HydroLAKES match:     {len(long_recs) - len(with_hydro)}")

# Show a sample for Mediterranean region (lat 28-48, lon -10-40)
med_geo = [r for r in long_recs
           if 28 <= float(r['Latitude']) <= 48
           and -10 <= float(r['Longitude']) <= 40]
print(f"\nReservatórios mediterrâneos geográficos (lat 28-48, lon -10-40, ≥5 anos storage):")
print(f"Total: {len(med_geo)}")
print(f"\n{'ID':>6}  {'Name':35s}  {'Country':15s}  {'Lat':>7}  {'Lon':>7}  {'Yrs_Stor':>8}  {'Hylak_ID':>10}")
print('-'*100)
for r in sorted(med_geo, key=lambda x: x['Country']):
    yrs = r.get('Storage_Record_Length_Years','').strip()
    hid = r.get('Hylak_ID','').strip()
    print(f"{r['RES_ID']:>6}  {r['Name']:35s}  {r['Country']:15s}  {float(r['Latitude']):7.3f}  "
          f"{float(r['Longitude']):7.3f}  {yrs:>8}  {hid:>10}")
