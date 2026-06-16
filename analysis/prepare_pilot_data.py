"""
Extrai e organiza dados GRDL + GROWL para os 24 reservatórios piloto.
Produz:
  - validation_data/pilot/GRDL_curves/RES_ID_GRAND_ID.csv  (curva AEV)
  - validation_data/pilot/GROWL_series/RES_ID.csv          (série storage)
  - validation_data/pilot/pilot_summary.csv                 (metadados)
"""
import csv, io, zipfile
from pathlib import Path

BASE      = Path('F:/reservoirs_s1_svm/validation_data')
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'
GRDL_ZIP  = BASE / 'GRDL.zip'
OUT_DIR   = BASE / 'pilot'
(OUT_DIR / 'GRDL_curves').mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'GROWL_series').mkdir(parents=True, exist_ok=True)

# ── Read pilot list ──────────────────────────────────────────────
with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

print(f"Pilot reservoirs: {len(pilot)}")
print(f"{'RES_ID':>6}  {'Name':35s}  {'Country':15s}  {'GDW_ID':>8}  {'Hylak_ID':>12}")
print('-'*85)
for r in pilot:
    print(f"{r['RES_ID']:>6}  {r['Name']:35s}  {r['Country']:15s}  "
          f"{r['GDW_ID']:>8}  {r['Hylak_ID']:>12}")

# ── Extract GRDL curves ─────────────────────────────────────────
print("\n=== Extracting GRDL AEV curves ===")
missing_grdl = []
with zipfile.ZipFile(GRDL_ZIP) as z:
    for r in pilot:
        gdw_id = r.get('GDW_ID', '').strip()
        if not gdw_id or gdw_id == '0.0':
            missing_grdl.append(r['Name'])
            continue
        # GRDL filename: zero-padded 5-digit GRAND_ID
        grand_id = int(float(gdw_id))
        fname = f'{grand_id:05d}.csv'
        try:
            content = z.read(fname).decode('utf-8')
            out = OUT_DIR / 'GRDL_curves' / f"{r['RES_ID']}_{grand_id}.csv"
            out.write_text(content, encoding='utf-8')
            # Quick parse to verify
            lines = [l for l in content.splitlines() if ';' in l]
            print(f"  {r['Name']:30s}: GRAND_ID={grand_id:5d}  AEV rows={len(lines)}")
        except KeyError:
            missing_grdl.append(f"{r['Name']} (GRAND_ID={grand_id})")

if missing_grdl:
    print(f"\n  MISSING GRDL: {missing_grdl}")

# ── Extract GROWL time series ────────────────────────────────────
print("\n=== Extracting GROWL storage time series ===")
missing_growl = []
with zipfile.ZipFile(GROWL_ZIP) as z:
    for r in pilot:
        res_id = r['RES_ID'].strip()
        fname  = f'{res_id}.csv'
        try:
            content = z.read(fname).decode('utf-8')
            out = OUT_DIR / 'GROWL_series' / f'{res_id}.csv'
            out.write_text(content, encoding='utf-8')
            rows = list(csv.DictReader(io.StringIO(content)))
            # Count rows with storage data
            stor_rows = [rw for rw in rows
                         if rw.get('Storage','').strip()
                         and rw['Storage'] not in ('', 'nan', 'NaN')
                         and rw.get('Flag_Storage','0') == '0']  # 0 = original
            # Also count 2014+ rows
            from datetime import datetime
            stor_2014 = [rw for rw in stor_rows
                         if rw['Date'] >= '2014-01-01']
            print(f"  [{res_id:>4}] {r['Name']:30s}: total={len(rows):5d}  "
                  f"storage_orig={len(stor_rows):5d}  storage_2014+={len(stor_2014):4d}")
        except KeyError:
            missing_growl.append(f"{r['Name']} (RES_ID={res_id})")

if missing_growl:
    print(f"\n  MISSING GROWL: {missing_growl}")

# ── Check GROWL column names (first valid file) ──────────────────
print("\n=== GROWL timeseries column structure ===")
with zipfile.ZipFile(GROWL_ZIP) as z:
    first_id = pilot[0]['RES_ID'].strip()
    content  = z.read(f'{first_id}.csv').decode('utf-8')
    rows     = list(csv.DictReader(io.StringIO(content)))
    print(f"Columns: {list(rows[0].keys())}")
    print(f"First 3 rows:")
    for rw in rows[:3]:
        print(f"  {dict(rw)}")

# ── Summary CSV ──────────────────────────────────────────────────
summary_path = OUT_DIR / 'pilot_summary.csv'
fields = ['RES_ID','Name','Country','Latitude','Longitude','Hylak_ID','GDW_ID',
          'Storage_Record_Length_Years','In-situ_Storage_Ratio','Source']
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(pilot)
print(f"\nSalvo: {summary_path}")
