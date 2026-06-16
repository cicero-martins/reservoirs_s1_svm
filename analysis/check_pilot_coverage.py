"""Check date coverage for each pilot reservoir in GROWL."""
import csv, io, zipfile
from pathlib import Path

BASE = Path('F:/reservoirs_s1_svm/validation_data')
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'

with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

print(f"{'ID':>5}  {'Name':30s}  {'cols':8s}  {'stor_2014+':>10}  {'date_range':25s}")
print('-'*85)
with zipfile.ZipFile(GROWL_ZIP) as z:
    for r in pilot:
        res_id = r['RES_ID'].strip()
        try:
            content = z.read(f'{res_id}.csv').decode('utf-8-sig')
            rows = list(csv.DictReader(io.StringIO(content)))
            cols = list(rows[0].keys()) if rows else []
            has_stor = 'Storage' in cols
            if has_stor:
                stor = [rw for rw in rows
                        if rw.get('Storage','').strip() not in ('','nan','NaN')
                        and rw.get('Date','') >= '2014-01-01']
                dates = sorted(rw['Date'] for rw in stor)
                date_rng = f"{dates[0]} → {dates[-1]}" if dates else 'no data'
                n = len(stor)
            else:
                n = 0
                date_rng = 'LEVEL only'
            print(f"{res_id:>5}  {r['Name']:30s}  {'STOR' if has_stor else 'LEV ':4s}  {n:>10}  {date_rng:25s}")
        except KeyError:
            print(f"{res_id:>5}  {r['Name']:30s}  NOT IN ZIP")
