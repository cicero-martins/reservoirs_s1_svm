"""Check Flag_Storage distribution for GROWL pilot reservoirs."""
import csv, io, zipfile
from pathlib import Path

BASE      = Path('F:/reservoirs_s1_svm/validation_data')
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'
PILOT_CSV = BASE / 'GROWL_pilot_sample.csv'

with open(PILOT_CSV, encoding='utf-8-sig') as f:
    pilot = [r for r in csv.DictReader(f) if r.get('note') == 'GEE_process']

# Sample: one per region
check_ids = ['2332', '2376', '2408',   # Spain
             '1240', '1268',            # USA
             '3938', '3936',            # Australia
             '3345', '3344',            # India
             '2760']                    # Belgium

with zipfile.ZipFile(GROWL_ZIP) as z:
    for res_id in check_ids:
        match = next((r for r in pilot if r['RES_ID'] == res_id), None)
        name = match['Name'] if match else '?'
        try:
            content = z.read(f'{res_id}.csv').decode('utf-8')
            rows = list(csv.DictReader(io.StringIO(content)))
            flag_dist = {}
            stor_any = 0
            for rw in rows:
                flag = rw.get('Flag_Storage', 'N/A').strip()
                stor = rw.get('Storage', '').strip()
                has_val = stor not in ('', 'nan', 'NaN')
                key = f'F{flag}'
                flag_dist[key] = flag_dist.get(key, 0) + 1
                if has_val:
                    stor_any += 1
            print(f'[{res_id}] {name:30s}: rows={len(rows):4d}  stor_any={stor_any:4d}  {flag_dist}')
        except KeyError:
            print(f'[{res_id}] NOT IN ZIP')
