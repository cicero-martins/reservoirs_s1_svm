import csv, io, zipfile
from pathlib import Path

BASE = Path('F:/reservoirs_s1_svm/validation_data')
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'

for res_id, name in [('2836', 'Poma'), ('2848', 'Scalzano/Rosamarina')]:
    with zipfile.ZipFile(GROWL_ZIP) as z:
        content = z.read(f'{res_id}.csv').decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        stor = [r for r in rows if r.get('Storage','').strip() not in ('','nan','NaN')]
        dates = sorted(r['Date'] for r in stor)
        print(f'RES_ID {res_id} ({name}): total={len(rows)}, stor_any={len(stor)}')
        if dates:
            print(f'  Date range: {dates[0]} to {dates[-1]}')
            for r in stor[:3]:
                print(f"  {r['Date']}  Stor={r['Storage']}  Flag={r['Flag_Storage']}")
        print()
