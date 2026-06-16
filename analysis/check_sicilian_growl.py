import csv, io, zipfile
from pathlib import Path

BASE = Path('F:/reservoirs_s1_svm/validation_data')
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'

for res_id, name in [('2836', 'Poma'), ('2848', 'Scalzano/Rosamarina')]:
    with zipfile.ZipFile(GROWL_ZIP) as z:
        content = z.read(f'{res_id}.csv').decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        stor = [r for r in rows
                if r.get('Storage','').strip() not in ('','nan','NaN')
                and r.get('Date','') >= '2014-01-01']
        print(f'RES_ID {res_id} ({name}): total={len(rows)}, stor_2014+={len(stor)}')
        for r in stor[:4]:
            print(f"  {r['Date']}  Storage={r['Storage']}  Flag={r['Flag_Storage']}")
        print()
