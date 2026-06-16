import csv, io, zipfile
from pathlib import Path

BASE = Path('F:/reservoirs_s1_svm/validation_data')
GROWL_ZIP = BASE / 'GROWL_Timeseries.zip'

for res_id, name in [('2836', 'Poma'), ('2848', 'Scalzano/Rosamarina')]:
    with zipfile.ZipFile(GROWL_ZIP) as z:
        content = z.read(f'{res_id}.csv').decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        print(f'RES_ID {res_id} ({name}): cols={list(rows[0].keys()) if rows else []}')
        # Show first 5 rows regardless
        for r in rows[:5]:
            print(f"  {dict(r)}")
        print()
