"""
gee_eecu_report.py

Pulls per-task EECU-seconds (batchEecuUsageSeconds) from the Earth Engine operations
API and tabulates computational cost by method, for the cost-discussion section.

EECU-seconds (Earth Engine Compute Unit-seconds) is GEE's hardware-independent measure
of batch compute — the correct metric for a cost comparison (unlike wall-clock, which
depends on server load and queueing).

Our export tasks are named:
  SAR_area_<name>           → dual-pol VV+VH SVM   (Tier 3)
  SAR_area_<name>_VVotsu    → single-pol VV Otsu   (Tier 1)
  JRC_area_<name>           → JRC reference (classifier-independent)
  Era5Wind_<name>           → ERA5 wind
so we can split cost cleanly by the task description suffix.

Requires the Earth Engine PYTHON API authenticated once:
    pip install earthengine-api
    earthengine authenticate
(The Code Editor / eetasks auth does NOT carry over to Python.)

Output: analysis/gee_eecu_costs.csv  +  printed summary.

NOTE on fair comparison: the SVM run currently also exports JRC (EXPORT_JRC=true),
the VV run does not. To compare *classifier* cost apples-to-apples, either subtract the
JRC_area_* cost from the SVM total, or re-run the SVM batch with EXPORT_JRC=false for the
cost measurement. This script reports JRC separately so you can do the subtraction.
"""

import csv
import sys
import re
from pathlib import Path

try:
    import ee
except ImportError:
    sys.exit('earthengine-api not installed. Run: pip install earthengine-api')

# Set your cloud project if needed: ee.Initialize(project='your-project')
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

OUT_CSV = Path('analysis/gee_eecu_costs.csv')


def classify_task(desc):
    """Map a task description to (method, reservoir)."""
    if desc is None:
        return None, None
    m = re.match(r'SAR_area_(.+?)(_VVotsu|_VVfast)?$', desc)
    if m:
        suffix = m.group(2)
        method = {'_VVotsu': 'vv_otsu', '_VVfast': 'vv_fast'}.get(suffix, 'svm_dual')
        return method, m.group(1)
    m = re.match(r'JRC_area_(.+)$', desc)
    if m:
        return 'jrc', m.group(1)
    m = re.match(r'Era5Wind_(.+)$', desc)
    if m:
        return 'era5_wind', m.group(1)
    return 'other', desc


# ── Pull operations ────────────────────────────────────────────────────────────
ops = ee.data.listOperations()
print(f'Fetched {len(ops)} operations from EE.\n')

rows = []
for op in ops:
    md   = op.get('metadata', {})
    desc = md.get('description')
    method, reservoir = classify_task(desc)
    if method in (None, 'other'):
        continue
    eecu = md.get('batchEecuUsageSeconds')
    rows.append({
        'description':  desc,
        'method':       method,
        'reservoir':    reservoir,
        'state':        md.get('state'),
        'eecu_seconds': float(eecu) if eecu is not None else None,
        'start':        md.get('startTime'),
        'end':          md.get('endTime'),
    })

if not rows:
    sys.exit('No matching SAR/JRC/Era5Wind tasks found in operations history.')

with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'Saved {len(rows)} tasks -> {OUT_CSV}\n')

# ── Summary by method ──────────────────────────────────────────────────────────
print(f'{"method":<12} {"n_tasks":>8} {"n_with_eecu":>12} {"total_eecu_s":>14} {"mean_eecu_s":>12}')
print('-' * 62)
for method in ['svm_dual', 'vv_otsu', 'vv_fast', 'jrc', 'era5_wind']:
    sub  = [r for r in rows if r['method'] == method]
    vals = [r['eecu_seconds'] for r in sub if r['eecu_seconds'] is not None]
    total = sum(vals) if vals else 0.0
    mean  = (total / len(vals)) if vals else 0.0
    print(f'{method:<12} {len(sub):>8} {len(vals):>12} {total:>14.1f} {mean:>12.1f}')

# ── Head-to-head per reservoir (svm vs vv) ────────────────────────────────────
svm = {r['reservoir']: r['eecu_seconds'] for r in rows
       if r['method'] == 'svm_dual' and r['eecu_seconds'] is not None}
vv  = {r['reservoir']: r['eecu_seconds'] for r in rows
       if r['method'] == 'vv_otsu' and r['eecu_seconds'] is not None}
common = sorted(set(svm) & set(vv))
if common:
    print(f'\n{"reservoir":<22} {"svm_eecu":>10} {"vv_eecu":>10} {"ratio svm/vv":>13}')
    print('-' * 57)
    ratios = []
    for name in common:
        ratio = svm[name] / vv[name] if vv[name] else float('nan')
        ratios.append(ratio)
        print(f'{name:<22} {svm[name]:>10.1f} {vv[name]:>10.1f} {ratio:>13.2f}')
    print('-' * 57)
    print(f'Median cost ratio SVM / VV-Otsu = {sorted(ratios)[len(ratios)//2]:.2f}x')
else:
    print('\n[no head-to-head yet] run both SVM and VV_OTSU exports, then re-run.')
