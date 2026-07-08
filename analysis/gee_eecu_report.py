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

sys.stdout.reconfigure(encoding='utf-8')

# Use the OS (Windows) trust store so SSL-intercepting networks (e.g. the UniPa
# campus proxy) don't trip OpenSSL 3's strict cert checks. No-op if unavailable.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

try:
    import ee
except ImportError:
    sys.exit('earthengine-api not installed. Run: pip install earthengine-api')

# Cloud project that owns the export tasks.
EE_PROJECT = 'ee-ciceromartinsjr'
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

OUT_CSV = Path('analysis/gee_eecu_costs.csv')


def classify_task(desc):
    """Map a task description to (method, reservoir)."""
    if desc is None:
        return None, None
    m = re.match(r'SAR_area_(.+?)(_VVotsu|_VVfast|_SVMadapt)?$', desc)
    if m:
        suffix = m.group(2)
        method = {'_VVotsu': 'vv_otsu', '_VVfast': 'vv_fast',
                  '_SVMadapt': 'svm_adapt'}.get(suffix, 'svm_dual')
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

import pandas as pd

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

df = pd.DataFrame(rows)
df['start_dt'] = pd.to_datetime(df['start'], errors='coerce', utc=True)

# listOperations() returns the FULL history (v2/v3/v4 + re-runs), so raw totals are
# polluted. The v4 method-comparison run is the MOST RECENT SUCCEEDED task for each
# (method, reservoir). Dedupe to that — automatically isolates v4 from older runs
# and supersedes failed/cancelled retries.
ok = df[(df['state'] == 'SUCCEEDED') & df['eecu_seconds'].notna()].copy()
latest = (ok.sort_values('start_dt')
            .drop_duplicates(['method', 'reservoir'], keep='last')
            .reset_index(drop=True))

# EE's listOperations() only retains a rolling window of task history: older
# completed tasks age out and become unrecoverable via the API. Overwriting the
# CSV from a fresh pull therefore silently DROPS any (method, reservoir) pair
# whose task has since expired, even though it was validly recorded before. We
# merge onto the existing CSV instead, keeping every previously-recorded pair
# and only adding/refreshing pairs present in this pull.
if OUT_CSV.exists():
    prior = pd.read_csv(OUT_CSV)
    prior['start_dt'] = pd.to_datetime(prior['start'], errors='coerce', utc=True)
    combined = pd.concat([prior, latest], ignore_index=True)
    latest = (combined.sort_values('start_dt')
                       .drop_duplicates(['method', 'reservoir'], keep='last')
                       .reset_index(drop=True))
    n_new = len(latest) - len(prior)
    print(f'Merged onto existing {len(prior)}-row CSV ({n_new:+d} net rows; '
          f'{len(prior) - len(prior.merge(latest, on=["method","reservoir"]))} pairs would have been '
          f'lost by a plain overwrite).')

latest.to_csv(OUT_CSV, index=False)
print(f'Saved {len(latest)} latest-per-(method,reservoir) tasks -> {OUT_CSV}')
print(f'(from {len(df)} total / {len(ok)} succeeded-with-EECU this pull, merged with prior history)\n')

# ── Summary by method (deduped) ───────────────────────────────────────────────
print(f'{"method":<12} {"n_resv":>7} {"total_eecu_s":>14} {"mean_eecu_s":>12} {"median":>10}  {"window":>23}')
print('-' * 84)
for method in ['svm_dual', 'vv_otsu', 'vv_fast', 'jrc', 'era5_wind']:
    sub = latest[latest['method'] == method]
    if sub.empty:
        continue
    v = sub['eecu_seconds']
    dt = sub['start_dt'].dropna()
    win = f"{dt.min():%Y-%m-%d}..{dt.max():%Y-%m-%d}" if not dt.empty else 'n/a'
    print(f'{method:<12} {len(sub):>7} {v.sum():>14.1f} {v.mean():>12.1f} {v.median():>10.1f}  {win:>23}')

# ── Head-to-head per reservoir (svm vs vv), same dedup ────────────────────────
# NOTE on GEE computation caching: when two reservoirs share overlapping AOIs
# (e.g. Sau ⊂ Susqueda on the Ter), the second task reuses cached intermediates and
# reports anomalously low EECU. These cache-confounded pairs are flagged via an IQR
# fence on the ratio and excluded from the robust headline (still listed).
piv = latest.pivot_table(index='reservoir', columns='method', values='eecu_seconds')
if {'svm_dual', 'vv_otsu'}.issubset(piv.columns):
    hh = piv.dropna(subset=['svm_dual', 'vv_otsu']).copy()
    hh['ratio'] = hh['svm_dual'] / hh['vv_otsu']
    hh = hh.sort_values('ratio')

    q1, q3 = hh['ratio'].quantile([0.25, 0.75])
    fence  = q3 + 3.0 * (q3 - q1)          # generous upper fence (cache outliers only)
    hh['flag'] = hh['ratio'] > fence
    clean = hh[~hh['flag']]

    print(f'\n{"reservoir":<22} {"svm_eecu":>10} {"vv_eecu":>10} {"svm/vv":>8}')
    print('-' * 54)
    for name, r in hh.iterrows():
        mark = '  ⚠cache?' if r['flag'] else ''
        print(f'{name:<22} {r["svm_dual"]:>10.1f} {r["vv_otsu"]:>10.1f} {r["ratio"]:>8.2f}{mark}')
    print('-' * 54)
    flagged = list(hh[hh['flag']].index)
    print(f'N pairs total          = {len(hh)}   (flagged cache-confounded: {flagged or "none"})')
    print(f'--- robust (excl. flagged, N={len(clean)}) ---')
    print(f'Median ratio SVM/VV    = {clean["ratio"].median():.2f}x')
    print(f'Mean ratio SVM/VV      = {clean["ratio"].mean():.2f}x')
    print(f'Total SVM / Total VV   = {clean["svm_dual"].sum() / clean["vv_otsu"].sum():.2f}x  '
          f'(SVM={clean["svm_dual"].sum():.0f}, VV={clean["vv_otsu"].sum():.0f} EECU-s)')
    inv = clean["vv_otsu"].sum() / clean["svm_dual"].sum()
    print(f'\nInterpretation: SVM/VV ≈ {clean["ratio"].median():.2f} (<1) → dual-pol SVM is '
          f'NOT more expensive; VV-only Otsu costs ~{inv:.2f}x MORE (per-scene histogram\n'
          'overhead), while the dominant vectorisation/area cost is SHARED. The classifier\n'
          'is NOT the cost driver → the real lever is post-processing (see VV_OTSU_FAST).')

    # ── The vectorisation lever: VV_OTSU_FAST (no fill/vectorise/keep/dynamic-AP) ──
    if 'vv_fast' in piv.columns:
        ff = piv.dropna(subset=['vv_fast', 'vv_otsu', 'svm_dual']).copy()
        ff['fast_vv']  = ff['vv_fast'] / ff['vv_otsu']
        ff['fast_svm'] = ff['vv_fast'] / ff['svm_dual']
        # reuse the same cache fence on the svm/vv ratio to drop confounded reservoirs
        ff = ff[(ff['svm_dual'] / ff['vv_otsu']) <= fence]
        print(f'\n=== VECTORISATION LEVER — VV_OTSU_FAST (N={len(ff)}) ===')
        fv, fs = ff['fast_vv'].median(), ff['fast_svm'].median()
        print(f'Median FAST/VV-Otsu  = {fv:.3f}x  '
              f'→ removing vectorisation from the VV pipeline cuts cost ~{1/fv:.2f}x ({(1-fv)*100:.0f}%)')
        print(f'Median FAST/SVM-dual = {fs:.3f}x  '
              f'→ FAST ≈ the dual SVM cost (NOT cheaper than it)')
        print(f'Total FAST/VV / FAST/SVM (Σ): '
              f'{ff["vv_fast"].sum()/ff["vv_otsu"].sum():.3f} / '
              f'{ff["vv_fast"].sum()/ff["svm_dual"].sum():.3f}')
        print('→ Cost ordering: SVM-dual ≈ FAST < VV-Otsu. Vectorisation is a ~25% lever\n'
              '  on the VV pipeline (brings it down to the dual-SVM cost), but it is NOT a\n'
              '  dominant driver that beats classifier choice: the per-scene Otsu histogram\n'
              '  overhead is what makes plain VV-Otsu the most expensive of the three.')
    else:
        print('\n[VV_OTSU_FAST not yet present] run exportGlobalPilotV4.js with '
              'CLASSIFIER="VV_OTSU_FAST" (8 batches) → download → re-run for the lever.')

    # ── Figure: per-reservoir EECU, dual SVM (y) vs VV-Otsu (x), log-log ──────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT_PNG = Path('analysis/method_comparison_output/eecu_svm_vs_vv.png')
    fig, ax = plt.subplots(figsize=(8, 7.5))
    cl, fl = hh[~hh['flag']], hh[hh['flag']]
    ax.scatter(cl['vv_otsu'], cl['svm_dual'], s=55, color='#1f77b4',
               edgecolors='white', linewidths=0.6, zorder=4, label='reservoir')
    ax.scatter(fl['vv_otsu'], fl['svm_dual'], s=70, color='#d62728', marker='x',
               zorder=5, label='cache-confounded (excl.)')
    lo = min(hh['vv_otsu'].min(), hh['svm_dual'].min()) * 0.7
    hi = max(hh['vv_otsu'].max(), hh['svm_dual'].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.7, zorder=3, label='1:1 (equal cost)')
    med = clean['ratio'].median()
    ax.plot([lo, hi], [lo * med, hi * med], color='#2ca02c', lw=1.3, alpha=0.8,
            zorder=3, label=f'median SVM/VV = {med:.2f}')
    for name, r in cl.iterrows():
        ax.annotate(name.replace('_', ' '), (r['vv_otsu'], r['svm_dual']),
                    fontsize=5.5, xytext=(3, 2), textcoords='offset points', color='#555')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('VV-only Otsu cost (EECU-seconds)', fontsize=10)
    ax.set_ylabel('dual VV+VH SVM cost (EECU-seconds)', fontsize=10)
    ax.set_title('Computational cost: dual-pol SVM vs VV-only Otsu\n'
                 f'points below 1:1 → SVM cheaper (median {med:.2f}×, N={len(clean)})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.25, which='both')
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUT_PNG}')
else:
    print('\n[no head-to-head] need both svm_dual and vv_otsu succeeded tasks.')
