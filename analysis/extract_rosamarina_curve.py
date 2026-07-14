"""
extract_rosamarina_curve.py

Extract the 2025 updated centimetric area-volume curve for Rosamarina from the
official bathymetric-survey report PDF (RILIEVO BATIMETRICO 2025). The table
spans ~45 pages, each laid out as 4 parallel blocks of (Quota[m] | Volume[mc] |
Area[mq]) at 1 cm resolution. Text extraction linearises and scrambles the
column association, so we reconstruct spatially: within each block's x-window we
cluster numeric words into rows by y, then order the three numbers left-to-right
(quota, volume, area) — robust to number right/left alignment.

Output: validation_data/updated_curves/rosamarina_2025.csv (quota_m, vol_m3, area_m2)
This is the Rosamarina analogue of Poma's NewCurves/POMA_new.XLS, enabling
poma_curve_validation.py-style validation of the SAR DEM_B against a 2nd
independent field reference.
"""

import fitz, sys, pathlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PDF = 'C:/Users/Unipa/Documents/GEE/Data/NewCurves/ROSAMARINA (Aree-Volumi 2025).pdf'
OUT_DIR = pathlib.Path('validation_data/updated_curves')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'rosamarina_2025.csv'

# x0 windows for the 4 side-by-side blocks (from coordinate probe: quota anchors
# at ~35/239/432/625; each block spans quota..volume..area before the next block).
BLOCK_WINDOWS = [(20, 230), (230, 422), (422, 618), (618, 860)]
Y_TOL = 6.0   # row clustering tolerance (row pitch ~16 px, intra-row spread ~3 px)


def to_num(s):
    s = s.strip().replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def cluster_rows(words):
    """words: list of (x0, y0, value); returns list of rows (each a list sorted by x0)."""
    words = sorted(words, key=lambda t: t[1])
    rows, cur, y_ref = [], [], None
    for x0, y0, v in words:
        if y_ref is None or abs(y0 - y_ref) <= Y_TOL:
            cur.append((x0, y0, v))
            y_ref = y0 if y_ref is None else (y_ref + y0) / 2
        else:
            rows.append(cur); cur = [(x0, y0, v)]; y_ref = y0
    if cur:
        rows.append(cur)
    return rows


d = fitz.open(PDF)
curve = {}   # quota -> (vol_m3, area_m2)
bad_rows = 0

for pi in range(d.page_count):
    nums = [(w[0], w[1], to_num(w[4])) for w in d[pi].get_text('words')]
    nums = [t for t in nums if t[2] is not None]
    if len(nums) < 30:
        continue   # not a table page (cover, diagrams, appendix)
    for xlo, xhi in BLOCK_WINDOWS:
        block = [t for t in nums if xlo <= t[0] < xhi]
        for row in cluster_rows(block):
            row.sort(key=lambda t: t[0])          # left -> right = quota, vol, area
            if len(row) != 3:
                bad_rows += 1
                continue
            q, v, a = row[0][2], row[1][2], row[2][2]
            if not (100.0 <= q <= 200.0):          # quota sanity (Rosamarina ~121-166 m)
                continue
            curve[round(q, 2)] = (v, a)

df = pd.DataFrame(
    [(q, v, a) for q, (v, a) in sorted(curve.items())],
    columns=['quota_m', 'vol_m3', 'area_m2'],
)

# ── Clean up block-boundary extraction spikes ─────────────────────────────────
# A handful of rows (first row of a block) pick up an anomalously high value,
# producing an impossible >1 Mm3 drop over 1 cm. Drop single-point upward spikes
# that break monotonicity, then re-grid onto the clean centimetric grid and apply
# a cumulative-max guard so the cumulative curve is strictly non-decreasing.
def despike(y):
    keep = np.ones(len(y), bool)
    for i in range(1, len(y) - 1):
        if y[i] > y[i + 1] + 1e-6 and y[i] >= y[i - 1] - 1e-6:   # local upward spike
            keep[i] = False
    return keep

n_before = len(df)
mask = despike(df.vol_m3.values) & despike(df.area_m2.values)
df = df[mask].reset_index(drop=True)

grid = np.round(np.arange(df.quota_m.min(), df.quota_m.max() + 1e-9, 0.01), 2)
v_i = np.maximum.accumulate(np.interp(grid, df.quota_m, df.vol_m3))
a_i = np.maximum.accumulate(np.interp(grid, df.quota_m, df.area_m2))
df = pd.DataFrame({'quota_m': grid,
                   'vol_m3':  np.round(v_i).astype(np.int64),
                   'area_m2': np.round(a_i).astype(np.int64)})
print(f"despiked {n_before - int(mask.sum())} block-boundary spikes; regridded to {len(df)} rows")

# ── Validation ────────────────────────────────────────────────────────────────
q = df.quota_m.values
step = np.diff(q)
gaps = np.sum(step > 0.0151)                       # centimetric grid should be 0.01 m
mono_v = int(np.sum(np.diff(df.vol_m3.values)  < -1e-6))
mono_a = int(np.sum(np.diff(df.area_m2.values) < -1e-6))

print(f"rows extracted : {len(df)}  (skipped {bad_rows} non-triplet rows)")
print(f"quota range    : {q.min():.2f} - {q.max():.2f} m  (step median {np.median(step):.3f} m)")
print(f"grid gaps >1cm : {gaps}")
print(f"volume         : {df.vol_m3.min():.0f} - {df.vol_m3.max():.0f} m3  "
      f"(= {df.vol_m3.max()/1e6:.2f} Mm3)   non-monotonic steps: {mono_v}")
print(f"area           : {df.area_m2.min():.0f} - {df.area_m2.max():.0f} m2  "
      f"(= {df.area_m2.max()/1e4:.1f} ha)   non-monotonic steps: {mono_a}")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
