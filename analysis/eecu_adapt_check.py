"""Compute the svm_adapt vs vv_otsu (and vs svm_dual) EECU cost ratio directly
from the freshly regenerated gee_eecu_costs.csv, on the CURRENT reservoir set
(including the 22-reservoir global-coverage expansion). gee_eecu_report.py
never paired svm_adapt; the paper's 1.21x/1.64x figures were a one-off from
the smaller original set and need re-verification here.
"""
import pandas as pd

d = pd.read_csv('analysis/gee_eecu_costs.csv')
d = d[d.state == 'SUCCEEDED'].copy()
d['start_dt'] = pd.to_datetime(d['start'], errors='coerce')
latest = d.sort_values('start_dt').groupby(['method', 'reservoir'], as_index=False).last()
piv = latest.pivot_table(index='reservoir', columns='method', values='eecu_seconds')

print("n per method:", piv.notna().sum().to_dict())

for a, b in [('svm_adapt', 'vv_otsu'), ('svm_adapt', 'svm_dual'), ('vv_fast', 'vv_otsu')]:
    sub = piv.dropna(subset=[a, b]).copy()
    sub['ratio'] = sub[a] / sub[b]
    q1, q3 = sub['ratio'].quantile([0.25, 0.75])
    fence = q3 + 3.0 * (q3 - q1)
    flagged = sub[sub['ratio'] > fence]
    clean = sub[sub['ratio'] <= fence]
    print(f"\n=== {a}/{b}  (N={len(sub)}, robust N={len(clean)}) ===")
    if len(flagged):
        print("  flagged (cache?):", list(flagged.index))
    print(f"  median = {clean['ratio'].median():.3f}   mean = {clean['ratio'].mean():.3f}")
    print(f"  total {a} / total {b} = {clean[a].sum()/clean[b].sum():.3f}")

# which reservoirs have svm_adapt cost data — how many are from the NEW expansion?
cand = pd.read_csv('analysis/global_pilot_v4_candidates.csv')
new_names = set(cand[cand.notes.astype(str).str.contains('coverage pool', case=False, na=False)].name)
have_adapt = set(piv['svm_adapt'].dropna().index) if 'svm_adapt' in piv.columns else set()
print(f"\nsvm_adapt coverage: {len(have_adapt)} reservoirs total, "
      f"{len(have_adapt & new_names)} of {len(new_names)} from the new expansion")
print("new-expansion reservoirs WITH svm_adapt cost:", sorted(have_adapt & new_names))
print("new-expansion reservoirs MISSING svm_adapt cost:", sorted(new_names - have_adapt))
