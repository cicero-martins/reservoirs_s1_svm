"""Sensitivity check: do results change if the 6 reference-noise-flagged
reservoirs (JRC noisier than SAR, reference_noise.csv ref_noise=True) are
dropped instead of kept? Re-tests: (1) A/P->KGE correlation, (2) VV-vs-dual
win pattern especially at low A/P, (3) climate residual effect.
"""
import pandas as pd
from scipy import stats

NOISY = pd.read_csv('analysis/reference_noise.csv')
noisy_names = set(NOISY[NOISY.ref_noise]['name'])
print("Noisy-JRC reservoirs (excluded in this sensitivity run):", sorted(noisy_names))

# ---- 1) A/P -> KGE curve ----
ap = pd.read_csv('analysis/pilot_kge_apcurve.csv')
ap_clean = ap[~ap.name.isin(noisy_names)]
print(f"\n=== A/P->KGE  (all N={len(ap)} vs clean N={len(ap_clean)}) ===")
for label, d in [('all', ap), ('clean', ap_clean)]:
    rho, p = stats.spearmanr(d.ap_m, d.best)
    print(f"  {label:6}: rho={rho:.3f} p={p:.2e}  median={d.best.median():.3f}  "
          f"low(<100)={d[d.ap_m<100].best.median():.3f}  high(>=200)={d[d.ap_m>=200].best.median():.3f}")

# ---- 2) VV vs dual win pattern, all vs clean, binned by A/P ----
fw = pd.read_csv('analysis/pilot_kge_4way.csv')
fw_clean = fw[~fw.name.isin(noisy_names)]
print(f"\n=== VV-Otsu vs per-scene dual SVM, by A/P bin (all N={len(fw)} vs clean N={len(fw_clean)}) ===")
bins = [0, 100, 150, 200, 600]
labels = ['<100', '100-150', '150-200', '>=200']
for label, d in [('all', fw), ('clean', fw_clean)]:
    d = d.copy()
    d['bin'] = pd.cut(d.ap_m, bins=bins, labels=labels)
    d['dvv'] = d.kge_adapt - d.kge_vv   # positive = dual wins
    print(f"  -- {label} --")
    g = d.groupby('bin', observed=True).agg(
        n=('dvv', 'size'),
        median_dual_minus_vv=('dvv', 'median'),
        dual_wins=('dvv', lambda x: (x > 0.02).sum()),
        vv_wins=('dvv', lambda x: (x < -0.02).sum()),
    )
    print(g.to_string())
    rho, p = stats.spearmanr(d.ap_m, d.dvv)
    print(f"  Spearman(A/P, dual-vv) = {rho:.3f}  p={p:.3f}")

# ---- 3) climate residual, all vs clean ----
bi = pd.read_csv('analysis/biome_kge.csv')
bi_clean = bi[~bi.name.isin(noisy_names)]
print(f"\n=== Climate effect (all N={len(bi)} vs clean N={len(bi_clean)}) ===")
for label, d in [('all', bi), ('clean', bi_clean)]:
    d = d.copy()
    # residual after removing A/P's monotonic effect via rank
    d['ap_rank'] = d.ap_m.rank()
    # simple partial: regress kge on ap_m (linear in rank), take residual, test vs biome
    import numpy as np
    coef = np.polyfit(d.ap_rank, d.kge, 1)
    d['resid'] = d.kge - np.polyval(coef, d.ap_rank)
    groups = [g['resid'].values for _, g in d.groupby('biome') if len(g) >= 3]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
    else:
        H, p = float('nan'), float('nan')
    print(f"  {label:6}: kruskal H={H:.2f} p={p:.3f}")
    print(d.groupby('biome')['kge'].agg(['median', 'count']).sort_values('median').to_string())
