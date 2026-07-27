"""
wl_gauge_swot_validation.py

Proper gauge-vs-SWOT water-level validation, addressing gaps found in the
original fullrs_wl_ladder.py comparison (2026-07-24 review):
  1. fullrs_wl_ladder.py's load_gauge() had NO stuck-sensor filter at all, so
     RMSE included known-bad gauge windows (Rosamarina, Garcia), comparing
     SWOT's real variation against a frozen/wrong gauge reading -- inflating
     the apparent SWOT error with what is actually a gauge fault.
  2. SWOT had no outlier removal at all (only the source quality_f filter).
     Applies the SAME cleaning pipeline as Paper 1's clean_and_smooth
     (_remove_global 2sigma + _remove_local(5,1.5) x2 + _remove_local(10,1.5)),
     adapted from area_ha to water level.
  3. Adds KGE alongside RMSE, using Paper 1's own kge() definition, gauge as
     the observed/reference series and SWOT as the candidate being validated.
  4. (2026-07-27) Removes each reservoir's gauge-vs-SWOT datum offset
     (CONFIGS[...]['swot_bias_corr'], estimated by an earlier run of this same
     script) before scoring -- SWOT WSE and a gauge reading are not guaranteed
     to share a vertical datum, and Poma's near-perfect r=1.00-but-bias=RMSE
     signature showed this offset was being scored as "error" when it is really
     a constant, physically uninteresting reference-frame mismatch (the same
     issue CONFIGS[...]['boletin_cfg']['bias_corr'] already corrects for the
     boletin V->h source). reported rmse_m/kge are POST-correction; bias_m is
     the small residual left over (should be ~0), and swot_bias_corr_m records
     what was removed.

Output: analysis/schwatke_output/fullrs/wl_gauge_swot_clean.csv
"""
import sys, pathlib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

SWOT_DIR = pathlib.Path('validation_data/SWOT')
OUT = pathlib.Path('analysis/schwatke_output/fullrs')
OUT.mkdir(parents=True, exist_ok=True)

AP = {'Arancio': 182.2, 'Poma': 190.1, 'Garcia': 167.7, 'Pozzillo': 240.5,
      'Rosamarina': 187.4, 'Ancipa': 90.5, 'Olivo': 50.7, 'Castello': 126.7,
      'Nicoletti': 119.7}
RESERVOIRS = list(AP.keys())


# ── Paper 1's cleaning pipeline (analysis/compute_kge_v3.py), verbatim ────────
def _remove_global(s, threshold=2.0):
    mean, sd = s.mean(), s.std()
    return s[np.abs(s - mean) <= threshold * sd]


def _remove_local(s, window=5, threshold=1.5):
    arr, idx = s.values.copy(), s.index.tolist()
    keep, half = [], window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        win = arr[lo:hi]
        mean, sd = win.mean(), win.std()
        if sd == 0 or abs(arr[i] - mean) <= threshold * sd:
            keep.append(idx[i])
    return s.loc[keep]


def clean_series(s):
    """Outlier removal only (no LOWESS): global 2sigma, then local (5,1.5) x2, (10,1.5)."""
    s = _remove_global(s, 2.0)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 5, 1.5)
    s = _remove_local(s, 10, 1.5)
    return s


def kge(obs, sim):
    if obs.std() == 0 or sim.std() == 0 or len(obs) < 4:
        return np.nan, np.nan, np.nan, np.nan
    r, _ = stats.pearsonr(obs, sim)
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), r, alpha, beta


def load_swot_series(name, corrected=False):
    """corrected=True applies CONFIGS[name]['swot_bias_corr'] (the gauge-vs-SWOT
    datum offset) -- False (default) gives the raw series this script itself uses to
    *derive* that offset in the first place; the main() loop below computes both."""
    f = SWOT_DIR / f'{name}_swot.csv'
    if not f.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    swot = m.load_swot(f)
    if corrected:
        corr = m.CONFIGS.get(name, {}).get('swot_bias_corr', 0.0)
        swot = (swot + corr).rename('wl_swot') if corr and len(swot) else swot
    return swot


def bad_windows_for(name):
    cfg = m.CONFIGS.get(name, {})
    gauge_bad = cfg.get('gauge_bad_window')
    if not gauge_bad:
        return []
    raw = gauge_bad if isinstance(gauge_bad[0], (tuple, list)) else [gauge_bad]
    return [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in raw]


rows = []
for name in RESERVOIRS:
    cfg = m.CONFIGS[name]
    try:
        gauge = m.load_gauge(cfg)   # already drops flat/stuck runs (flat_tol=0.005, min 5d)
    except Exception:
        gauge = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    # Explicitly drop CONFIRMED bad windows too, in case a window's residual
    # noise (e.g. Garcia's dry-lakebed-floor readings still wobble slightly)
    # doesn't strictly trip the generic flat_tol filter.
    for lo, hi in bad_windows_for(name):
        gauge = gauge.loc[(gauge.index < lo) | (gauge.index > hi)]

    swot_raw = load_swot_series(name, corrected=False)
    n_raw = len(swot_raw)
    swot_clean_raw = clean_series(swot_raw) if n_raw > 0 else swot_raw
    n_removed = n_raw - len(swot_clean_raw)
    corr = m.CONFIGS.get(name, {}).get('swot_bias_corr', 0.0)
    swot_clean = (swot_clean_raw + corr).rename('wl_swot') if corr and len(swot_clean_raw) else swot_clean_raw

    merged = pd.merge_asof(
        gauge.rename_axis('date').rename('gauge').reset_index(),
        swot_clean.rename_axis('date').rename('swot').reset_index(),
        on='date', tolerance=pd.Timedelta('1D'), direction='nearest',
    ).dropna()

    if len(merged) >= 4:
        rmse = float(np.sqrt(((merged.gauge - merged.swot) ** 2).mean()))
        bias = float((merged.swot - merged.gauge).mean())  # residual bias AFTER swot_bias_corr; should be ~0
        kge_val, r, alpha, beta = kge(merged.gauge, merged.swot)
    else:
        rmse = bias = kge_val = r = alpha = beta = np.nan

    rows.append(dict(reservoir=name, ap_m=AP[name],
                     n_swot_raw=n_raw, n_swot_removed=n_removed,
                     n_pairs=len(merged), swot_bias_corr_m=corr,
                     rmse_m=round(rmse, 2) if rmse == rmse else np.nan,
                     bias_m=round(bias, 2) if bias == bias else np.nan,
                     kge=round(kge_val, 2) if kge_val == kge_val else np.nan,
                     r=round(r, 2) if r == r else np.nan,
                     alpha=round(alpha, 2) if alpha == alpha else np.nan,
                     beta=round(beta, 2) if beta == beta else np.nan)
    )
    print(f'{name}: SWOT {n_raw} raw -> {n_raw-n_removed} after cleaning ({n_removed} removed), '
          f'datum-corrected {corr:+.2f} m, {len(merged)} pairs with gauge -> '
          f'RMSE={rmse:.2f} m, residual bias={bias:+.2f} m, KGE={kge_val:.2f} '
          f'(r={r:.2f}, alpha={alpha:.2f}, beta={beta:.2f})')

df = pd.DataFrame(rows)
df.to_csv(OUT / 'wl_gauge_swot_clean.csv', index=False)
print(f'\nSaved {OUT / "wl_gauge_swot_clean.csv"}')
