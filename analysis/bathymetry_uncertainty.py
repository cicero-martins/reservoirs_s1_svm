"""
bathymetry_uncertainty.py  (E2 — bias model + uncertainty budget; bridge to Paper 1)

The one field-validated reservoir (Garcia echosounder) shows a systematic
ELEVATION-DEPENDENT bias in the SAR-waterline DEM: it over-estimates elevation
near the floor and under-estimates near the top (level-slicing artifact). This
script:

  1. Fits a NORMALISED bias model from Garcia — bias(f) and residual scatter s(f)
     as functions of the fractional position f=(elev-floor)/(top-floor) in the
     SAR-observable band. Because the reconstruction method is identical across
     reservoirs, this normalised model should transfer.
  2. TRANSFERABILITY TEST: applies the Garcia-derived correction to Rosamarina and
     Poma (which have independent survey curves) and checks whether the band
     capacity-change error shrinks. Garcia itself is shown as a (self-consistent)
     sanity check.
  3. UNCERTAINTY BUDGET: Monte-Carlo perturbs each DEM by the residual scatter s(f)
     to put a confidence interval on the SAR capacity-change estimate. (A uniform
     WL-datum error cancels in the band-relative volume, so the per-pixel
     reconstruction error — captured empirically by Garcia — dominates.)

Reuses the clean data layer in tool/bathymetry.py so numbers match the tool and
analysis/consolidate_bathymetry.py.

Outputs: analysis/schwatke_output/ bathymetry_uncertainty.csv, bathymetry_uncertainty.png
"""

import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str((pathlib.Path(__file__).resolve().parent.parent / 'tool')))
import bathymetry as bt   # clean data layer (load_dem, aev, design_curve, updated_curve, RESERVOIRS)

OUT = bt.DEM_DIR
PIXEL_M2 = 100.0          # 10 m pixel
K = 500                   # Monte-Carlo iterations
rng = np.random.default_rng(42)


def vol_rel_analytic(elev, mask, floor, top):
    """Volume above floor (Mm3) = sum of per-pixel water columns clipped to [0, band]."""
    col = np.clip(top - elev, 0.0, top - floor)
    return float(np.sum(col[mask]) * PIXEL_M2 / 1e6)


# ── 1. Fit the normalised bias model from Garcia's per-band errors ─────────────
band = pd.read_csv(OUT / 'garcia_survey' / 'garcia_error_by_band.csv')
G = bt.load_dem('Garcia', 'B')
gf, gt = G['floor'], G['top']
band['frac'] = ((band.band_lo + band.band_hi) / 2 - gf) / (gt - gf)
bias_coef, bias_cov = np.polyfit(band.frac, band.bias_m, 1, cov=True)  # bias(f) ≈ b0 + b1 f (+cov)
std_coef  = np.polyfit(band.frac, band.std_m,  1)     # residual scatter s(f)
print(f"Garcia bias model:  bias(f) = {bias_coef[1]:+.2f} {bias_coef[0]:+.2f}*f   (m)")
print(f"Garcia scatter:     s(f)    = {std_coef[1]:+.2f} {std_coef[0]:+.2f}*f   (m)")

def bias_of(frac):  return np.polyval(bias_coef, np.clip(frac, 0, 1))
def std_of(frac):   return np.clip(np.polyval(std_coef, np.clip(frac, 0, 1)), 0.3, None)


# ── Independent truth (band-relative capacity change vs design) ────────────────
def truth_band_pct(name):
    if name == 'Garcia':
        g = pd.read_csv(OUT / 'garcia_survey' / 'garcia_volume_change.csv').iloc[-1]
        return -float(g['loss_true_pct'])
    cap = bt.capacity_change(name)
    return cap.get('truth_band_pct') if cap else None


# ── 2 + 3. Per reservoir: correction transfer + Monte-Carlo CI ────────────────
rows = []
for name, cfg in bt.RESERVOIRS.items():
    dem = bt.load_dem(name, 'B')
    if dem is None:
        continue
    arr, mask, floor, top = dem['arr'], dem['mask'], dem['floor'], dem['top']
    frac = np.zeros_like(arr)
    frac[mask] = (arr[mask] - floor) / (top - floor)

    dc = bt.design_curve(name)
    vdes_rel = float(dc[1](top) - dc[1](floor)) if dc else np.nan

    # point estimates: uncorrected vs bias-corrected
    v_raw  = vol_rel_analytic(arr, mask, floor, top)
    arr_c  = arr - bias_of(frac)
    v_corr = vol_rel_analytic(np.where(mask, arr_c, np.nan), mask, floor, top)
    sar_raw_pct  = (v_raw  - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan
    sar_corr_pct = (v_corr - vdes_rel) / vdes_rel * 100 if vdes_rel else np.nan

    # Coherent Monte-Carlo: sample the (correlated) bias-model uncertainty. The
    # per-pixel random scatter s(f) averages out over ~10^4 pixels, so at reservoir
    # scale the SYSTEMATIC bias-model uncertainty dominates the volume error.
    fclip = np.clip(frac, 0, 1)
    mc = np.empty(K)
    for k in range(K):
        b = rng.multivariate_normal(bias_coef, bias_cov)
        arr_k = arr - np.polyval(b, fclip)
        mc[k] = vol_rel_analytic(np.where(mask, arr_k, np.nan), mask, floor, top)
    mc_pct = (mc - vdes_rel) / vdes_rel * 100
    lo, hi = np.percentile(mc_pct, [5, 95])

    tb = truth_band_pct(name)
    err_raw  = None if tb is None else round(sar_raw_pct  - tb, 1)
    err_corr = None if tb is None else round(sar_corr_pct - tb, 1)
    rows.append({
        'reservoir': name, 'ap_m': cfg['ap'],
        'sar_raw_pct': round(sar_raw_pct, 1), 'sar_corr_pct': round(sar_corr_pct, 1),
        'truth_band_pct': None if tb is None else round(tb, 1),
        'err_raw_pct': err_raw, 'err_corr_pct': err_corr,
        'mc_ci5_pct': round(lo, 1), 'mc_ci95_pct': round(hi, 1),
        'mc_halfwidth_pct': round((hi - lo) / 2, 1),
    })

df = pd.DataFrame(rows).sort_values('ap_m').reset_index(drop=True)
df.to_csv(OUT / 'bathymetry_uncertainty.csv', index=False)
print('\n' + df.to_string(index=False))

val = df.dropna(subset=['err_raw_pct'])
if len(val):
    print(f"\nBias-correction transferability (|error| vs independent truth):")
    print(f"  mean |err| raw       = {val.err_raw_pct.abs().mean():.1f} %")
    print(f"  mean |err| corrected = {val.err_corr_pct.abs().mean():.1f} %  "
          f"({'IMPROVED' if val.err_corr_pct.abs().mean() < val.err_raw_pct.abs().mean() else 'no gain'})")
    indep = val[val.reservoir != 'Garcia']   # Garcia is the self-calibrated case
    if len(indep):
        sig = float(np.sqrt((indep.err_corr_pct.astype(float) ** 2).mean()))
        print(f"  TRANSFER 1σ (independent reservoirs, excl. Garcia self-cal) = ±{sig:.1f} % "
              f"→ this systematic transfer error, not the MC calibration CI, is the dominant uncertainty.")
print(f"MC (bias-model calibration) 90% CI half-width, median: {df.mc_halfwidth_pct.median():.1f} % of design capacity.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.2))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28)

axA = fig.add_subplot(gs[0])
ff = np.linspace(0, 1, 50)
axA.axhline(0, color='gray', lw=0.8)
axA.errorbar(band.frac, band.bias_m, yerr=band.std_m, fmt='o', color='C3', capsize=3, label='Garcia bands')
axA.plot(ff, bias_of(ff), 'C0-', lw=2, label='bias(f) fit')
axA.fill_between(ff, bias_of(ff) - std_of(ff), bias_of(ff) + std_of(ff), color='C0', alpha=0.15,
                 label='± residual s(f)')
axA.set_xlabel('Fractional height in exposed band  f'); axA.set_ylabel('DEM − survey (m)')
axA.set_title('1 · Normalised bias model (Garcia)'); axA.legend(fontsize=8); axA.grid(True, alpha=0.3)

axB = fig.add_subplot(gs[1])
vv = df.dropna(subset=['truth_band_pct']).reset_index(drop=True)
x = np.arange(len(vv)); w = 0.26
axB.axhline(0, color='gray', lw=0.8)
axB.bar(x - w, vv.sar_raw_pct,     w, color='#90caf9', label='SAR raw')
axB.bar(x,     vv.sar_corr_pct,    w, color='#1565c0', label='SAR bias-corrected')
axB.bar(x + w, vv.truth_band_pct,  w, color='C2',      label='Independent truth')
axB.set_xticks(x); axB.set_xticklabels(vv.reservoir, fontsize=9)
axB.set_ylabel('Capacity change vs design (%)')
axB.set_title('2 · Bias-correction transfer'); axB.legend(fontsize=8); axB.grid(True, alpha=0.3, axis='y')

axC = fig.add_subplot(gs[2])
x = np.arange(len(df))
yerr = np.vstack([np.clip(df.sar_corr_pct - df.mc_ci5_pct, 0, None),
                  np.clip(df.mc_ci95_pct - df.sar_corr_pct, 0, None)])
axC.axhline(0, color='gray', lw=0.8)
axC.errorbar(x, df.sar_corr_pct, yerr=yerr, fmt='o', color='#1565c0', capsize=4,
             label='SAR corrected ± MC 90% CI')
tv = df.dropna(subset=['truth_band_pct'])
axC.scatter(tv.index, tv.truth_band_pct, marker='s', color='C2', zorder=5, label='Independent truth')
axC.set_xticks(x); axC.set_xticklabels([f'{r.reservoir}\nA/P {r.ap_m:.0f}' for _, r in df.iterrows()], fontsize=8)
axC.set_ylabel('Capacity change vs design (%)')
axC.set_title('3 · Uncertainty budget (Monte-Carlo)'); axC.legend(fontsize=8); axC.grid(True, alpha=0.3, axis='y')

fig.subplots_adjust(top=0.88)
fig.savefig(OUT / 'bathymetry_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {OUT / 'bathymetry_uncertainty.csv'}\nSaved: {OUT / 'bathymetry_uncertainty.png'}")
