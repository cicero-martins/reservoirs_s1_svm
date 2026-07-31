"""
hypsometry_comparison.py (2026-07-31)

Item 1 of the new sec:res_change sequence: for all 9 reservoirs, compare the
hypsometric curve built three ways --
  (a) gauge+SWOT-fallback -- the production pairing already used everywhere
      else in the paper (mask_wl_pairs_{name}.csv, period B, wl_source column
      already encodes gauge-primary/SWOT-in-malfunction-windows).
  (b) SWOT-only / full remote sensing (FRS) -- reuses
      build_frs_dem.fit_swot_curve(), the power-law fit on genuine SAR-area/
      SWOT-WL coincident pairs (no gauge anywhere).
  (c) the best available independent reference -- bt.updated_curve(name),
      falling back to bt.design_curve(name) where no updated survey exists
      (Ancipa, Pozzillo).
against each other, measured as band volume (Mm3) over the WL range (a) and
(b) have in common -- same metric/convention as fullrs_wl_ladder.py, which
this generalizes from 6 to 9 reservoirs (adding Castello/Nicoletti/Olivo) and
corrects the "gauge" tier to the actual gauge+SWOT-fallback production
definition (fullrs_wl_ladder.py's own gauge tier excludes malfunction windows
but does not fill them with SWOT, so it is not quite (a) as defined here).

Output: analysis/schwatke_output/hypsometry_comparison/
  hypsometry_comparison.csv, hypsometry_comparison.png (3x3 grid)
"""
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'tool'))
import schwatke_bathymetry_3d as m
import bathymetry as bt
from build_frs_dem import fit_swot_curve, RESERVOIRS

OUT_DIR = pathlib.Path('analysis/schwatke_output')
OUT = OUT_DIR / 'hypsometry_comparison'
OUT.mkdir(parents=True, exist_ok=True)


def fit_production_curve(name):
    pairs = pd.read_csv(OUT_DIR / f'mask_wl_pairs_{name}.csv')
    pairs = pairs[pairs.period == 'B'].copy()
    fit = m.fit_hyps_model(pairs, m.CONFIGS[name]['h0_bound_lo'])
    return fit, pairs


def reference_curve(name):
    """(area_interp_or_None, vol_interp, label) for the best available
    independent reference: updated survey curve, else the design curve."""
    upd = bt.updated_curve(name)
    if upd is not None and upd[1] is not None:
        return upd[0], upd[1], 'updated curve'
    area_i, vol_i = bt.design_curve(name)
    return area_i, vol_i, 'design curve'


def band_vol(popt, hs):
    return float(cumulative_trapezoid(m.power_law(hs, *popt), hs, initial=0)[-1] * 0.01)


rows = []
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for ax, name in zip(axes.flat, RESERVOIRS):
    ap = bt.RESERVOIRS[name]['ap']
    fit_a, pairs_a = fit_production_curve(name)
    fit_b, pairs_b = fit_swot_curve(name)
    if fit_a is None or fit_b is None:
        print(f'{name}: skip (fit_a={fit_a is not None}, fit_b={fit_b is not None})')
        continue

    pairs_a.assign(tier='a_gaugeSWOT').to_csv(OUT / f'pairs_{name}_a_gaugeSWOT.csv', index=False)
    pairs_b.assign(tier='b_FRS').to_csv(OUT / f'pairs_{name}_b_FRS.csv', index=False)

    lo = max(pairs_a.wl_m.min(), pairs_b.wl_m.min())
    hi = min(pairs_a.wl_m.max(), pairs_b.wl_m.max())
    hs = np.linspace(lo, hi, 100)
    vol_a, vol_b = band_vol(fit_a, hs), band_vol(fit_b, hs)

    area_c, vol_c, c_label = reference_curve(name)
    v_c = float(vol_c(hi) - vol_c(lo))
    # Poma's official curve has no direct area column (POMA_new.XLS is
    # quota+volume only), so bt.updated_curve derives area via np.gradient on
    # coarsely-rounded volume values -- genuinely spiky/stair-stepped, not a
    # smooth reference; suppressed from the plot, matching the precedent
    # already established in fullrs_wl_ladder.py's own figure for this reason.
    if name == 'Poma':
        area_c = None

    rows.append(dict(
        reservoir=name, ap_m=ap, n_a=len(pairs_a), n_b=len(pairs_b),
        band=f'{lo:.1f}-{hi:.1f}', ref=c_label,
        vol_a_gaugeSWOT=round(vol_a, 2), vol_b_FRS=round(vol_b, 2), vol_c_ref=round(v_c, 2),
        b_minus_a=round(vol_b - vol_a, 2),
        a_minus_c=round(vol_a - v_c, 2), b_minus_c=round(vol_b - v_c, 2),
    ))

    ax.scatter(pairs_a.area_ha, pairs_a.wl_m, s=14, color='#999999', alpha=0.7,
               zorder=1, label='production pairs (gauge+SWOT)')
    ax.plot(m.power_law(hs, *fit_a), hs, color='#1565c0', lw=2.2, label='(a) gauge+SWOT-fallback')
    ax.plot(m.power_law(hs, *fit_b), hs, color='#8e24aa', lw=2.2, label='(b) SWOT-only (FRS)')
    if area_c is not None:
        ax.plot(area_c(hs), hs, color='#2e7d32', lw=1.8, ls='--', label=f'(c) {c_label}')
    ax.set_title(f'{name} (A/P {ap:.0f} m)', fontsize=9.5)
    ax.set_xlabel('Area (ha)', fontsize=8); ax.set_ylabel('Water level (m ASL)', fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5); ax.grid(alpha=0.25)

fig.suptitle('Hypsometric curve: gauge+SWOT-fallback vs. SWOT-only (FRS) vs. best available reference',
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'hypsometry_comparison.png', dpi=160)
plt.close(fig)

df = pd.DataFrame(rows).sort_values('ap_m')
df.to_csv(OUT / 'hypsometry_comparison.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print(f'\nSaved {OUT}/hypsometry_comparison.{{csv,png}}')
