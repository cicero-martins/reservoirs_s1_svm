"""
fix_rosamarina_swot_curve.py (2026-07-29)

Replaces the nearest-neighbour-snapped SWOT water level (interp_wl) with a
curve-inverted one for every 'swot'-sourced date in
rosamarina_densify_prototype_pairs.csv, mirroring the fix applied for Poma's
SWOT-only experiment: fit A=a(h-h0)^b on genuine SAR-area/SWOT-WL coincident
pairs (continuous SAR series matched to every raw SWOT observation, +/-3
days), then invert using each mask's own observed area. Avoids assigning the
same stale SWOT reading to several different-area masks (Rosamarina's SWOT
revisit leaves multiple Sentinel-1 dates snapping to one SWOT pass). Gauge-
sourced rows are untouched (gauge is dense/daily, no stuck-value issue).
"""
import pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import schwatke_bathymetry_3d as m

RES = 'Rosamarina'
OUT_DIR = pathlib.Path('analysis/schwatke_output')
cfg = m.CONFIGS[RES]

swot = m.load_swot_corrected(cfg, cfg['swot_csv'], RES)
cont_area = pd.read_csv(cfg['sar_csv'], parse_dates=['date']).sort_values('date')
cont_area = cont_area[['date', 'area_ha']].dropna().groupby('date')['area_ha'].mean()

swot_pairs = []
for dt, wl in swot.items():
    near = cont_area[(cont_area.index >= dt - pd.Timedelta(days=3)) &
                     (cont_area.index <= dt + pd.Timedelta(days=3))]
    if len(near):
        idx = (near.index - dt).to_series().abs().values.argmin()
        swot_pairs.append({'wl_m': wl, 'area_ha': float(near.iloc[idx])})
swot_pairs = pd.DataFrame(swot_pairs)
print(f'{len(swot_pairs)} genuine SAR-area/SWOT-WL coincident pairs '
      f'(of {len(swot)} raw SWOT observations)')

model = m.fit_hyps_model(swot_pairs, cfg['h0_bound_lo'])
if model is None:
    raise SystemExit('SWOT-calibrated curve fit failed -- cannot proceed')
a, h0, b = model
print(f'SWOT-calibrated model: A = {a:.3f}*(h-{h0:.2f})^{b:.3f}')

df = pd.read_csv(OUT_DIR / 'rosamarina_densify_prototype_pairs.csv', parse_dates=['date'])
is_swot = df['source'] == 'swot'
old_wl = df.loc[is_swot, 'wl_m'].copy()
df.loc[is_swot, 'wl_m'] = df.loc[is_swot, 'area_ha'].apply(
    lambda area: m.invert_power_law(area, a, h0, b))

print(f'\n{is_swot.sum()} SWOT-sourced dates re-leveled (curve-inverted vs snapped):')
comp = pd.DataFrame({'date': df.loc[is_swot, 'date'].dt.strftime('%Y-%m-%d'),
                      'snapped_wl': old_wl, 'curve_wl': df.loc[is_swot, 'wl_m']})
print(comp.to_string(index=False))

df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.to_csv(OUT_DIR / 'rosamarina_densify_prototype_pairs.csv', index=False, float_format='%.4f')
print(f'\nUpdated {OUT_DIR / "rosamarina_densify_prototype_pairs.csv"} in place')
