"""
audit_area_level_monotonicity.py (2026-07-29)

Complementary check to audit_mask_area_outliers.py: for each reservoir and
period, sort the production (mask_wl_pairs_{res}.csv) dates by water level
and flag any point where area DECREASES while WL INCREASES relative to the
previous (by-level) point. A physically filling/draining reservoir should
show area moving with level; a drop while level rises is either a genuine
classification failure (found this way for Rosamarina: two March 2026 dates
with wind-roughening-style salt-and-pepper misclassification, 14.8% and 2.9%
area deficits -- both well under the >60% deviation threshold the OTHER
audit uses, so that check alone misses this failure mode) or, more rarely,
real hysteresis.

Flags are sorted by severity (area_drop_pct). This does NOT auto-exclude
anything -- it is a review list, not a filter.
"""
import pathlib
import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path('analysis/schwatke_output')
RESERVOIRS = ['Poma', 'Rosamarina', 'Pozzillo', 'Ancipa', 'Garcia',
              'Arancio', 'Castello', 'Olivo', 'Nicoletti']

all_flags = []
for res in RESERVOIRS:
    f = OUT_DIR / f'mask_wl_pairs_{res}.csv'
    if not f.exists():
        continue
    df = pd.read_csv(f, parse_dates=['date'])
    for period, sub in df.groupby('period'):
        sub = sub.dropna(subset=['area_ha', 'wl_m']).sort_values('wl_m').reset_index(drop=True)
        if len(sub) < 2:
            continue
        sub['area_prev'] = sub['area_ha'].shift(1)
        sub['wl_prev'] = sub['wl_m'].shift(1)
        sub['date_prev'] = sub['date'].shift(1)
        sub['area_drop_pct'] = (sub['area_prev'] - sub['area_ha']) / sub['area_prev'] * 100
        sub['day_gap'] = (sub['date'] - sub['date_prev']).abs().dt.days
        # Only meaningful within a single monotonic fill/drain episode: WL-adjacent
        # points must also be temporally close, else "adjacent by WL" can pair two
        # dates from different years/hydrological cycles that coincidentally share
        # a similar level -- Period A (2014-2016, area-percentile-selected across
        # multiple cycles) produces exactly that false-positive pattern.
        flagged = sub[(sub['wl_m'] > sub['wl_prev']) & (sub['area_drop_pct'] > 0)
                      & (sub['day_gap'] <= 45)]
        for _, row in flagged.iterrows():
            all_flags.append({
                'reservoir': res, 'period': period,
                'date': row['date'].strftime('%Y-%m-%d'),
                'date_prev': row['date_prev'].strftime('%Y-%m-%d'),
                'day_gap': int(row['day_gap']),
                'area_ha': round(row['area_ha'], 1), 'area_prev': round(row['area_prev'], 1),
                'wl_m': round(row['wl_m'], 2), 'wl_prev': round(row['wl_prev'], 2),
                'area_drop_pct': round(row['area_drop_pct'], 1),
                'wl_source': row.get('wl_source', None),
            })

out = pd.DataFrame(all_flags).sort_values('area_drop_pct', ascending=False)
out.to_csv(OUT_DIR / 'area_level_monotonicity_audit.csv', index=False)
print(f'{len(out)} non-monotonic (level up, area down) points flagged across '
      f'{out["reservoir"].nunique()} reservoirs.\n')
print(out.to_string(index=False))
print(f'\nSaved analysis/schwatke_output/area_level_monotonicity_audit.csv')
