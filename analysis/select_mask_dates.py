"""
select_mask_dates.py

Selects ~20 mask-export dates per Sicilian reservoir from the GEE area series.
Stratified by area percentile within two temporal windows:
  - Period A (2014-2016): 10 dates covering 5th to 95th percentile of area
  - Period B (2022-2026): 10 dates covering 5th to 95th percentile of area

Input:  raw_data/GEE_GlobalPilotV2a/SAR_area_{reservoir}.csv
Output: analysis/selected_mask_dates.json
        analysis/selected_mask_dates.csv  (human-readable summary)

Run:
    python analysis/select_mask_dates.py
"""

import json
import pathlib
import numpy as np
import pandas as pd

RESERVOIRS = ['Poma', 'Rosamarina', 'Pozzillo', 'Ancipa', 'Garcia']

DATA_DIR  = pathlib.Path('raw_data/GEE_GlobalPilotV2a')
OUT_JSON  = pathlib.Path('analysis/selected_mask_dates.json')
OUT_CSV   = pathlib.Path('analysis/selected_mask_dates.csv')

# Dates with known classification failures — never selected, treated as missing.
FORCE_EXCLUDE = {
    'Ancipa': {'2015-06-11', '2016-10-03', '2025-05-31', '2022-01-29'},
}

PERIODS = {
    'A': ('2014-10-01', '2016-12-31'),
    'B': ('2022-01-01', '2026-06-30'),
}
N_PER_PERIOD = 10
PERCENTILES  = np.linspace(5, 95, N_PER_PERIOD)   # 5, 15, ..., 95


def load_area_series(reservoir: str) -> pd.DataFrame:
    path = DATA_DIR / f'SAR_area_{reservoir}.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['date', 'area_ha']].dropna()
    df = df.sort_values('date').reset_index(drop=True)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Global outliers: >3σ from median
    median = df['area_ha'].median()
    std    = df['area_ha'].std()
    mask   = (df['area_ha'] - median).abs() <= 3 * std

    # Temporal spike filter: flag dates where area deviates >60% from a
    # 5-image rolling median of neighbours. Catches S1A/S1C misclassifications
    # that appear as sudden near-zero drops surrounded by correct values
    # (e.g. Ancipa 2026-01-08: 5 ha between dates showing 60–73 ha).
    rolled = df['area_ha'].rolling(5, center=True, min_periods=2).median()
    with np.errstate(invalid='ignore'):
        spike = (df['area_ha'] - rolled).abs() / (rolled + 1e-6) > 0.60
    mask = mask & ~spike

    return df[mask].copy()


def select_dates_for_period(df: pd.DataFrame, start: str, end: str,
                            n: int = N_PER_PERIOD,
                            excluded: set = None) -> list[dict]:
    sub = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    if excluded:
        sub = sub[~sub['date'].dt.strftime('%Y-%m-%d').isin(excluded)]
    if len(sub) == 0:
        return []
    sub = remove_outliers(sub)
    if len(sub) == 0:
        return []

    targets = np.percentile(sub['area_ha'], PERCENTILES)
    selected = []
    used_dates = set()
    for pct, target in zip(PERCENTILES, targets):
        residual = (sub['area_ha'] - target).abs()
        idx = residual.idxmin()
        row = sub.loc[idx]
        date_str = row['date'].strftime('%Y-%m-%d')
        if date_str in used_dates:
            residual2 = residual.copy()
            for already in used_dates:
                already_idx = sub[sub['date'] == pd.Timestamp(already)].index
                residual2.loc[already_idx] = np.inf
            if residual2.min() < np.inf:
                idx = residual2.idxmin()
                row = sub.loc[idx]
                date_str = row['date'].strftime('%Y-%m-%d')
        used_dates.add(date_str)
        selected.append({
            'date':     date_str,
            'area_ha':  round(float(row['area_ha']), 2),
            'pct':      int(round(pct)),
        })

    selected.sort(key=lambda x: x['date'])
    return selected


def main():
    result  = {}
    rows    = []

    for res in RESERVOIRS:
        df = load_area_series(res)
        print(f'\n{res}: {len(df)} images, area {df["area_ha"].min():.1f}–{df["area_ha"].max():.1f} ha')

        per_period = {}
        excluded = FORCE_EXCLUDE.get(res, set())
        for period_name, (start, end) in PERIODS.items():
            dates = select_dates_for_period(df, start, end, excluded=excluded)
            per_period[period_name] = dates
            for d in dates:
                print(f'  {period_name} pct{d["pct"]:02d}  {d["date"]}  {d["area_ha"]} ha')
                rows.append({
                    'reservoir': res,
                    'period':    period_name,
                    'date':      d['date'],
                    'area_ha':   d['area_ha'],
                    'pct':       d['pct'],
                })
            if not dates:
                print(f'  {period_name}: NO DATA in {start}–{end}')

        result[res] = per_period

    OUT_JSON.write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f'\nSaved {OUT_JSON}')
    print(f'Saved {OUT_CSV}')

    # Print JavaScript-ready arrays for paste into exportSicilyMasks.js
    print('\n// -- paste into exportSicilyMasks.js --')
    print('var SELECTED_DATES = {')
    for res, periods in result.items():
        all_dates = []
        for p in periods.values():
            all_dates += [d['date'] for d in p]
        all_dates = sorted(set(all_dates))
        dates_str = ', '.join(f"'{d}'" for d in all_dates)
        print(f"  '{res}': [{dates_str}],")
    print('};')


if __name__ == '__main__':
    main()
