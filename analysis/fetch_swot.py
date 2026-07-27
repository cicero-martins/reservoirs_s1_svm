"""
fetch_swot.py  (BACKBONE Level 2 — SWOT swath-altimetry water level)

Pulls the SWOT water-surface-elevation time series for the Sicilian reservoirs via
the PODAAC Hydrocron API (no Earthdata download needed once the PLD lake_id is
known). Lake_ids were resolved from SWOT_L2_HR_LakeSP Prior granules by name +
location (see project_paper2 memory). SWOT is TRACK-INDEPENDENT (swath) → the
answer for reservoirs a nadir altimetry track misses.

Saves validation_data/SWOT/<name>_swot.csv (datetime, wse, wse_u, area_km2, quality_f).
"""

import sys, pathlib, requests, io
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
import urllib3; urllib3.disable_warnings()

OUT = pathlib.Path('validation_data/SWOT'); OUT.mkdir(parents=True, exist_ok=True)
BASE = 'https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries'
FILL = -1e11   # SWOT fill value is -999999999999

LAKE_ID = {
    'Poma':       '2190000993',   # LAGO POMA
    'Rosamarina': '2190001003',   # LAGO DI CACCAMO
    'Garcia':     '2190001353',   # LAGO DI GARCIA
    'Arancio':    '2190001033',   # LAGO ARANCIO
    'Pozzillo':   '2190000763',   # LAGO DI POZZILLO
    'Ancipa':     '2190001013',   # LAGO DELL'ANCIPA
    # Added 2026-07-27: found by the author directly from the PLD vector shapefile
    # (SWOT L2 Lake Single-Pass Vector Prior Data Product, Version D, downloaded from
    # Earthdata Search and spatially filtered to Sicily -- raw_data/swot_l2_hr_lakeSP/),
    # NOT from the earlier CMR granule-bbox search, which over-matched and returned
    # unrelated lakes for these 3 (see manuscript sec:res_inputvalid before this fix).
    'Olivo':      '2190001062',   # LAGO OLIVO
    'Castello':   '2190001122',   # LAGO CASTELLO
    'Nicoletti':  '2190000863',   # LAGO NICOLETTI
}

S = requests.Session(); S.verify = False
rows = []
for name, lid in LAKE_ID.items():
    r = S.get(BASE, params={'feature': 'PriorLake', 'feature_id': lid,
                            'start_time': '2023-01-01T00:00:00Z', 'end_time': '2026-08-01T00:00:00Z',
                            'output': 'csv',
                            'fields': 'lake_id,time_str,wse,wse_u,area_total,quality_f'}, timeout=90)
    if r.status_code != 200:
        print(f'{name}: HTTP {r.status_code} — {r.text[:100]}'); continue
    csv = r.json().get('results', {}).get('csv', '')
    df = pd.read_csv(io.StringIO(csv))
    df['datetime'] = pd.to_datetime(df['time_str'], errors='coerce')
    for c in ['wse', 'wse_u', 'area_total', 'quality_f']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n_raw = len(df)
    # keep physically valid WSE and good/suspect quality (0=good, 1=suspect)
    df = df[(df.wse > FILL) & df.wse.notna() & df.datetime.notna()]
    df = df[df.quality_f.isin([0, 1])]
    df = df.rename(columns={'area_total': 'area_km2'})[
        ['datetime', 'wse', 'wse_u', 'area_km2', 'quality_f']].sort_values('datetime')
    df.to_csv(OUT / f'{name}_swot.csv', index=False)
    rows.append(dict(reservoir=name, lake_id=lid, n_raw=n_raw, n_valid=len(df),
                     wl_min=round(df.wse.min(), 2), wl_max=round(df.wse.max(), 2),
                     start=str(df.datetime.min().date()), end=str(df.datetime.max().date())))

rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
print(f"\nSaved {len(rep)} SWOT series → {OUT}")
