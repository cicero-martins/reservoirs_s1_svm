"""
_scan_sar_v4.py  -  Quick diagnostic of GEE_GlobalPilotV4b SAR exports.

Checks per reservoir:
  - n_total   : total S1 acquisitions
  - n_nz      : rows with area_ha > 5 ha (non-trivial water extent)
  - area_mean : mean SAR area (non-zero rows only)
  - area_max  : max SAR area
  - ap_m      : static A/P from JRC max_extent (constant per reservoir)
  - orbit_dir : ascending / descending / mixed
  - flag      : problems worth inspecting

Run from project root:
  python analysis/_scan_sar_v4.py
"""

import pathlib
import re as _re
import numpy as np
import pandas as pd

SAR_DIR   = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4')
MIN_VALID = 12      # minimum non-zero acquisitions for usable KGE
MIN_AREA  = 5       # ha threshold to distinguish real water from noise

# ── Load candidate list ───────────────────────────────────────────────────────
CANDIDATES = pathlib.Path('analysis/global_pilot_v4_candidates.csv')
cand = pd.read_csv(CANDIDATES)[['name', 'ap_expected', 'area_ha_approx', 'country']]
cand = cand.set_index('name')

# ── Load SAR files (deduplicate Drive "(1)" suffix) ───────────────────────────
_all = sorted(SAR_DIR.glob('SAR_area_*.csv'))
_seen = {}
for p in _all:
    key = _re.sub(r'\s*\(\d+\)$', '', p.stem)
    if key not in _seen or '(' not in p.stem:
        _seen[key] = p
sar_files = {k: v for k, v in _seen.items()}

rows = []
for name in cand.index:
    key  = f'SAR_area_{name}'
    path = sar_files.get(key)

    if path is None:
        rows.append({
            'name': name, 'ap': cand.loc[name, 'ap_expected'],
            'country': cand.loc[name, 'country'],
            'n_total': 0, 'n_nz': 0,
            'area_mean': np.nan, 'area_max': np.nan,
            'ap_m': np.nan, 'orbit': '—', 'dates': '—',
            'flags': 'MISSING FILE',
        })
        continue

    try:
        df = pd.read_csv(path, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    n_total = len(df)

    if n_total == 0:
        rows.append({
            'name': name, 'ap': cand.loc[name, 'ap_expected'],
            'country': cand.loc[name, 'country'],
            'n_total': 0, 'n_nz': 0,
            'area_mean': np.nan, 'area_max': np.nan,
            'ap_m': np.nan, 'orbit': '—', 'dates': '—',
            'flags': 'EMPTY',
        })
        continue

    df_nz    = df[df['area_ha'] > MIN_AREA]
    n_nz     = len(df_nz)
    area_v   = df_nz['area_ha'] if n_nz else pd.Series(dtype=float)
    area_mean = area_v.mean() if n_nz else np.nan
    area_max  = area_v.max()  if n_nz else np.nan
    ap_m      = df['ap_m'].iloc[0] if 'ap_m' in df.columns else np.nan

    date_min = df['date'].min().strftime('%Y-%m') if n_total else '—'
    date_max = df['date'].max().strftime('%Y-%m') if n_total else '—'

    dirs = df['passDirection'].dropna().unique() if 'passDirection' in df.columns else []
    orbit = '/'.join(sorted(set(d[:3].upper() for d in dirs))) if len(dirs) else '—'

    # flags
    flags = []
    if n_total == 0:
        flags.append('EMPTY')
    elif n_nz == 0:
        flags.append('all area=0')
    elif n_nz < MIN_VALID:
        flags.append(f'only {n_nz} non-zero acq')
    if not np.isnan(area_mean) and area_mean < 20:
        flags.append(f'area_mean={area_mean:.0f}ha tiny')
    if not np.isnan(area_max) and area_max > 3000:
        flags.append(f'area_max={area_max:.0f}ha huge')
    if np.isnan(ap_m) or ap_m == 0:
        flags.append('ap_m missing')

    rows.append({
        'name':      name,
        'ap':        cand.loc[name, 'ap_expected'],
        'country':   cand.loc[name, 'country'],
        'n_total':   n_total,
        'n_nz':      n_nz,
        'area_mean': round(area_mean, 0) if not np.isnan(area_mean) else np.nan,
        'area_max':  round(area_max,  0) if not np.isnan(area_max)  else np.nan,
        'ap_m':      round(ap_m, 1)      if not np.isnan(ap_m)      else np.nan,
        'orbit':     orbit,
        'dates':     f'{date_min} to {date_max}',
        'flags':     '; '.join(flags),
    })

out = pd.DataFrame(rows)

# ── Print ─────────────────────────────────────────────────────────────────────
print(f'\nSAR v4 scan - {len(sar_files)} files in {SAR_DIR}\n')
hdr = f'{"Name":<22} {"AP":>4} {"n_tot":>6} {"n_nz":>5} {"mean_ha":>8} {"max_ha":>7} {"ap_m":>7}  {"orbit":<5}  {"dates":<22}  flags'
print(hdr)
print('-' * len(hdr))
for _, r in out.iterrows():
    flag_str = f'  ! {r["flags"]}' if r['flags'] else ''
    ap_m_str = f'{r["ap_m"]:7.1f}' if not (isinstance(r['ap_m'], float) and np.isnan(r['ap_m'])) else '    nan'
    area_mean_str = f'{r["area_mean"]:8.0f}' if not (isinstance(r['area_mean'], float) and np.isnan(r['area_mean'])) else '     nan'
    area_max_str  = f'{r["area_max"]:7.0f}' if not (isinstance(r['area_max'], float) and np.isnan(r['area_max'])) else '    nan'
    print(f'{r["name"]:<22} {r["ap"]:>4} {r["n_total"]:>6} {r["n_nz"]:>5}'
          f' {area_mean_str} {area_max_str} {ap_m_str}  {r["orbit"]:<5}  {r["dates"]:<22}{flag_str}')

print('-' * len(hdr))
flagged = out[out['flags'] != '']
ok      = out[out['flags'] == '']
print(f'\nOK: {len(ok)} / {len(out)}   Flagged: {len(flagged)}')
if len(out['n_nz']) > 0:
    print(f'Median non-zero acq : {out["n_nz"].median():.0f}')
    print(f'ap_m range          : {out["ap_m"].min():.0f} - {out["ap_m"].max():.0f} m')
    print(f'area_mean range     : {out["area_mean"].min():.0f} - {out["area_mean"].max():.0f} ha')

if len(flagged):
    print(f'\nFlagged ({len(flagged)}):')
    for _, r in flagged.iterrows():
        print(f'  {r["name"]:22}  {r["flags"]}')

# ── Missing / extra ───────────────────────────────────────────────────────────
found_names = {k.replace('SAR_area_', '') for k in sar_files}
all_names   = set(cand.index)
missing = all_names - found_names
extra   = found_names - all_names
if missing:
    print(f'\nMissing from Drive ({len(missing)}): {", ".join(sorted(missing))}')
else:
    print(f'\nAll {len(all_names)} candidates have SAR files.')
if extra:
    print(f'Extra files (dropped reservoirs): {", ".join(sorted(extra))}')
