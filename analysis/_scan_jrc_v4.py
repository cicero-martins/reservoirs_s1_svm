"""
_scan_jrc_v4.py  —  Quick diagnostic of GEE_GlobalPilotV4_JRC exports.

Checks per reservoir:
  - n_months : total rows
  - n_valid  : rows with valid_frac >= 0.80
  - area_mean: mean jrc_area_ha (valid rows only)
  - area_max : max jrc_area_ha  (valid rows only)
  - date_range: first → last observation
  - flag     : problems worth inspecting

Run from project root:
  python analysis/_scan_jrc_v4.py
"""

import pathlib
import numpy as np
import pandas as pd

JRC_DIR       = pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC')
VALID_FRAC    = 0.80
MIN_VALID_MON = 12   # minimum months needed for KGE

# ── Load candidate list for reference ────────────────────────────────────────
CANDIDATES = pathlib.Path('analysis/global_pilot_v4_candidates.csv')
cand = pd.read_csv(CANDIDATES)[['name', 'ap_expected', 'area_ha_approx', 'country']]
cand = cand.set_index('name')

rows = []
# Deduplicate: Google Drive appends "(1)" when re-exported; keep the plain name.
import re as _re
_all = sorted(JRC_DIR.glob('JRC_area_*.csv'))
_seen = {}
for p in _all:
    key = _re.sub(r'\s*\(\d+\)$', '', p.stem)  # strip "(1)" suffix
    if key not in _seen or '(' not in p.stem:   # prefer the plain file
        _seen[key] = p
files = sorted(_seen.values(), key=lambda p: p.stem)

if not files:
    print(f'No files found in {JRC_DIR}')
    print('Download CSVs from Google Drive → GEE_GlobalPilotV4_JRC first.')
    raise SystemExit

for p in files:
    name = p.stem.replace('JRC_area_', '')
    df   = pd.read_csv(p, parse_dates=['date'])

    n_total = len(df)
    has_vf  = 'valid_frac' in df.columns
    df_v    = df[df['valid_frac'] >= VALID_FRAC] if has_vf else df

    area_valid = df_v['jrc_area_ha'].replace(0, np.nan).dropna()
    area_mean  = area_valid.mean() if len(area_valid) else np.nan
    area_max   = area_valid.max()  if len(area_valid) else np.nan

    date_min = df['date'].min().strftime('%Y-%m') if n_total else '—'
    date_max = df['date'].max().strftime('%Y-%m') if n_total else '—'

    # flags
    flags = []
    if n_total == 0:
        flags.append('EMPTY')
    if len(df_v) < MIN_VALID_MON:
        flags.append(f'only {len(df_v)} valid months')
    if not np.isnan(area_max) and area_max > 1000:
        flags.append(f'area_max={area_max:.0f}ha > 1000')
    if not np.isnan(area_mean) and area_mean < 50:
        flags.append(f'area_mean={area_mean:.0f}ha very small')

    ap_exp = cand.loc[name, 'ap_expected'] if name in cand.index else '?'

    rows.append({
        'name':       name,
        'ap':         ap_exp,
        'country':    cand.loc[name, 'country'] if name in cand.index else '?',
        'n_total':    n_total,
        'n_valid':    len(df_v),
        'area_mean':  round(area_mean, 0) if not np.isnan(area_mean) else np.nan,
        'area_max':   round(area_max, 0)  if not np.isnan(area_max)  else np.nan,
        'dates':      f'{date_min} to {date_max}',
        'flags':      '; '.join(flags),
    })

out = pd.DataFrame(rows)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f'\nJRC v4 scan — {len(files)} files found in {JRC_DIR}\n')
hdr = f'{"Name":<22} {"AP":>4} {"n_tot":>6} {"n_val":>6} {"mean_ha":>8} {"max_ha":>7}  {"dates":<22}  flags'
print(hdr)
print('-' * len(hdr))
for _, r in out.iterrows():
    flag_str = f'  ! {r["flags"]}' if r['flags'] else ''
    print(f'{r["name"]:<22} {r["ap"]:>4} {r["n_total"]:>6} {r["n_valid"]:>6} '
          f'{r["area_mean"]:>8.0f} {r["area_max"]:>7.0f}  {r["dates"]:<22}{flag_str}')

# ── Summary stats ─────────────────────────────────────────────────────────────
print('-' * len(hdr))
flagged = out[out['flags'] != '']
ok      = out[out['flags'] == '']
print(f'\nOK: {len(ok)} / {len(out)}   Flagged: {len(flagged)}')
print(f'Median valid months : {out["n_valid"].median():.0f}')
print(f'area_mean range     : {out["area_mean"].min():.0f} – {out["area_mean"].max():.0f} ha')

if len(flagged):
    print(f'\nFlagged ({len(flagged)}):')
    for _, r in flagged.iterrows():
        print(f'  {r["name"]:22}  {r["flags"]}')

# ── Missing / extra from Drive ────────────────────────────────────────────────
found_names = {p.stem.replace('JRC_area_', '') for p in files}
all_names   = set(cand.index)
missing = all_names - found_names
extra   = found_names - all_names
if missing:
    print(f'\nMissing from Drive ({len(missing)}): {", ".join(sorted(missing))}')
else:
    print(f'\nAll {len(all_names)} reservoirs present.')
if extra:
    print(f'Extra files in Drive (dropped from v4 list): {", ".join(sorted(extra))}')
