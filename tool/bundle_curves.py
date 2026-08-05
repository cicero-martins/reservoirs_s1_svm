"""
tool/bundle_curves.py — export the extended reservoirs' updated survey curves to
compact CSVs bundled for deployment.

Arancio, Castello, Olivo and Nicoletti carry their updated official curves in large
workbooks under NEWCURVE_EXT (a local path, ~1.6 MB total, and each workbook holds
sheets unrelated to this study). The deployed Streamlit app has no access to that
path, so those four reservoirs showed no curves at all. This script extracts only
the (quota, area, volume) columns bathymetry.py actually reads, using the very same
EXT_CURVE_SPEC blocks, and writes them to tool/data/curves/updated_{kind}.csv in the
schema already used by garcia_2026.csv / rosamarina_2025.csv (quota_m, area_m2,
vol_m3).

Re-run only if a source workbook is replaced by the water authority.

Run:  python tool/bundle_curves.py
"""

import sys
import pandas as pd

import bathymetry as bt


def extract(kind):
    """Parse one updated curve straight from its source workbook, exactly as
    bathymetry.updated_curve() does for the EXT_CURVE_SPEC branch."""
    pat, sheet, blocks = bt.EXT_CURVE_SPEC[kind]
    hits = [h for h in __import__('glob').glob(str(bt.NEWCURVE_EXT / pat))
            if h.lower().endswith(('.xls', '.xlsx'))]
    if not hits:
        return None
    raw = pd.read_excel(hits[0], sheet_name=sheet, header=None, engine='openpyxl')
    parts = []
    for qc, vc, ac in blocks:
        p = raw[[qc, vc, ac]].apply(pd.to_numeric, errors='coerce').dropna()
        p.columns = ['quota', 'vol_m3', 'area_m2']
        parts.append(p)
    u = pd.concat(parts).drop_duplicates(subset='quota')
    u = u[(u.quota > 50) & (u.quota < 1000)].sort_values('quota')
    return u.rename(columns={'quota': 'quota_m'})[['quota_m', 'area_m2', 'vol_m3']]


def main():
    bt.CURVE_BUNDLE.mkdir(parents=True, exist_ok=True)
    rc = 0
    for kind in bt.EXT_CURVE_SPEC:
        u = extract(kind)
        if u is None:
            print(f'  MISSING source workbook for {kind}')
            rc = 1
            continue
        out = bt.CURVE_BUNDLE / f'updated_{kind}.csv'
        u.to_csv(out, index=False)
        print(f'  {kind:20s} {len(u):5d} rows  '
              f'{u.quota_m.min():.2f}-{u.quota_m.max():.2f} m  '
              f'{out.stat().st_size / 1024:.0f} KB')
    return rc


if __name__ == '__main__':
    sys.exit(main())
