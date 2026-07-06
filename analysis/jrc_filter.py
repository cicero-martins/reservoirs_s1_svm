"""
jrc_filter.py — shared JRC reference loader with the validated valid_frac-gated de-spike.

Fixes the SAR/JRC asymmetry: the SAR side gets full clean+smooth+local-outlier removal, so
the JRC reference should get comparable outlier handling instead of only valid_frac + a
global σ clip (which leaks isolated cloud/partial-coverage spikes into the KGE obs).

FINAL filter (validated 6 Jul, N=59 dual-vs-JRC: 32 improved / 5 worsened, KGE<0.5 23→18):
  valid_frac ≥ 0.90
  global 3σ clip (loose net)
  isolated single-month spike removal (linear-neighbour-interp residual) GATED on
  valid_frac < 0.95 — only drop excursions with contamination evidence; real drawdowns
  happen at near-full coverage (vf≈1) and are kept.

Set despike=False to reproduce the OLD filter (valid_frac≥0.80 + 2.5σ).
"""
import re as _re
import numpy as np
import pandas as pd

VF_MIN_NEW = 0.90
VF_MIN_OLD = 0.80


def _resolve(dirs, name):
    for d in dirs:
        c = sorted(d.glob(f'JRC_area_{name}*.csv'))
        pl = [p for p in c if not _re.search(r'\s*\(\d+\)', p.stem)]
        p = pl[0] if pl else (c[0] if c else None)
        if p:
            return p
    return None


def _despike_gated(a, vf, k=4.0, minrel=0.12, vf_gate=0.95):
    """Keep-mask. Drop point i iff it is an ISOLATED excursion from the linear interpolation
    of its neighbours AND valid_frac_i < vf_gate (contamination evidence)."""
    a = np.asarray(a, float); n = len(a)
    if n < 5:
        return np.ones(n, bool)
    interp = a.copy(); interp[1:-1] = 0.5 * (a[:-2] + a[2:])
    resid = np.abs(a - interp); resid[0] = resid[-1] = 0.0
    scale = np.median(resid[resid > 0]) if np.any(resid > 0) else 1.0
    thr = max(k * scale, minrel * np.median(a))
    is_spike = resid > thr
    if vf is not None:
        is_spike &= (np.asarray(vf, float) < vf_gate)
    return ~is_spike


def load_jrc_monthly(name, dirs, end='2021-12-31', despike=True):
    """Return monthly JRC as DataFrame[date, jrc_area_ha] (date = month start), filtered.
    None if unavailable/empty."""
    p = _resolve(dirs, name)
    if p is None:
        return None
    try:
        df = pd.read_csv(p, parse_dates=['date'])
    except pd.errors.EmptyDataError:
        return None
    df = df[df['date'] <= end].sort_values('date').reset_index(drop=True)
    vf_min = VF_MIN_NEW if despike else VF_MIN_OLD
    if 'valid_frac' in df.columns:
        df = df[df['valid_frac'] >= vf_min].copy()
    if df.empty:
        return None
    m, sd = df['jrc_area_ha'].mean(), df['jrc_area_ha'].std()
    if despike:
        if sd > 0:
            df = df[np.abs(df['jrc_area_ha'] - m) <= 3.0 * sd]
        vf = df['valid_frac'].values if 'valid_frac' in df.columns else None
        df = df[_despike_gated(df['jrc_area_ha'].values, vf)]
    else:
        if sd > 0:
            df = df[np.abs(df['jrc_area_ha'] - m) <= 2.5 * sd]
    if df.empty:
        return None
    return df[['date', 'jrc_area_ha']].dropna().reset_index(drop=True)
