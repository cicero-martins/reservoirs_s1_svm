"""
fullrs_wl_ladder.py  (BACKBONE — the water-level-substitution test)

100%-remote-sensing volume: keep the Sentinel-1 SAR area and the hypsometric
reconstruction identical, and SWAP the water-level source between the TWO
INDEPENDENT TIERS:
      in-situ GAUGE   vs   SWOT swath altimetry (track-independent).
If the SWOT reconstruction matches the gauge one AND the independent field/survey
truth, the gauge is dispensable → gauge-free, track-free reservoir volume.

IMPORTANT (verified): DAHITI is NOT an independent third source here. For these
small reservoirs DAHITI has data only from 2023-07 (SWOT era) and its WSE matches
SWOT within ~0.2–0.6 m at >96% of epochs — i.e. DAHITI = a reprocessed-SWOT product.
It is reported ONLY as a cross-check that the two access paths (DAHITI vs Hydrocron)
agree, never as an independent rung.

Per reservoir + tier: pair SAR area with WL (±MAX_DT d) → fit A=a·(h-h0)^b →
integrate over the common gauge∩SWOT band → band volume (Mm³). Compare to each
other and to the survey truth (Garcia echosounder; Poma/Rosamarina/Arancio survey
curves). Pozzillo/Ancipa have no modern survey → gauge↔SWOT consistency only.

Outputs: analysis/schwatke_output/fullrs/  (fullrs_ladder.csv, fullrs_ladder.png)
"""

import sys, glob, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str((pathlib.Path(__file__).resolve().parent.parent / 'tool')))
import bathymetry as bt

REPO  = pathlib.Path('.')
GAUGE = REPO / 'analysis' / 'schwatke_output' / 'gauge_downloads'
DAH   = REPO / 'validation_data' / 'DAHITI'
SWOT  = REPO / 'validation_data' / 'SWOT'
OUT   = REPO / 'analysis' / 'schwatke_output' / 'fullrs'
OUT.mkdir(parents=True, exist_ok=True)
MAX_DT = 10

# tier: 'truth' = has an independent modern survey; 'consist' = gauge↔SWOT only
RES = {
    'Garcia':     dict(ap=167.7, gauge='garcia_idrometro_radar.csv', dahiti='42123_Garcia_wl.csv',     tier='truth'),
    'Poma':       dict(ap=190.1, gauge='poma_wl.csv',                dahiti='42134_Poma_wl.csv',       tier='truth'),
    'Rosamarina': dict(ap=187.4, gauge='rosamarina_wl.csv',          dahiti='42122_Rosamarina_wl.csv', tier='truth'),
    'Arancio':    dict(ap=182.2, gauge='arancio_wl.csv',             dahiti=None,                      tier='truth'),
    'Pozzillo':   dict(ap=240.5, gauge='pozzillo_wl.csv',            dahiti=None,                      tier='consist'),
    'Ancipa':     dict(ap=90.5,  gauge='ancipa_livello_secca.csv',   dahiti=None,                      tier='consist'),
}
SAR_CANDIDATES = [
    'raw_data/GEE_GlobalPilotV4_final/GEE_GlobalPilotV4_VVotsu/SAR_area_{n}.csv',
    'raw_data/GEE_GlobalPilotV2a/SAR_area_{n}.csv',
    'raw_data/exportSicilyExtended/GEE_SicilyExtended_VVotsu/SAR_area_{n}.csv',
]


def load_sar(name):
    for pat in SAR_CANDIDATES:
        p = REPO / pat.format(n=name)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])[['date', 'area_ha']].sort_values('date')
        if len(df) and df['date'].max().year >= 2023:
            return df
    return None


def _wl(df, dcol, wcol):
    df = df[[dcol, wcol]].copy(); df.columns = ['date', 'wl']
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    return df.dropna().groupby('date', as_index=False).wl.mean().sort_values('date')

def load_gauge(f):
    g = pd.read_csv(GAUGE / f); g.columns = [c.strip().lower() for c in g.columns]
    return _wl(g, next(c for c in g.columns if 'time' in c or 'date' in c),
                  next(c for c in g.columns if 'wl' in c or 'quota' in c or 'value' in c))

def load_dahiti(f):
    d = pd.read_csv(DAH / f)
    return _wl(d, next(c for c in d.columns if 'date' in c.lower() or 'time' in c.lower()),
                  next(c for c in d.columns if 'wse' in c.lower() or 'water' in c.lower()))

def load_swot(name):
    p = SWOT / f'{name}_swot.csv'
    return _wl(pd.read_csv(p), 'datetime', 'wse') if p.exists() else None


def survey_truth(name):
    """(area_ha_interp, vol_Mm3_interp) for the modern survey, or None."""
    if name == 'Arancio':   # 2022 topo-bathy curve (BASE sheet: quota|vol_mc|area_mq)
        f = [h for h in glob.glob('C:/Users/Unipa/Documents/GEE/Data/NewCurves/ARANCIO*')
             if h.lower().endswith(('.xls', '.xlsx'))][0]
        r = pd.read_excel(f, sheet_name='BASE', header=None, engine='openpyxl')[[0, 1, 2]]
        r.columns = ['q', 'vol_mc', 'area_mq']
        r = r.apply(pd.to_numeric, errors='coerce').dropna()
        r = r[(r.q > 100) & (r.q < 300)].sort_values('q')
        return (interp1d(r.q, r.area_mq / 1e4, bounds_error=False, fill_value='extrapolate'),
                interp1d(r.q, r.vol_mc / 1e6, bounds_error=False, fill_value='extrapolate'))
    return bt.updated_curve(name)   # core 5 (echosounder / updated curve)


def power_law(h, a, b, h0):
    return a * np.maximum(h - h0, 1e-6) ** b

def fit(sar, wl):
    m = pd.merge_asof(sar, wl, on='date', tolerance=pd.Timedelta(f'{MAX_DT}D'),
                      direction='nearest').dropna()
    m = m[m.area_ha > 0]
    if len(m) < 8:
        return None
    h, A = m.wl.values, m.area_ha.values
    h0_hi = float(h.min()) - 0.01
    try:
        popt, _ = curve_fit(power_law, h, A, p0=[1.0, 1.5, h0_hi - 5],
                            bounds=([0, 0.5, h0_hi - 60], [1e6, 5, h0_hi]), maxfev=30000)
    except Exception:
        return None
    r2 = 1 - np.sum((A - power_law(h, *popt))**2) / np.sum((A - A.mean())**2)
    return dict(popt=popt, n=len(m), wl=(float(h.min()), float(h.max())), r2=r2, pairs=m)

def band_vol(popt, hs):
    return float(cumulative_trapezoid(power_law(hs, *popt), hs, initial=0)[-1] * 0.01)

def wl_rmse(a, b):
    m = pd.merge_asof(a.rename(columns={'wl': 'x'}), b.rename(columns={'wl': 'y'}),
                      on='date', tolerance=pd.Timedelta('1D'), direction='nearest').dropna()
    return (len(m), float(np.sqrt(((m.x - m.y)**2).mean())), float((m.x - m.y).mean())) if len(m) else (0, np.nan, np.nan)


rows, panels = [], {}
for name, cfg in RES.items():
    sar = load_sar(name)
    if sar is None:
        print(f'{name}: no SAR ≥2023 series'); continue
    g, sw = load_gauge(cfg['gauge']), load_swot(name)
    fg, fs = fit(sar, g), fit(sar, sw)
    if fg is None or fs is None:
        print(f'{name}: need both gauge & SWOT fits (g={fg is not None}, s={fs is not None})'); continue

    lo = max(fg['wl'][0], fs['wl'][0]); hi = min(fg['wl'][1], fs['wl'][1])
    hs = np.linspace(lo, hi, 100)
    vol_g, vol_s = band_vol(fg['popt'], hs), band_vol(fs['popt'], hs)
    _, sw_rmse, _ = wl_rmse(sw, g)   # SWOT vs gauge WL

    truth = survey_truth(name) if cfg['tier'] == 'truth' else None
    vol_t = float(truth[1](hi) - truth[1](lo)) if truth is not None else np.nan

    rec = dict(reservoir=name, ap_m=cfg['ap'], tier=cfg['tier'], band=f'{lo:.1f}-{hi:.1f}',
               vol_gauge=round(vol_g, 2), vol_SWOT=round(vol_s, 2),
               vol_SWOT_minus_gauge=round(vol_s - vol_g, 2), wl_SWOT_vs_gauge_rmse_m=round(sw_rmse, 2),
               vol_survey=None if np.isnan(vol_t) else round(vol_t, 2),
               vol_SWOT_vs_survey=None if np.isnan(vol_t) else round(vol_s - vol_t, 2),
               vol_gauge_vs_survey=None if np.isnan(vol_t) else round(vol_g - vol_t, 2))
    # DAHITI = reprocessed-SWOT cross-check (NOT an independent tier)
    if cfg['dahiti']:
        d = load_dahiti(cfg['dahiti'])
        nmatch, d_sw_rmse, d_sw_bias = wl_rmse(d, sw)
        fd = fit(sar, d)
        rec['xcheck_DAHITI_vs_SWOT_wse_m'] = round(d_sw_bias, 2)
        rec['xcheck_DAHITI_matches_SWOT'] = f'{nmatch}/{len(d)}'
        rec['xcheck_vol_DAHITI'] = round(band_vol(fd['popt'], hs), 2) if fd else None
    rows.append(rec)
    panels[name] = dict(cfg=cfg, fg=fg, fs=fs, hs=hs,
                        a_tru=(truth[0](hs) if (truth is not None and bt.RESERVOIRS.get(name, {}).get('updated') != 'poma_new') else None))

df = pd.DataFrame(rows)
df.to_csv(OUT / 'fullrs_ladder.csv', index=False)
pd.set_option('display.width', 220, 'display.max_columns', 40)
print(df.to_string(index=False))

tr = df[df.tier == 'truth']
print("\nTWO-TIER TEST (in-situ gauge vs SWOT satellite altimetry; band volume Mm³):")
for _, r in df.iterrows():
    s = f"  {r.reservoir:11s} A/P {r.ap_m:>5.0f} | gauge {r.vol_gauge:6.1f}  SWOT {r.vol_SWOT:6.1f}  (Δ {r.vol_SWOT_minus_gauge:+.1f}, WL RMSE {r.wl_SWOT_vs_gauge_rmse_m} m)"
    if pd.notna(r.vol_survey): s += f"  | survey {r.vol_survey:.1f}  SWOT−survey {r.vol_SWOT_vs_survey:+.1f}"
    print(s)
if len(tr):
    print(f"\nSWOT vs field/survey truth (4 reservoirs): "
          f"mean |Δ| {tr.vol_SWOT_vs_survey.abs().mean():.1f} Mm³ "
          f"({', '.join(f'{n} {v:+.1f}' for n,v in zip(tr.reservoir, tr.vol_SWOT_vs_survey))}).")
xc = df.dropna(subset=['xcheck_DAHITI_vs_SWOT_wse_m']) if 'xcheck_DAHITI_vs_SWOT_wse_m' in df else df.iloc[:0]
if len(xc):
    print("Cross-check — DAHITI = reprocessed SWOT (NOT independent): WSE Δ "
          f"{', '.join(f'{r.reservoir} {r.xcheck_DAHITI_vs_SWOT_wse_m:+.2f}m ({r.xcheck_DAHITI_matches_SWOT})' for _,r in xc.iterrows())}.")
print(f"\nSaved: {OUT/'fullrs_ladder.csv'}")

# ── Figure: gauge (in-situ) vs SWOT (altimetry) + survey truth ─────────────────
order = list(RES.keys())
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.24)
fig.suptitle('Full-RS water-level substitution — in-situ GAUGE vs SWOT satellite altimetry '
             '(SAR area identical). DAHITI = reprocessed SWOT (cross-check, not shown).',
             fontsize=11.5, fontweight='bold')
for i, name in enumerate(order):
    if name not in panels: continue
    P = panels[name]; ax = fig.add_subplot(gs[i])
    pg = P['fg']['pairs']
    ax.scatter(pg.area_ha, pg.wl, s=10, c='#cccccc', alpha=0.6, zorder=1, label='SAR–gauge pairs')
    ax.plot(power_law(P['hs'], *P['fg']['popt']), P['hs'], color='#1565c0', lw=2.4, label='fit · gauge (in-situ)')
    ax.plot(power_law(P['hs'], *P['fs']['popt']), P['hs'], color='#6a1b9a', lw=2.4, label='fit · SWOT (altimetry)')
    if P['a_tru'] is not None:
        ax.plot(P['a_tru'], P['hs'], color='#2e7d32', lw=2, label='survey truth')
    r = df[df.reservoir == name].iloc[0]
    extra = f" · SWOT−survey {r.vol_SWOT_vs_survey:+.1f} Mm³" if pd.notna(r.vol_survey) else '  (no survey)'
    ax.set_title(f"{name} (A/P {P['cfg']['ap']:.0f} m){extra}", fontsize=9)
    ax.set_xlabel('Area (ha)'); ax.set_ylabel('Water level (m ASL)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

fig.subplots_adjust(top=0.9)
fig.savefig(OUT / 'fullrs_ladder.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT/'fullrs_ladder.png'}")
