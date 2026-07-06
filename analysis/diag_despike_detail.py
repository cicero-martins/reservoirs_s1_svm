"""
diag_despike_detail.py

Close inspection of the JRC de-spike for a few reservoirs the user is unsure about.
For each: show every raw JRC month coloured by valid_frac, the NEW-filter kept series,
and mark EXACTLY which points the de-spike dropped (with their valid_frac). Also prints,
per point dropped, the reason (vf<0.90 pre-filter, or isolated-spike gated on vf<0.95).
Lets us judge whether removals are justified (contaminated) or wrong (real dynamics).
"""
import pathlib, re as _re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

JRC = [pathlib.Path('raw_data/GEE_GlobalPilotV4_JRC'),
       pathlib.Path('raw_data/GEE_GlobalPilotV4b/GEE_GlobalPilotV4_JRC'),
       pathlib.Path('raw_data/GEE_GlobalPilotV2c/GEE_GlobalPilotV2_JRC')]
NAMES = ['Egorlyskaia', 'Yamba', 'Wusijiang', 'Wushantou', 'Paraibuna']
VF_MIN, VF_GATE = 0.90, 0.95

def jrcp(n):
    for d in JRC:
        c = sorted(d.glob(f'JRC_area_{n}*.csv'))
        pl = [p for p in c if not _re.search(r'\s*\(\d+\)', p.stem)]
        p = pl[0] if pl else (c[0] if c else None)
        if p: return p
    return None

def despike_mask(a, vf, k=4.0, minrel=0.12, vf_gate=VF_GATE):
    a = np.asarray(a, float); n = len(a)
    if n < 5: return np.ones(n, bool), np.zeros(n)
    interp = a.copy(); interp[1:-1] = 0.5 * (a[:-2] + a[2:])
    resid = np.abs(a - interp); resid[0] = resid[-1] = 0.0
    scale = np.median(resid[resid > 0]) if np.any(resid > 0) else 1.0
    thr = max(k * scale, minrel * np.median(a))
    is_spike = (resid > thr) & (np.asarray(vf, float) < vf_gate)
    return ~is_spike, resid

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(NAMES), 1, figsize=(15, 3.0 * len(NAMES)))
for ax, n in zip(np.atleast_1d(axes), NAMES):
    p = jrcp(n)
    if p is None:
        ax.set_title(f'{n}: NOT FOUND'); continue
    df = pd.read_csv(p, parse_dates=['date'])
    df = df[df['date'] <= '2021-12-31'].sort_values('date').reset_index(drop=True)
    # NEW pipeline stepwise
    vfpass = df['valid_frac'] >= VF_MIN
    step = df[vfpass].reset_index(drop=True)
    m, sd = step['jrc_area_ha'].mean(), step['jrc_area_ha'].std()
    keep3 = np.abs(step['jrc_area_ha'] - m) <= 3.0 * sd if sd > 0 else np.ones(len(step), bool)
    step2 = step[keep3].reset_index(drop=True)
    dmask, resid = despike_mask(step2['jrc_area_ha'].values, step2['valid_frac'].values)
    final = step2[dmask].reset_index(drop=True)

    # raw points coloured by valid_frac
    sc = ax.scatter(df['date'], df['jrc_area_ha'], c=df['valid_frac'], cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, s=28, zorder=3, edgecolors='#888', linewidths=0.3)
    ax.plot(final['date'], final['jrc_area_ha'], '-', color='#1f77b4', lw=1.3, zorder=4,
            label='NEW kept (vf≥.9 + gated de-spike)')
    # mark removed-by-despike
    removed = step2[~dmask]
    ax.scatter(removed['date'], removed['jrc_area_ha'], marker='x', s=110, color='k',
               linewidths=2, zorder=6, label='de-spiked (dropped)')
    # mark removed by vf<0.9 pre-filter
    lowvf = df[df['valid_frac'] < VF_MIN]
    ax.scatter(lowvf['date'], lowvf['jrc_area_ha'], marker='o', s=70, facecolors='none',
               edgecolors='purple', linewidths=1.3, zorder=5, label='dropped vf<0.90')
    ax.set_title(f'{n}  (raw n={len(df)} → vf≥.9 n={len(step)} → de-spiked n={len(final)})',
                 fontsize=10, fontweight='bold')
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc='best')
    ax.set_ylabel('JRC area (ha)', fontsize=8)

    print(f'\n=== {n} ===  raw={len(df)}  vf<0.9 dropped={int((~vfpass).sum())}  '
          f'de-spiked dropped={int((~dmask).sum())}  final={len(final)}')
    if len(removed):
        print('  de-spiked points (date, area, valid_frac):')
        for _, r in removed.iterrows():
            print(f"    {r['date']:%Y-%m}  area={r['jrc_area_ha']:8.1f}  vf={r['valid_frac']:.2f}")
    # what the LOW-vf points look like (were they extreme?)
    if len(lowvf):
        print(f'  vf<0.90 points: areas ' +
              ', '.join(f"{a:.0f}(vf{v:.2f})" for a, v in zip(lowvf['jrc_area_ha'], lowvf['valid_frac'])))

plt.colorbar(sc, ax=np.atleast_1d(axes).tolist(), label='valid_frac', shrink=0.6, pad=0.01)
fig.suptitle('JRC de-spike inspection — points coloured by valid_frac (red=low coverage)',
             fontsize=13, fontweight='bold')
OUT = pathlib.Path('analysis/method_comparison_output/despike_detail.png')
fig.savefig(OUT, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'\nSaved: {OUT}')
