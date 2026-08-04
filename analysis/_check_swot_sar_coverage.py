"""
_check_swot_sar_coverage.py (2026-08-03, diagnostic only)

Per-reservoir figure (2 subplots) for Olivo/Nicoletti/Castello/Arancio: SWOT
water-level timeseries (top) and continuous SAR area timeseries (bottom),
sharing the x-axis, to see directly where the two overlap/gap in time --
motivated by finding that the SAR-series extension (2026-08-03) widened the
common calibration band but did not fully close it for 3 of the 4 (Nicoletti
especially, whose band only spans the top 29% of its true SWOT range).
"""
import pathlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = pathlib.Path('analysis/schwatke_output')
NAMES = ['Olivo', 'Nicoletti', 'Castello', 'Arancio']
SAR_DIR = pathlib.Path('raw_data/exportSicilyExtended/GEE_SicilyExtended_VVotsu')

for name in NAMES:
    swot = pd.read_csv(f'validation_data/SWOT/{name}_swot.csv')
    swot['datetime'] = pd.to_datetime(swot['datetime']).dt.tz_localize(None)
    sar = pd.read_csv(SAR_DIR / f'SAR_area_{name}.csv')
    sar['date'] = pd.to_datetime(sar['date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    good = swot[swot.quality_f == 1] if 'quality_f' in swot.columns else swot
    bad = swot[swot.quality_f == 0] if 'quality_f' in swot.columns else swot.iloc[0:0]
    ax1.scatter(good.datetime, good.wse, s=16, color='#1565c0', label='SWOT wse (quality_f=1)')
    ax1.scatter(bad.datetime, bad.wse, s=16, color='#c62828', marker='x', label='SWOT wse (quality_f=0)')
    ax1.set_ylabel('SWOT WSE (m ASL)')
    ax1.set_title(f'{name}: SWOT water level vs. continuous SAR area', fontsize=12, loc='left')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)

    ax2.plot(sar.date, sar.area_ha, color='#6a1b9a', lw=0.9, marker='.', ms=3)
    ax2.axvline(pd.Timestamp('2025-12-31'), color='gray', ls=':', lw=1.2)
    ax2.text(pd.Timestamp('2025-12-31'), sar.area_ha.max(), '  old cutoff', fontsize=7, color='gray', va='top')
    ax2.set_ylabel('SAR area (ha)')
    ax2.set_xlabel('Date')
    ax2.grid(alpha=0.3)

    xlo = min(swot.datetime.min(), sar.date.min())
    xhi = max(swot.datetime.max(), sar.date.max())
    ax1.set_xlim(xlo - pd.Timedelta(days=30), xhi + pd.Timedelta(days=30))

    fig.tight_layout()
    fig.savefig(OUT / f'_diag_swot_sar_coverage_{name}.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/_diag_swot_sar_coverage_{name}.png')
