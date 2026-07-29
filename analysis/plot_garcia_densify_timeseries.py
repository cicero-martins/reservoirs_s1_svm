"""
plot_garcia_densify_timeseries.py (2026-07-29)

Visual diagnostic: SAR area and assigned water level over time for all 52 Garcia
densification candidates, colour/shape-coded by source (gauge/swot/none) and
excluded/kept, to review the reselection before trusting the new capture-ratio
number. Mirrors plot_rosamarina_densify_timeseries.py.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path('analysis/schwatke_output')
BAD = {'2026-02-12', '2026-03-26'}

df = pd.read_csv(OUT_DIR / 'garcia_densify_prototype_pairs.csv', parse_dates=['date'])
df = df.sort_values('date')
df['excluded'] = df['date'].dt.strftime('%Y-%m-%d').isin(BAD)

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

style = {'gauge': ('tab:blue', 'o'), 'swot': ('tab:orange', '^'), 'model': ('tab:green', 's'),
         'none': ('tab:red', 'x')}

ax = axes[0]
for src, (color, marker) in style.items():
    sub = df[(df['source'] == src) & (~df['excluded'])]
    if len(sub):
        ax.scatter(sub['date'], sub['area_ha'], c=color, marker=marker, s=55, alpha=0.9,
                   label=f'{src}')
excl = df[df['excluded']]
ax.scatter(excl['date'], excl['area_ha'], c='none', marker='o', s=220,
           edgecolors='red', linewidths=2.2, label='EXCLUDED (bad mask)', zorder=5)
ax.plot(df['date'], df['area_ha'], '-', color='gray', alpha=0.3, zorder=0)
ax.set_ylabel('SAR area (ha)')
ax.set_title('Garcia Period-B: SAR area, all 52 candidates')
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.3)

ax = axes[1]
for src, (color, marker) in style.items():
    sub = df[(df['source'] == src) & df['wl_m'].notna() & (~df['excluded'])]
    if len(sub):
        ax.scatter(sub['date'], sub['wl_m'], c=color, marker=marker, s=55, alpha=0.9)
excl2 = df[df['excluded'] & df['wl_m'].notna()]
ax.scatter(excl2['date'], excl2['wl_m'], c='none', marker='o', s=220,
           edgecolors='red', linewidths=2.2, zorder=5)
ax.set_ylabel('Assigned WL (m)')
ax.set_xlabel('Date')
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
out_png = OUT_DIR / 'garcia_densify_timeseries.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')

print('\nProduction (new) 10-date selection vs area/WL:')
prod_dates = {'2025-08-10','2025-08-22','2025-10-03','2025-11-20','2026-02-06',
              '2026-02-18','2026-02-24','2026-03-14','2026-04-07','2026-05-19'}
sub = df[df['date'].dt.strftime('%Y-%m-%d').isin(prod_dates)].sort_values('wl_m')
print(sub[['date','area_ha','wl_m','source']].to_string(index=False))
