"""
plot_rosamarina_densify_timeseries.py (2026-07-29)

Visual diagnostic: SAR area and assigned water level over time for all 41
Rosamarina densification candidates, colour/shape-coded by source
(gauge/swot/none) and original-vs-new, to check the area<->level relationship
directly before attributing DEM disagreement to mask noise.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path('analysis/schwatke_output')
df = pd.read_csv(OUT_DIR / 'rosamarina_densify_prototype_pairs.csv', parse_dates=['date'])
df = df.sort_values('date')

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

style = {'gauge': ('tab:blue', 'o'), 'swot': ('tab:orange', '^'), 'model': ('tab:green', 's'),
         'none': ('tab:red', 'x')}

ax = axes[0]
for src, (color, marker) in style.items():
    sub = df[df['source'] == src]
    if len(sub) == 0:
        continue
    for is_new, alpha, ec in [(False, 1.0, 'k'), (True, 0.9, None)]:
        s2 = sub[sub['is_new'] == is_new]
        if len(s2) == 0:
            continue
        ax.scatter(s2['date'], s2['area_ha'], c=color, marker=marker,
                   s=90 if not is_new else 60, alpha=alpha,
                   edgecolors=ec, linewidths=1.2 if not is_new else 0,
                   label=f'{src} ({"orig" if not is_new else "new"})')
ax.plot(df['date'], df['area_ha'], '-', color='gray', alpha=0.3, zorder=0)
ax.set_ylabel('SAR area (ha)')
ax.set_title('Rosamarina Period-B: SAR area, all 41 candidates')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
for src, (color, marker) in style.items():
    sub = df[(df['source'] == src) & df['wl_m'].notna()]
    if len(sub) == 0:
        continue
    for is_new, alpha in [(False, 1.0), (True, 0.9)]:
        s2 = sub[sub['is_new'] == is_new]
        if len(s2) == 0:
            continue
        ax.scatter(s2['date'], s2['wl_m'], c=color, marker=marker,
                   s=90 if not is_new else 60, alpha=alpha,
                   edgecolors='k' if not is_new else None,
                   linewidths=1.2 if not is_new else 0)
ax.set_ylabel('Assigned WL (m)')
ax.set_xlabel('Date')
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
out_png = OUT_DIR / 'rosamarina_densify_timeseries.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')

# Also print area-vs-WL sorted by area, to sanity-check monotonicity directly
print('\nSorted by area_ha (checking area<->WL monotonicity):')
print(df.dropna(subset=['wl_m']).sort_values('area_ha')[['date', 'area_ha', 'wl_m', 'source', 'is_new']].to_string(index=False))
