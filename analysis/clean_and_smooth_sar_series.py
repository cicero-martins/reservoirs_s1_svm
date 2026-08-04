"""
clean_and_smooth_sar_series.py (2026-08-04)

Faithful Python/pandas port of gee_reservoir_monitor_app_v226.js's cleanAndSmooth()
chain -- removeOutliers(2 sigma) -> detectAndRemoveLocalOutliers(5, 1.5) x2 ->
detectAndRemoveLocalOutliers(10, 1.5) -> lowessSmoothing(+-20 d, bandwidth 7) --
itself "a faithful port ... from reservoirs_s1_svm.js" per that file's own comment,
i.e. the post-processing of the published Paper-1 pipeline.

Why it is needed: the operational volume-record comparison
(validate_area_volume_timeseries.py, manuscript Table 5) scores the five original
reservoirs on a SMOOTHED area series -- Ancipa/Poma/Pozzillo/Rosamarina via the
'value' column (median scene-to-scene jump roughly half that of the raw 'areaLago'
column beside it) and Garcia via 'areaLago_smoothed'. The four Fase-3 extended
reservoirs, rebuilt by rebuild_sar_series_otsufix.py, are raw. Scoring 5 smoothed
series against 4 raw ones is not a like-for-like comparison: smoothing suppresses
per-scene classification noise and so mechanically lowers RMSE. This module supplies
the missing step for those four.

Scope: the Table-5 path only. The FRS hypsometric fit (build_frs_dem.fit_swot_curve,
Tables 3 and 4) deliberately stays on RAW per-acquisition areas for all nine, since
it pairs individual SAR acquisitions with individual SWOT passes -- smoothing there
would blend neighbouring dates into a pair that was never actually observed.
"""
import numpy as np
import pandas as pd


def remove_outliers(df, threshold, col='area_ha'):
    mean, std = df[col].mean(), df[col].std()
    dev = (df[col] - mean).abs() / std
    return df[dev <= threshold].reset_index(drop=True)


def detect_and_remove_local_outliers(df, window_size, std_dev_threshold, col='area_ha'):
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    half = window_size // 2
    keep = np.ones(n, dtype=bool)
    vals = df[col].values
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half)
        window = vals[lo:hi]
        mean, sd = window.mean(), window.std(ddof=0)
        if sd == 0:
            continue
        dev = abs(vals[i] - mean) / sd
        keep[i] = dev <= std_dev_threshold
    return df[keep].reset_index(drop=True)


def lowess_smoothing(df, window_days, bandwidth, col='area_ha'):
    df = df.sort_values('date').reset_index(drop=True)
    dates = df['date'].values.astype('datetime64[ns]')
    vals = df[col].values
    n = len(df)
    smoothed = np.empty(n)
    for i in range(n):
        t = dates[i]
        w_lo = t - np.timedelta64(window_days, 'D')
        w_hi = t + np.timedelta64(window_days, 'D')
        mask = (dates >= w_lo) & (dates <= w_hi)
        diff_days = np.abs((dates[mask] - t) / np.timedelta64(1, 'D'))
        weight = np.exp(-(diff_days / bandwidth) ** 2)
        smoothed[i] = np.sum(weight * vals[mask]) / np.sum(weight)
    df = df.copy()
    df[f'{col}_smoothed'] = smoothed
    return df


def clean_and_smooth(df, col='area_ha'):
    """Full original chain: 1 global pass + 3 local passes + LOWESS."""
    ts1 = remove_outliers(df, 2, col)
    ts2 = detect_and_remove_local_outliers(ts1, 5, 1.5, col)
    ts3 = detect_and_remove_local_outliers(ts2, 5, 1.5, col)
    ts4 = detect_and_remove_local_outliers(ts3, 10, 1.5, col)
    return lowess_smoothing(ts4, 20, 7, col)


if __name__ == '__main__':
    import pathlib
    SAR_DIR = pathlib.Path('raw_data/exportSicilyExtended/GEE_SicilyExtended_VVotsu')
    name = 'Olivo'
    df = pd.read_csv(SAR_DIR / f'SAR_area_{name}.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f'{name}: {len(df)} raw rows')
    out = clean_and_smooth(df, 'area_ha')
    print(f'{name}: {len(out)} rows survive outlier rejection ({len(df)-len(out)} dropped)')
    out.to_csv(SAR_DIR / f'_cleaned_SAR_area_{name}.csv', index=False)
    print(out[['date', 'area_ha', 'area_ha_smoothed']].tail(40).to_string())
