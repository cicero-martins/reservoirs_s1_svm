"""
_dem_recon.py — shared waterline-stacking DEM reconstruction (bathtub geometry).

Used by BOTH the SAR pipeline (schwatke_bathymetry_3d.py) and the optical one
(planet_bathymetry.py) so they stay identical. Given a set of binary water masks
and their water levels, reconstruct the exposed-basin DEM shaped as a physical
reservoir 'bathtub': the max-extent boundary is the highest shoreline (max WL) and
the always-wet core the lowest.

Improvements over a naive level-slice:
  1. FOOTPRINT from persistence (wet in >= persist_frac of masks) → drops transient /
     spurious water that inflates the extent, but keeps disconnected in-reservoir
     pools (e.g. an upstream + a near-dam pool at low water) — fixes near-dam
     over-estimation. Largest connected body only → rejects external water bodies.
  2. Per-mask despeckle (remove blobs < min_blob_m2) → removes classification needles.
  3. The footprint boundary is forced to the max WL, so the rim is the highest contour
     and interior gaps interpolate down into the basin (bathtub, not mountain range).
  4. MASKED (normalised) Gaussian smoothing → the edge is not dragged toward the
     outside fill (which produced the spurious border dip).
"""
import numpy as np
from scipy.ndimage import (binary_fill_holes, binary_closing, binary_erosion,
                           gaussian_filter, median_filter, distance_transform_edt, label)
from scipy.interpolate import griddata


def _keep_major(binary, frac=0.05):
    """Keep every connected component >= frac of the largest — retains disconnected
    in-reservoir pools (both Ancipa basins) while dropping small spurious blobs."""
    lab, n = label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return np.isin(lab, np.where(sizes >= sizes.max() * frac)[0])


def _remove_small(binary, min_px):
    lab, n = label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return np.isin(lab, np.where(sizes >= min_px)[0])


def build_dem(masks_raw, wls, pixel_m, min_blob_m2=250.0, persist_frac=0.15,
              smooth_m=18.0, median_px=3):
    """masks_raw: list of 0/1 arrays; wls: matching water levels; pixel_m: grid size (m).
    Returns a float32 DEM (NaN outside the reservoir footprint)."""
    order = np.argsort(wls)
    wl = np.array(wls, float)[order]
    masks = [binary_fill_holes(np.asarray(masks_raw[i]) == 1) for i in order]
    min_px = max(4, int(min_blob_m2 / (pixel_m ** 2)))

    # footprint from persistence, keeping ALL major pools (not just the largest)
    persist = np.mean(np.stack(masks).astype(np.float32), 0)
    footprint = _keep_major(binary_fill_holes(binary_closing(persist >= persist_frac, iterations=3)))
    cln = [binary_fill_holes(_remove_small(m & footprint, min_px)) for m in masks]

    # Bed elevation via a per-pixel OPTIMAL WATER-LEVEL THRESHOLD. A pixel is wet whenever
    # WL > bed, so ideally it is dry in every mask below its bed and wet in every mask
    # above. Real masks are non-nested (SAR mis-classifies the narrow deep channel as land
    # at low water, then sees it again higher up), so we pick, per pixel, the split level k
    # that minimises the disagreement — wet-below-bed plus dry-above-bed — instead of the
    # naive last dry→wet transition, which overwrote deep pixels with shallow elevations
    # (the bug that sank Ancipa's near-dam pool in Period A). Robust to flicker in BOTH
    # directions; reduces to the plain level-slice when the masks are nested.
    W = np.stack(cln).astype(np.int32)                    # (n, H, W), WL-sorted ascending
    n = len(cln)
    wet_cum = np.concatenate([np.zeros((1,) + W.shape[1:], np.int32),
                              np.cumsum(W, axis=0)], 0)    # wet_cum[k] = wet count in levels < k
    kk = np.arange(n + 1, dtype=np.int32).reshape(-1, 1, 1)
    kbest = np.argmin(2 * wet_cum - kk, axis=0)            # dry levels below the bed, per pixel
    ever = W.any(0)
    bed_of_k = np.empty(n + 1, float)                      # k dry levels below -> bed elevation
    bed_of_k[0] = wl[0]                                    # wet from the bottom -> deepest
    bed_of_k[1:n] = (wl[:-1] + wl[1:]) / 2.0              # bed between levels k-1 and k
    bed_of_k[n] = wl[-1]                                   # dry throughout -> shallowest
    dem = np.full(footprint.shape, np.nan, np.float32)
    dem[ever] = bed_of_k[kbest][ever].astype(np.float32)

    # bathtub rim: the reservoir max-extent boundary is the highest shoreline
    dem[footprint & ~binary_erosion(footprint)] = wl[-1]

    known = footprint & np.isfinite(dem)
    todo = footprint & ~np.isfinite(dem)
    if todo.any() and known.sum() > 10:
        yk, xk = np.where(known); zk = dem[known]; yt, xt = np.where(todo)
        dem[yt, xt] = griddata(np.column_stack([xk, yk]), zk,
                               np.column_stack([xt, yt]), method='linear')
        still = footprint & ~np.isfinite(dem)
        if still.any():
            ys, xs = np.where(still)
            dem[ys, xs] = griddata(np.column_stack([xk, yk]), zk,
                                   np.column_stack([xs, ys]), method='nearest')

    # De-spike (median) + smooth (Gaussian), extending nearest footprint values outward
    # first so neither filter sees an artificial low edge (avoids the border dip).
    inside = footprint & np.isfinite(dem)
    _, idx = distance_transform_edt(~inside, return_distances=True, return_indices=True)
    ext = dem[tuple(idx)]                                  # nearest footprint value everywhere
    if median_px and median_px >= 2:
        ext = median_filter(ext, size=int(median_px))     # kill isolated 'mountain-range' spikes
    ext = gaussian_filter(ext, max(1.0, smooth_m / pixel_m))
    return np.where(footprint, ext, np.nan).astype(np.float32)
