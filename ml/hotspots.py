"""Getis-Ord Gi* hotspot analysis — Phase 2.

Finds *statistically significant* clusters of high (hot) and low (cold) land
surface temperature, so the rest of the platform can talk about "hotspots"
rigorously rather than just "pixels above some threshold".

Method
------
We compute the Getis-Ord Gi* statistic with **binary spatial weights** over a
fixed-radius (circular) neighbourhood that includes the focal pixel itself
(the "star" in Gi*). On a regular raster this statistic has an exact closed
form, so we evaluate it by convolution instead of running a permutation engine
over ~700k pixels (which is intractable):

    Gi*_i = (Σ_j w_ij x_j  −  X̄ · Σ_j w_ij)
            ----------------------------------------------
            S · sqrt[ (n · Σ_j w_ij²  −  (Σ_j w_ij)²) / (n − 1) ]

With binary weights Σ_j w_ij = W_i (neighbour count incl. self) and
Σ_j w_ij² = W_i, so every term is either a global scalar or a windowed sum /
count — both obtained with one convolution each. Pixels near the AOI edge (or
near invalid pixels) simply have a smaller W_i; the formula handles that
exactly, so no edge padding fudge is needed.

This is the *same* statistic libpysal/esda compute for a fixed distance band;
``--validate`` cross-checks our analytical z against ``esda.G_Local`` on a
small window and asserts they match to a tight tolerance.

Significance
------------
Two-sided p-values come from the normal approximation, then a
Benjamini-Hochberg FDR correction (the standard guard for the ~700k
simultaneous tests — uncorrected, spatial autocorrelation would flag huge
areas). Pixels are classed into signed confidence bands:

    +3/+2/+1  hot  at 99/95/90%      0  not significant
    -1/-2/-3  cold at 90/95/99%

Outputs (data/{city}/)
----------------------
  hotspots.tif          2-band COG aligned to stack.tif: [gi_z, sig_class]
  hotspots.png          diverging preview for the notebook / dashboard
  hotspots_summary.json params + per-class pixel/area counts + sanity stats

Run:
    CITY=ahmedabad uv run python hotspots.py
    CITY=ahmedabad uv run python hotspots.py --no-validate   # skip esda check
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling  # noqa: F401  (kept for profile parity)
from scipy import stats
from scipy.ndimage import convolve

from config import city_data_dir, load_city

# --- Neighbourhood + significance parameters -------------------------------
DEFAULT_RADIUS_M = 150.0   # circular Gi* neighbourhood radius (~5 px at 30 m)
MIN_NEIGHBOURS = 5         # focal pixels with fewer valid neighbours stay NS
NODATA = -9999.0
# Two-sided FDR thresholds -> signed confidence level (1=90%, 2=95%, 3=99%).
# Ordered weakest-first so the strongest level a pixel qualifies for wins.
Q_LEVELS = ((0.10, 1), (0.05, 2), (0.01, 3))


def disk(radius_px: int) -> np.ndarray:
    """Boolean circular footprint of the given pixel radius (incl. centre)."""
    yy, xx = np.mgrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def gi_star(values: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytical Getis-Ord Gi* z-scores over a binary-weight neighbourhood.

    Returns ``(z, neighbours)`` where both are float arrays the shape of the
    raster; ``z`` is NaN wherever the statistic is undefined (outside the mask
    or too few neighbours).
    """
    k = kernel.astype("float64")
    x = np.where(mask, values, 0.0).astype("float64")
    maskf = mask.astype("float64")

    neighbours = convolve(maskf, k, mode="constant", cval=0.0)   # W_i (incl. self)
    local_sum = convolve(x, k, mode="constant", cval=0.0)        # Σ_j w_ij x_j

    valid = values[mask]
    n = float(valid.size)
    xbar = float(valid.mean())
    s = float(np.sqrt(max((valid * valid).mean() - xbar * xbar, 0.0)))

    z = np.full(values.shape, np.nan, dtype="float64")
    ok = mask & (neighbours >= MIN_NEIGHBOURS) & (neighbours < n)
    w = neighbours[ok]
    var = (n * w - w * w) / (n - 1.0)
    denom = s * np.sqrt(var)
    good = denom > 0
    idx = np.where(ok)
    rows, cols = idx[0][good], idx[1][good]
    z[rows, cols] = (local_sum[ok][good] - xbar * w[good]) / denom[good]
    return z, neighbours


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) for a 1-D array."""
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def classify(z: np.ndarray) -> np.ndarray:
    """Signed FDR-corrected confidence class per pixel (int8)."""
    cls = np.zeros(z.shape, dtype="int8")
    focal = np.isfinite(z)
    if not focal.any():
        return cls
    p = 2.0 * stats.norm.sf(np.abs(z[focal]))     # two-sided
    q = bh_qvalues(p)
    sign = np.sign(z[focal]).astype("int8")
    level = np.zeros(q.shape, dtype="int8")
    for thr, lv in Q_LEVELS:
        level[q <= thr] = lv
    cls[focal] = sign * level
    return cls


def validate_against_esda(values: np.ndarray, mask: np.ndarray, radius_px: int, res_m: float) -> None:
    """Cross-check analytical Gi* against esda.G_Local on a small window."""
    try:
        from esda.getisord import G_Local
        from libpysal.weights import DistanceBand
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"[hotspots] validate: esda/libpysal unavailable ({exc}); skipping")
        return

    # Pick a fully-valid square block away from the AOI edge.
    h, w = values.shape
    b = 40
    r0 = max((h - b) // 2, radius_px)
    c0 = max((w - b) // 2, radius_px)
    block = values[r0 : r0 + b, c0 : c0 + b]
    bmask = mask[r0 : r0 + b, c0 : c0 + b]
    if not bmask.all():
        print("[hotspots] validate: chosen block has gaps; skipping")
        return

    rr, cc = np.mgrid[0:b, 0:b]
    coords = np.column_stack([cc.ravel() * res_m, rr.ravel() * res_m]).astype("float64")
    y = block.ravel().astype("float64")
    thr = radius_px * res_m + 1e-6
    wts = DistanceBand(coords, threshold=thr, binary=True, silence_warnings=True)
    gl = G_Local(y, wts, transform="B", star=True, permutations=0)
    esda_z = gl.Zs.reshape(b, b)

    # Our analytical z on the SAME subset (subset-global stats), compared only on
    # interior pixels whose full disk lies inside the block (neighbour sets match).
    z_ours, nb = gi_star(block, bmask, disk(radius_px))
    full = int(disk(radius_px).sum())
    interior = nb == full
    diff = np.abs(z_ours[interior] - esda_z[interior])
    max_diff = float(np.nanmax(diff)) if diff.size else float("nan")
    print(
        f"[hotspots] validate vs esda.G_Local: {interior.sum()} interior px, "
        f"max|Δz|={max_diff:.2e}"
    )
    assert max_diff < 1e-3, f"analytical Gi* disagrees with esda (max|Δz|={max_diff})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Getis-Ord Gi* hotspot analysis")
    ap.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    args = ap.parse_args()

    cfg = load_city()
    data_dir = city_data_dir(cfg.key)
    stack_path = data_dir / "stack.tif"
    if not stack_path.exists():
        raise SystemExit(f"No stack at {stack_path}. Run the data pipeline first.")

    with rasterio.open(stack_path) as src:
        names = [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)]
        if "lst_c" not in names:
            raise SystemExit(f"stack.tif has no lst_c band (bands={names})")
        lst = src.read(names.index("lst_c") + 1).astype("float64")
        src_nodata = src.nodata
        profile = src.profile
        transform, crs = src.transform, src.crs
        bounds = src.bounds

    res_m = float(abs(transform.a))
    radius_px = max(int(round(args.radius_m / res_m)), 1)
    kernel = disk(radius_px)

    mask = np.isfinite(lst)
    if src_nodata is not None:
        mask &= lst != src_nodata
    n_valid = int(mask.sum())
    print(
        f"[hotspots] city={cfg.key} grid={lst.shape[1]}x{lst.shape[0]}@{res_m:g}m "
        f"radius={args.radius_m:g}m ({radius_px}px, {int(kernel.sum())} px window) "
        f"n_valid={n_valid:,}"
    )

    if args.validate:
        validate_against_esda(lst, mask, radius_px, res_m)

    z, _ = gi_star(lst, mask, kernel)
    sig = classify(z)

    # --- Write aligned 2-band COG: [gi_z, sig_class] ---
    out = np.stack(
        [
            np.where(np.isfinite(z), z, NODATA).astype("float32"),
            sig.astype("float32"),
        ]
    )
    out[:, ~mask] = NODATA
    hot_path = data_dir / "hotspots.tif"
    profile.update(count=2, dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(hot_path, "w", **profile) as dst:
        dst.write(out)
        dst.set_band_description(1, "gi_z")
        dst.set_band_description(2, "sig_class")

    _write_png(z, mask, data_dir / "hotspots.png")

    # --- Summary + sanity stats ---
    z_valid = z[np.isfinite(z)]
    labels = {3: "hot_99", 2: "hot_95", 1: "hot_90", 0: "not_sig",
              -1: "cold_90", -2: "cold_95", -3: "cold_99"}
    px_km2 = (res_m / 1000.0) ** 2
    classes = {}
    for code, label in labels.items():
        count = int(((sig == code) & mask).sum())
        classes[label] = {
            "pixels": count,
            "area_km2": round(count * px_km2, 3),
            "pct": round(100.0 * count / n_valid, 2),
        }

    hot = mask & (sig >= 2)
    cold = mask & (sig <= -2)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = to_wgs.transform(bounds.right, bounds.top)

    summary = {
        "city": cfg.key,
        "method": "Getis-Ord Gi* (binary weights, circular neighbourhood, FDR-corrected)",
        "params": {
            "radius_m": args.radius_m,
            "radius_px": radius_px,
            "window_px": int(kernel.sum()),
            "min_neighbours": MIN_NEIGHBOURS,
            "fdr": "benjamini-hochberg",
            "n_pixels": n_valid,
        },
        "z_stats": {
            "min": float(z_valid.min()),
            "mean": float(z_valid.mean()),
            "max": float(z_valid.max()),
        },
        "classes": classes,
        "sanity": {
            "mean_lst_hot95plus": round(float(lst[hot].mean()), 2) if hot.any() else None,
            "mean_lst_cold95plus": round(float(lst[cold].mean()), 2) if cold.any() else None,
            "mean_lst_overall": round(float(lst[mask].mean()), 2),
        },
        "bounds_wgs84": [lon_min, lat_min, lon_max, lat_max],
    }
    (data_dir / "hotspots_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[hotspots] z min={summary['z_stats']['min']:.1f} "
          f"mean={summary['z_stats']['mean']:.2f} max={summary['z_stats']['max']:.1f}")
    print(f"[hotspots] hot95+ {classes['hot_95']['pct'] + classes['hot_99']['pct']:.1f}% "
          f"cold95+ {classes['cold_95']['pct'] + classes['cold_99']['pct']:.1f}%  "
          f"(LST hot={summary['sanity']['mean_lst_hot95plus']} "
          f"vs cold={summary['sanity']['mean_lst_cold95plus']} "
          f"overall={summary['sanity']['mean_lst_overall']})")
    print(f"[hotspots] wrote {hot_path.name}, hotspots.png, hotspots_summary.json")


def _write_png(z: np.ndarray, mask: np.ndarray, path) -> None:
    """Diverging Gi* z preview (transparent outside the AOI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8 * z.shape[0] / z.shape[1]), dpi=120)
    disp = np.where(np.isfinite(z), z, np.nan)
    im = ax.imshow(np.ma.masked_invalid(disp), cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Gi* z-score")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
