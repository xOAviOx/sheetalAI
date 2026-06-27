"""SHAP global + per-zone explainability — Phase 3.

Turns the trained driver model into *attribution*: for every pixel, how much did
each driver push land-surface temperature above or below the city baseline? This
is the "why is this place hot" layer the dashboard narrates and the equity stage
leans on.

Method
------
TreeExplainer gives exact, additive SHAP values for the XGBoost booster: for each
pixel, ``base_value + Σ_d shap[d] == model prediction`` (in °C). So a driver's
SHAP value is directly readable as "this driver added/removed N °C here".

We produce three views:
  * **global**   mean|SHAP| per driver -> overall importance ranking.
  * **per-pixel** dominant *warming* driver (max positive SHAP) -> raster + PNG.
  * **per-zone**  mean signed SHAP per driver aggregated to the city's zone grid
                  (``zone_grid_size_m``, 750 m = 25 px), with the dominant warming
                  driver per zone -> GeoJSON in lon/lat for the MapLibre dashboard.

Note (carried from train.py): vulnerability/pop_density are exposure layers,
collinear with built-up land, so their SHAP reads as association not actionable
cause. The dashboard should frame biophysical drivers as the leverable ones.

Outputs (data/{city}/)
----------------------
  shap_global.json     base value + mean|SHAP| importance (sorted)
  shap_dominant.tif    1-band COG aligned to stack.tif: dominant-warming driver id
  shap_dominant.png    categorical preview
  shap_zones.geojson   per-zone polygons (EPSG:4326) + signed SHAP per driver
  shap_summary.json    params, driver id map, per-zone dominant-driver counts

Run:
    CITY=ahmedabad uv run python explain.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
import shap
import xgboost as xgb

from config import MODELS_DIR, city_data_dir, load_city

NODATA = -9999.0
INT_NODATA = 255


def _load_meta() -> dict:
    return json.loads((MODELS_DIR / "driver_meta.json").read_text())


def _write_dominant_png(ddir, dom_grid: np.ndarray, drivers: list[str]) -> None:
    """Categorical preview of the dominant-warming-driver raster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n = len(drivers)
    disp = np.where(dom_grid == INT_NODATA, np.nan, dom_grid).astype("float64")
    cmap = plt.get_cmap("tab10", n)
    colors = [cmap(i) for i in range(n)]
    listed = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), n)

    fig, ax = plt.subplots(figsize=(8, 8 * dom_grid.shape[0] / dom_grid.shape[1]), dpi=120)
    ax.imshow(np.ma.masked_invalid(disp), cmap=listed, norm=norm)
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n)]
    ax.legend(handles, drivers, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.savefig(ddir / "shap_dominant.png", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def _write_zones_geojson(ddir, cfg, zmean, drivers: list[str], zone_px: int, profile) -> None:
    """Per-zone polygons in EPSG:4326 with signed mean SHAP per driver."""
    from pyproj import Transformer
    from rasterio.transform import xy

    transform = profile["transform"]
    to_wgs84 = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True)

    features = []
    for zid, row in zmean.iterrows():
        zr = int(row["zrow"])
        zc = int(row["zcol"])
        # zone pixel-space corners -> map coords (xy gives cell centre, so offset
        # by -0.5 px to hit the cell edge) -> lon/lat ring.
        r0, c0 = zr * zone_px, zc * zone_px
        r1, c1 = r0 + zone_px, c0 + zone_px
        corners_px = [(r0, c0), (r0, c1), (r1, c1), (r1, c0), (r0, c0)]
        ring = []
        for rpx, cpx in corners_px:
            mx, my = xy(transform, rpx - 0.5, cpx - 0.5)
            lon, lat = to_wgs84.transform(mx, my)
            ring.append([round(lon, 6), round(lat, 6)])
        props = {
            "zone_id": int(zid),
            "n_pixels": int(row["n_pixels"]),
            "lst_c": round(float(row["lst_c"]), 2),
            "dominant_driver": row["dominant_driver"],
        }
        for d in drivers:
            props[f"shap_{d}"] = round(float(row[f"shap_{d}"]), 3)
        features.append(
            {"type": "Feature", "properties": props,
             "geometry": {"type": "Polygon", "coordinates": [ring]}}
        )

    fc = {"type": "FeatureCollection", "name": f"{cfg.key}_shap_zones",
          "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
          "features": features}
    (ddir / "shap_zones.geojson").write_text(json.dumps(fc))


def main() -> None:
    cfg = load_city()
    ddir = city_data_dir()
    meta = _load_meta()
    drivers: list[str] = meta["features"]
    print(f"[explain] city={cfg.key} drivers={len(drivers)}")

    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "driver_xgb.json"))

    df = pd.read_parquet(ddir / "pixels.parquet")
    X = df[drivers].to_numpy("float64")
    print(f"[explain] pixels={len(df):,} -> TreeExplainer")

    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)  # (n_pixels, n_drivers), additive in °C
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    # --- additivity check: base + Σshap == model prediction ---------------
    recon = base_value + sv.sum(axis=1)
    pred = booster.predict(xgb.DMatrix(X, feature_names=drivers))
    max_dev = float(np.max(np.abs(recon - pred)))
    print(f"[explain] SHAP additivity max|Δ| vs prediction = {max_dev:.2e}")
    assert max_dev < 1e-2, f"SHAP not additive (max dev {max_dev})"

    # --- global importance ------------------------------------------------
    mean_abs = np.abs(sv).mean(axis=0)
    global_imp = {d: float(v) for d, v in zip(drivers, mean_abs)}
    global_sorted = dict(sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True))
    print("[explain] global mean|SHAP| (°C):")
    for d, v in global_sorted.items():
        print(f"    {d:18s} {v:.3f}")
    (ddir / "shap_global.json").write_text(
        json.dumps(
            {"city": cfg.key, "base_value_c": base_value, "mean_abs_shap_c": global_sorted},
            indent=2,
        )
    )

    # --- per-pixel dominant WARMING driver (max positive SHAP) ------------
    # Pixels with no positive driver (net cooled) are marked INT_NODATA.
    pos = np.where(sv > 0, sv, -np.inf)
    dom = pos.argmax(axis=1).astype(np.int64)
    has_pos = np.isfinite(pos.max(axis=1))
    dom_codes = np.where(has_pos, dom, INT_NODATA)

    with rasterio.open(ddir / "stack.tif") as src:
        profile = src.profile.copy()
    h, w = profile["height"], profile["width"]
    rr, cc = df["row"].to_numpy(), df["col"].to_numpy()

    dom_grid = np.full((h, w), INT_NODATA, dtype="uint8")
    dom_grid[rr, cc] = dom_codes.astype("uint8")
    prof_i = profile.copy()
    prof_i.update(count=1, dtype="uint8", nodata=INT_NODATA, compress="deflate")
    with rasterio.open(ddir / "shap_dominant.tif", "w", **prof_i) as dst:
        dst.write(dom_grid, 1)
        dst.set_band_description(1, "dominant_warming_driver_id")

    _write_dominant_png(ddir, dom_grid, drivers)

    # --- per-zone aggregation (750 m grid = 25 px @ 30 m) -----------------
    zone_px = max(1, round(cfg.zone_grid_size_m / cfg.grid_size_m))
    zrow = (rr // zone_px).astype(np.int64)
    zcol = (cc // zone_px).astype(np.int64)
    nzc = int(zcol.max()) + 1
    zid = zrow * nzc + zcol

    sv_df = pd.DataFrame(sv, columns=[f"shap_{d}" for d in drivers])
    sv_df["zid"] = zid
    sv_df["zrow"] = zrow
    sv_df["zcol"] = zcol
    sv_df["lst_c"] = df["lst_c"].to_numpy()
    grouped = sv_df.groupby("zid")
    zmean = grouped.mean(numeric_only=True)
    zmean["n_pixels"] = grouped.size()

    shap_cols = [f"shap_{d}" for d in drivers]
    # dominant warming driver per zone = max positive mean SHAP
    zsv = zmean[shap_cols].to_numpy()
    zpos = np.where(zsv > 0, zsv, -np.inf)
    zdom_idx = zpos.argmax(axis=1)
    zhas_pos = np.isfinite(zpos.max(axis=1))
    zmean["dominant_driver"] = [
        drivers[i] if ok else "none" for i, ok in zip(zdom_idx, zhas_pos)
    ]

    _write_zones_geojson(ddir, cfg, zmean, drivers, zone_px, profile)

    dom_counts = zmean["dominant_driver"].value_counts().to_dict()
    summary = {
        "city": cfg.key,
        "base_value_c": base_value,
        "shap_additivity_max_dev": max_dev,
        "driver_id_map": {i: d for i, d in enumerate(drivers)},
        "zone_grid_size_m": cfg.zone_grid_size_m,
        "zone_px": zone_px,
        "n_zones": int(len(zmean)),
        "global_mean_abs_shap_c": global_sorted,
        "zone_dominant_driver_counts": {k: int(v) for k, v in dom_counts.items()},
    }
    (ddir / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[explain] wrote shap_global.json, shap_dominant.tif/png, shap_zones.geojson, shap_summary.json -> {ddir}")


if __name__ == "__main__":
    main()
