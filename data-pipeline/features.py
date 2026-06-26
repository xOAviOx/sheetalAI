"""Feature stack builder (Phase 1).

Reads the raw COGs, reprojects each onto the canonical grid (guaranteeing
co-registration), computes the driver/feature stack, and writes:

  data/{city}/stack.tif    multiband COG, one band per feature (named)
  data/{city}/pixels.parquet   tidy per-pixel rows for ML

Feature bands (in order):
  lst_c, ndvi, ndbi, mndwi, albedo, impervious_frac, dist_to_water,
  elevation, slope, pop_density, vulnerability
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt

import indices as ix
from config import city_data_dir, load_city
from grid import build_grid
from raster_io import read_onto_grid, write_cog
from sources.base import WORLDCOVER_BUILTUP, WORLDCOVER_WATER

FEATURE_BANDS = [
    "lst_c",
    "ndvi",
    "ndbi",
    "mndwi",
    "albedo",
    "impervious_frac",
    "dist_to_water",
    "elevation",
    "slope",
    "pop_density",
    "vulnerability",
]
# Drivers used by the ML model (everything except the target lst_c).
DRIVER_BANDS = [b for b in FEATURE_BANDS if b != "lst_c"]


def main() -> None:
    cfg = load_city()
    data_dir = city_data_dir(cfg.key)
    raw_dir = data_dir / "raw"
    grid = build_grid(cfg)
    h, w = grid.shape

    if not (raw_dir / "manifest.json").exists():
        raise SystemExit(f"No raw layers in {raw_dir}. Run gee_export.py first.")

    print(f"[features] city={cfg.key} grid={w}x{h}@{grid.res}m — reprojecting raw layers")

    # --- Read raw layers onto the canonical grid ---
    lst_c = read_onto_grid(raw_dir / "lst.tif", grid, Resampling.bilinear, band=1)
    blue = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=1)
    green = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=2)
    red = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=3)
    nir = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=4)
    swir1 = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=5)
    swir2 = read_onto_grid(raw_dir / "sr.tif", grid, Resampling.bilinear, band=6)
    worldcover = read_onto_grid(raw_dir / "worldcover.tif", grid, Resampling.nearest, band=1)
    pop_count = read_onto_grid(raw_dir / "pop.tif", grid, Resampling.bilinear, band=1)
    dem = read_onto_grid(raw_dir / "dem.tif", grid, Resampling.bilinear, band=1)

    # --- Derived indices ---
    ndvi = ix.ndvi(nir, red)
    ndbi = ix.ndbi(swir1, nir)
    mndwi = ix.mndwi(green, swir1)
    albedo = ix.albedo_liang(blue, red, nir, swir1, swir2)

    builtup_mask = np.isclose(worldcover, WORLDCOVER_BUILTUP)
    impervious = ix.impervious_fraction(ndbi, builtup_mask)

    water_mask = (mndwi > 0.0) | np.isclose(worldcover, WORLDCOVER_WATER)
    if not water_mask.any():
        water_mask = mndwi > np.nanpercentile(mndwi, 99)
    dist_to_water = (distance_transform_edt(~water_mask) * grid.res).astype("float32")

    elevation = dem
    slope = ix.slope_degrees(np.nan_to_num(dem, nan=float(np.nanmedian(dem))), grid.res)

    # Population density (persons/km²). For synthetic, pop_count is per grid cell.
    # For WorldPop (100 m) resampled to 30 m this is an approximation up to a
    # constant factor — fine for relative prioritisation; a sum-conserving
    # reprojection is the documented upgrade path.
    cell_area_km2 = (grid.res / 1000.0) ** 2
    pop_density = (np.clip(pop_count, 0, None) / cell_area_km2).astype("float32")

    vulnerability = (ix.normalize(pop_density) * impervious).astype("float32")

    stack = {
        "lst_c": lst_c,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "mndwi": mndwi,
        "albedo": albedo,
        "impervious_frac": impervious,
        "dist_to_water": dist_to_water,
        "elevation": elevation,
        "slope": slope,
        "pop_density": pop_density,
        "vulnerability": vulnerability,
    }

    # --- Common valid mask (co-registration / no-NaN-misalignment guarantee) ---
    valid = np.ones((h, w), dtype=bool)
    for key in ("lst_c", "ndvi", "albedo", "elevation"):
        valid &= np.isfinite(stack[key])
    coverage = float(valid.mean())
    print(f"[features] valid coverage = {coverage:.1%} of {h * w} pixels")

    # Write the multiband stack; non-valid pixels become NODATA uniformly.
    arr = np.stack([stack[b] for b in FEATURE_BANDS], axis=0).astype("float32")
    arr[:, ~valid] = np.nan
    stack_path = data_dir / "stack.tif"
    write_cog(stack_path, np.where(np.isfinite(arr), arr, -9999.0), grid, FEATURE_BANDS)

    # --- Tidy parquet of valid pixels (for ML) ---
    xs, ys = grid.xy_coords()
    xx, yy = np.meshgrid(xs, ys)
    to_wgs = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(xx, yy)

    rr, cc = np.mgrid[0:h, 0:w]
    df = pd.DataFrame(
        {
            "row": rr[valid].astype("int32"),
            "col": cc[valid].astype("int32"),
            "x": xx[valid].astype("float64"),
            "y": yy[valid].astype("float64"),
            "lon": lon[valid].astype("float64"),
            "lat": lat[valid].astype("float64"),
            **{b: stack[b][valid].astype("float32") for b in FEATURE_BANDS},
        }
    )
    pixels_path = data_dir / "pixels.parquet"
    df.to_parquet(pixels_path, index=False)

    # --- Report ---
    summary = {
        "city": cfg.key,
        "grid": {"epsg": grid.epsg, "res_m": grid.res, "width": w, "height": h},
        "valid_coverage": coverage,
        "n_pixels": int(valid.sum()),
        "bands": FEATURE_BANDS,
        "lst_c_stats": {
            "min": float(np.nanmin(lst_c[valid])),
            "mean": float(np.nanmean(lst_c[valid])),
            "max": float(np.nanmax(lst_c[valid])),
        },
    }
    (data_dir / "features_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[features] stack -> {stack_path}")
    print(f"[features] pixels -> {pixels_path}  ({len(df):,} rows)")
    print(
        f"[features] LST °C  min={summary['lst_c_stats']['min']:.1f} "
        f"mean={summary['lst_c_stats']['mean']:.1f} max={summary['lst_c_stats']['max']:.1f}"
    )


if __name__ == "__main__":
    main()
