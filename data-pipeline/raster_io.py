"""Raster I/O helpers: write Cloud-Optimized GeoTIFFs and reproject any raster
onto the canonical grid (the alignment guarantee).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from grid import CanonicalGrid

NODATA = -9999.0


def write_cog(
    path: Path,
    data: np.ndarray,
    grid: CanonicalGrid,
    band_names: Sequence[str] | None = None,
    nodata: float = NODATA,
    dtype: str = "float32",
) -> Path:
    """Write a (bands, H, W) or (H, W) array as a COG on the canonical grid."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    count, h, w = data.shape
    if (h, w) != grid.shape:
        raise ValueError(f"data shape {(h, w)} != grid shape {grid.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "COG",
        "dtype": dtype,
        "count": count,
        "height": h,
        "width": w,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "blocksize": 512,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
    return path


def read_onto_grid(
    src_path: Path,
    grid: CanonicalGrid,
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
    src_nodata: float | None = None,
) -> np.ndarray:
    """Read a single band of any raster, reprojected/resampled onto the grid.

    This is the function that makes alignment robust: no matter what CRS,
    resolution, or extent the source raster has, the result is exactly
    ``grid.shape`` on ``grid.transform``. Out-of-coverage pixels are NaN.
    """
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        src_arr = src.read(band).astype("float32")
        nodata = src_nodata if src_nodata is not None else src.nodata
        if nodata is not None:
            src_arr = np.where(src_arr == nodata, np.nan, src_arr)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def read_band_count(src_path: Path) -> int:
    with rasterio.open(src_path) as src:
        return src.count


def band_descriptions(src_path: Path) -> list[str]:
    with rasterio.open(src_path) as src:
        return [src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
