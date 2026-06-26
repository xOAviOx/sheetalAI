"""Canonical analysis grid for a city.

Every raw raster (whatever the source) is reprojected onto *this* grid in
``features.py``, which is what guarantees pixel-perfect alignment across layers
("no NaN misalignment"). The grid is fully determined by the city config:
bbox (EPSG:4326) reprojected to the city UTM, snapped to ``grid_size_m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Transformer

from config import CityConfig


@dataclass(frozen=True)
class CanonicalGrid:
    """Immutable description of the analysis raster grid (in city UTM)."""

    epsg: int
    res: float  # metres per pixel
    width: int
    height: int
    xmin: float  # left edge (easting)
    ymax: float  # top edge (northing)

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def transform(self) -> Affine:
        return Affine.translation(self.xmin, self.ymax) * Affine.scale(self.res, -self.res)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in UTM metres."""
        xmax = self.xmin + self.width * self.res
        ymin = self.ymax - self.height * self.res
        return (self.xmin, ymin, xmax, self.ymax)

    def xy_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D arrays of pixel-centre eastings (x) and northings (y)."""
        xs = self.xmin + (np.arange(self.width) + 0.5) * self.res
        ys = self.ymax - (np.arange(self.height) + 0.5) * self.res
        return xs, ys


def build_grid(cfg: CityConfig) -> CanonicalGrid:
    """Derive the canonical grid from a :class:`CityConfig`."""
    lon_min, lat_min, lon_max, lat_max = cfg.bbox
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg.utm_epsg}", always_xy=True)

    # Densify the bbox boundary so reprojection captures curvature, not just corners.
    n = 25
    lons = np.concatenate(
        [
            np.linspace(lon_min, lon_max, n),
            np.full(n, lon_max),
            np.linspace(lon_max, lon_min, n),
            np.full(n, lon_min),
        ]
    )
    lats = np.concatenate(
        [
            np.full(n, lat_min),
            np.linspace(lat_min, lat_max, n),
            np.full(n, lat_max),
            np.linspace(lat_max, lat_min, n),
        ]
    )
    xs, ys = tf.transform(lons, lats)
    res = float(cfg.grid_size_m)

    # Snap outward to whole pixels so the grid fully covers the AOI.
    xmin = np.floor(min(xs) / res) * res
    xmax = np.ceil(max(xs) / res) * res
    ymin = np.floor(min(ys) / res) * res
    ymax = np.ceil(max(ys) / res) * res

    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    return CanonicalGrid(
        epsg=cfg.utm_epsg, res=res, width=width, height=height, xmin=xmin, ymax=ymax
    )
