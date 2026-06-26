"""Pure functions for spectral indices and derived drivers.

Kept dependency-light (numpy only) and side-effect free so they are trivially
unit-testable and reusable by both the GEE and synthetic paths.

Band convention (surface reflectance, 0-1): blue, green, red, nir, swir1, swir2.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / (den + np.sign(den) * EPS + EPS * (den == 0))


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index, (NIR-Red)/(NIR+Red)."""
    return np.clip(_safe_ratio(nir - red, nir + red), -1.0, 1.0)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index, (SWIR1-NIR)/(SWIR1+NIR)."""
    return np.clip(_safe_ratio(swir1 - nir, swir1 + nir), -1.0, 1.0)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified Normalised Difference Water Index, (Green-SWIR1)/(Green+SWIR1)."""
    return np.clip(_safe_ratio(green - swir1, green + swir1), -1.0, 1.0)


def albedo_liang(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    """Shortwave broadband albedo via Liang (2001) Landsat coefficients.

    α = (0.356·blue + 0.130·red + 0.373·nir + 0.085·swir1 + 0.072·swir2 − 0.0018) / 1.016
    """
    a = (
        0.356 * blue
        + 0.130 * red
        + 0.373 * nir
        + 0.085 * swir1
        + 0.072 * swir2
        - 0.0018
    ) / 1.016
    return np.clip(a, 0.0, 1.0)


def impervious_fraction(
    ndbi_arr: np.ndarray,
    builtup_mask: np.ndarray,
    ndbi_threshold: float = 0.0,
) -> np.ndarray:
    """Impervious-surface fraction proxy from WorldCover built-up + NDBI.

    Built-up pixels start at 0.6; NDBI above ``ndbi_threshold`` adds up to 0.4.
    Non-built pixels are driven purely by (positive) NDBI, capped at 0.5.
    """
    ndbi_pos = np.clip((ndbi_arr - ndbi_threshold) / (1.0 - ndbi_threshold + EPS), 0.0, 1.0)
    built = builtup_mask.astype("float32")
    frac = built * (0.6 + 0.4 * ndbi_pos) + (1.0 - built) * (0.5 * ndbi_pos)
    return np.clip(frac, 0.0, 1.0)


def slope_degrees(elevation: np.ndarray, res_m: float) -> np.ndarray:
    """Slope in degrees from a DEM via horizontal gradient."""
    dy, dx = np.gradient(elevation, res_m, res_m)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def normalize(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Min-max normalise to [0, 1], ignoring NaNs."""
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo = float(np.nanmin(arr)) if lo is None else lo
    hi = float(np.nanmax(arr)) if hi is None else hi
    if hi - lo < EPS:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
