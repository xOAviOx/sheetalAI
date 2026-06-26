"""Synthetic data source — offline, deterministic, physically-plausible (Phase 1).

Lets the *entire* downstream pipeline (Phases 2-7) run and be demonstrated
without Google Earth Engine credentials, and makes the Phase 1 acceptance test
(aligned stack, no NaN, LST renders, hot zones in the right place) verifiable
offline. Outputs are clearly labelled SYNTHETIC in provenance.

Construction is internally consistent: per-pixel fractional cover of four
endmembers (vegetation / built-up / water / bare soil) drives BOTH the surface
reflectance (via linear spectral mixing) AND the land-surface temperature, so
indices and LST share the same latent structure — exactly what a real city has,
and what makes the driver model learn a meaningful relationship.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from config import CityConfig
from grid import build_grid
from raster_io import write_cog
from sources.base import (
    SR_BANDS,
    WORLDCOVER_BUILTUP,
    WORLDCOVER_TREES,
    WORLDCOVER_WATER,
    DataSource,
    SourceResult,
)

# Endmember surface-reflectance spectra (blue,green,red,nir,swir1,swir2), 0-1.
_ENDMEMBERS = {
    "veg": np.array([0.03, 0.06, 0.04, 0.35, 0.18, 0.10], dtype="float32"),
    "built": np.array([0.12, 0.14, 0.16, 0.20, 0.26, 0.24], dtype="float32"),
    "water": np.array([0.05, 0.06, 0.04, 0.02, 0.01, 0.01], dtype="float32"),
    "soil": np.array([0.10, 0.13, 0.18, 0.25, 0.30, 0.25], dtype="float32"),
}
WORLDCOVER_GRASS = 30
WORLDCOVER_CROP = 40
WORLDCOVER_BARE = 60


def _seed(city_key: str) -> int:
    return int(hashlib.sha256(city_key.encode()).hexdigest(), 16) % (2**32)


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Low-frequency field in [0,1] from blurred white noise."""
    f = gaussian_filter(rng.standard_normal(shape).astype("float32"), sigma=sigma)
    f -= f.min()
    return f / (f.max() + 1e-9)


class SyntheticSource(DataSource):
    """Deterministic synthetic city. Same output contract as :class:`GEESource`."""

    name = "synthetic"

    def export(self, cfg: CityConfig, raw_dir: Path) -> SourceResult:
        raw_dir.mkdir(parents=True, exist_ok=True)
        grid = build_grid(cfg)
        h, w = grid.shape
        rng = np.random.default_rng(_seed(cfg.key))

        rows, cols = np.mgrid[0:h, 0:w].astype("float32")
        cy, cx = h / 2.0, w / 2.0

        # --- River: sinuous near-vertical channel (a Sabarmati-like spine) ---
        river_x = cx + 0.18 * w * np.sin(2 * np.pi * rows / max(h, 1) * 1.5 + 0.6)
        dist_river_px = np.abs(cols - river_x)
        half_width = max(2.0, 0.012 * w)
        water_mask = (dist_river_px < half_width).astype("float32")
        water_soft = np.clip(1.0 - dist_river_px / (half_width * 2.0), 0.0, 1.0) * (1 - water_mask)
        water_frac0 = np.clip(water_mask + 0.4 * water_soft, 0.0, 1.0)

        # --- Urban core: central blob + secondary nuclei + texture ---
        core = np.exp(-(((cols - cx) / (0.30 * w)) ** 2 + ((rows - cy) / (0.30 * h)) ** 2))
        nuclei = np.zeros((h, w), dtype="float32")
        for _ in range(3):
            ny, nx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
            nuclei += np.exp(-(((cols - nx) / (0.12 * w)) ** 2 + ((rows - ny) / (0.12 * h)) ** 2))
        urban = np.clip(0.7 * core + 0.5 * nuclei + 0.3 * _smooth_field(rng, (h, w), 8), 0, 1)
        urban = urban / (urban.max() + 1e-9)

        veg_field = _smooth_field(rng, (h, w), 10)

        # --- Fractional cover of 4 endmembers (sum to 1) ---
        land = 1.0 - water_frac0
        built_frac = land * np.clip(urban**1.3, 0, 1)
        veg_frac = land * (1 - built_frac) * np.clip(veg_field * 1.2, 0, 1)
        soil_frac = np.clip(land - built_frac - veg_frac, 0, None)
        water_frac = water_frac0
        total = built_frac + veg_frac + soil_frac + water_frac + 1e-9
        built_frac, veg_frac, soil_frac, water_frac = (
            built_frac / total,
            veg_frac / total,
            soil_frac / total,
            water_frac / total,
        )

        # --- Surface reflectance via linear spectral mixing (+ small noise) ---
        sr = np.zeros((6, h, w), dtype="float32")
        for frac, key in (
            (veg_frac, "veg"),
            (built_frac, "built"),
            (water_frac, "water"),
            (soil_frac, "soil"),
        ):
            sr += frac[np.newaxis] * _ENDMEMBERS[key][:, np.newaxis, np.newaxis]
        sr += rng.normal(0, 0.006, sr.shape).astype("float32")
        sr = np.clip(sr, 0.0, 1.0)

        # --- DEM: flat-ish plain (~50 m), gentle gradient + low-freq relief ---
        dem = (
            45.0
            + 25.0 * _smooth_field(rng, (h, w), 14)
            + 0.01 * (rows - cy)  # faint regional tilt
        ).astype("float32")
        dem -= 6.0 * water_frac0  # river sits slightly lower

        # --- Distance to water (m) for the LST relationship ---
        dist_to_water_m = distance_transform_edt(water_mask < 0.5) * grid.res

        # --- LST (°C): hot where built/dry/far-from-water, cool where veg/water ---
        d_norm = np.clip(dist_to_water_m / (0.4 * max(h, w) * grid.res), 0, 1)
        elev_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-9)
        lst_c = (
            38.0
            + 8.5 * built_frac
            - 6.0 * veg_frac
            - 9.0 * water_frac
            + 2.5 * soil_frac
            + 3.0 * d_norm
            - 2.0 * elev_norm
            + rng.normal(0, 0.5, (h, w)).astype("float32")
        ).astype("float32")

        # --- WorldCover class codes ---
        wc = np.full((h, w), WORLDCOVER_GRASS, dtype="float32")
        wc[soil_frac > 0.45] = WORLDCOVER_BARE
        wc[veg_frac > 0.35] = WORLDCOVER_CROP
        wc[veg_frac > 0.55] = WORLDCOVER_TREES
        wc[built_frac > 0.45] = WORLDCOVER_BUILTUP
        wc[water_mask > 0.5] = WORLDCOVER_WATER

        # --- Population count per pixel: concentrated in built-up areas ---
        pop = (built_frac**1.5 * 180.0 * (0.6 + 0.8 * _smooth_field(rng, (h, w), 6))).astype(
            "float32"
        )

        # --- Write raw COGs on the canonical grid (already aligned) ---
        paths = {
            "lst": write_cog(raw_dir / "lst.tif", lst_c, grid, ["lst_c"]),
            "sr": write_cog(raw_dir / "sr.tif", sr, grid, list(SR_BANDS)),
            "worldcover": write_cog(raw_dir / "worldcover.tif", wc, grid, ["worldcover"]),
            "pop": write_cog(raw_dir / "pop.tif", pop, grid, ["population"]),
            "dem": write_cog(raw_dir / "dem.tif", dem, grid, ["elevation"]),
        }
        return SourceResult(
            source_name=self.name,
            lst=paths["lst"],
            sr=paths["sr"],
            worldcover=paths["worldcover"],
            pop=paths["pop"],
            dem=paths["dem"],
            note="SYNTHETIC deterministic city — not real satellite data",
        )
