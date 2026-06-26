"""Earth Engine data source — real satellite export (Phase 1).

Builds a cloud-masked median composite for the city/date-range and exports:
  - LST (°C) from Landsat 8 C02 L2 ST_B10
  - 6 SR bands (blue,green,red,nir,swir1,swir2), scaled to reflectance
  - ESA WorldCover v200 land cover
  - WorldPop population
  - NASADEM elevation

Each layer is downloaded to ``raw_dir`` as a GeoTIFF in the city UTM CRS at the
configured scale via ``geemap.ee_export_image`` (uses EE's getDownloadURL).
Exact grid alignment is handled later by ``features.read_onto_grid``.

``ee`` is imported lazily and only from here — downstream code never touches
Earth Engine, honouring the swappable-source contract.
"""

from __future__ import annotations

from pathlib import Path

from config import CityConfig
from gee_auth import init_ee
from sources.base import SR_BANDS, DataSource, SourceResult

# Landsat 8 C02 L2 scaling (USGS): SR = DN*0.0000275 - 0.2 ; ST = DN*0.00341802 + 149.0 (K)
_SR_SCALE, _SR_OFFSET = 0.0000275, -0.2
_ST_SCALE, _ST_OFFSET_K = 0.00341802, 149.0
_KELVIN_TO_C = -273.15

_L8_BANDS = {
    "SR_B2": "blue",
    "SR_B3": "green",
    "SR_B4": "red",
    "SR_B5": "nir",
    "SR_B6": "swir1",
    "SR_B7": "swir2",
}


def _mask_l8_sr(img):  # noqa: ANN001 — ee.Image
    """Cloud / cloud-shadow / cirrus mask from QA_PIXEL bitmask (bits 1-4)."""
    qa = img.select("QA_PIXEL")
    dilated = qa.bitwiseAnd(1 << 1).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return img.updateMask(dilated.And(cirrus).And(cloud).And(shadow))


class GEESource(DataSource):
    """Google Earth Engine export. Requires one-time ``earthengine authenticate``."""

    name = "gee"

    def export(self, cfg: CityConfig, raw_dir: Path) -> SourceResult:
        import ee  # noqa: F401 — ensure ee is importable before building expressions
        import geemap

        ee = init_ee()
        raw_dir.mkdir(parents=True, exist_ok=True)

        region = ee.Geometry.Rectangle(cfg.bbox_list, proj="EPSG:4326", geodesic=False)
        crs = f"EPSG:{cfg.utm_epsg}"
        scale = float(cfg.grid_size_m)
        start, end = cfg.date_range

        # ---- Landsat 8 cloud-masked median composite ----
        col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cfg.cloud_cover_max))
            .map(_mask_l8_sr)
        )
        n_scenes = col.size().getInfo()
        if n_scenes == 0:
            raise RuntimeError(
                f"No Landsat 8 scenes for {cfg.key} in {start}..{end} "
                f"(CLOUD_COVER<{cfg.cloud_cover_max}). Widen date_range or cloud_cover_max."
            )
        composite = col.median()

        lst_c = (
            composite.select("ST_B10")
            .multiply(_ST_SCALE)
            .add(_ST_OFFSET_K + _KELVIN_TO_C)
            .rename("lst_c")
        )
        sr = (
            composite.select(list(_L8_BANDS))
            .multiply(_SR_SCALE)
            .add(_SR_OFFSET)
            .rename(list(_L8_BANDS.values()))
        )
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")
        pop = (
            ee.ImageCollection("WorldPop/GP/100m/pop")
            .filterBounds(region)
            .filter(ee.Filter.eq("year", 2020))
            .mosaic()
            .select("population")
        )
        dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

        layers = {"lst": lst_c, "sr": sr, "worldcover": worldcover, "pop": pop, "dem": dem}
        paths: dict[str, Path] = {}
        for key, image in layers.items():
            out = raw_dir / f"{key}.tif"
            # ee_export_image downloads via getDownloadURL. For larger AOIs that hit
            # the request-size limit, switch to geemap.download_ee_image (tiled).
            geemap.ee_export_image(
                image,
                filename=str(out),
                scale=scale,
                crs=crs,
                region=region,
                file_per_band=False,
            )
            if not out.exists():
                raise RuntimeError(
                    f"GEE export of '{key}' produced no file. If this is a size-limit "
                    "error, use geemap.download_ee_image (tiled) for this AOI."
                )
            paths[key] = out

        return SourceResult(
            source_name=self.name,
            lst=paths["lst"],
            sr=paths["sr"],
            worldcover=paths["worldcover"],
            pop=paths["pop"],
            dem=paths["dem"],
            note=f"Landsat8 median {start}..{end}, {n_scenes} scenes, cloud<{cfg.cloud_cover_max}%",
        )


__all__ = ["GEESource", "SR_BANDS"]
