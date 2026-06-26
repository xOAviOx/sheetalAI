"""Configuration loading for the SheetalAI data pipeline.

Single source of truth for city AOIs and run parameters. Everything reads from
``config/cities.yaml`` so that adding a city is a config change, not code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root = parent of the data-pipeline directory.
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CITIES_YAML = PKG_DIR / "config" / "cities.yaml"

# Load .env from the repo root if present (non-fatal if missing).
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class CityConfig:
    """Resolved configuration for a single city."""

    key: str
    display_name: str
    country: str
    bbox: tuple[float, float, float, float]  # minLon, minLat, maxLon, maxLat
    utm_epsg: int
    grid_size_m: int
    zone_aggregation: str
    zone_grid_size_m: int
    date_range: tuple[str, str]
    cloud_cover_max: float
    ward_boundary: str | None = field(default=None)

    @property
    def bbox_list(self) -> list[float]:
        return list(self.bbox)


def _data_root() -> Path:
    root = os.environ.get("DATA_ROOT", str(REPO_ROOT / "data"))
    return Path(root).expanduser().resolve()


def active_city_key(override: str | None = None) -> str:
    """Resolve the active city key from arg > $CITY > 'ahmedabad'."""
    return (override or os.environ.get("CITY") or "ahmedabad").strip().lower()


def load_city(city_key: str | None = None) -> CityConfig:
    """Load a :class:`CityConfig` from ``cities.yaml``."""
    key = active_city_key(city_key)
    with CITIES_YAML.open("r", encoding="utf-8") as fh:
        cities = yaml.safe_load(fh) or {}
    if key not in cities:
        available = ", ".join(sorted(cities)) or "(none)"
        raise KeyError(f"City '{key}' not found in cities.yaml. Available: {available}")
    c = cities[key]
    bbox = tuple(float(x) for x in c["bbox"])
    if len(bbox) != 4:
        raise ValueError(f"City '{key}' bbox must have 4 values, got {len(bbox)}")
    dr = tuple(str(x) for x in c["date_range"])
    return CityConfig(
        key=key,
        display_name=c.get("display_name", key.title()),
        country=c.get("country", ""),
        bbox=bbox,  # type: ignore[arg-type]
        utm_epsg=int(c["utm_epsg"]),
        grid_size_m=int(c.get("grid_size_m", 30)),
        zone_aggregation=str(c.get("zone_aggregation", "grid")),
        zone_grid_size_m=int(c.get("zone_grid_size_m", 750)),
        date_range=dr,  # type: ignore[arg-type]
        cloud_cover_max=float(c.get("cloud_cover_max", 20)),
        ward_boundary=c.get("ward_boundary"),
    )


def list_cities() -> list[str]:
    """Return all configured city keys."""
    with CITIES_YAML.open("r", encoding="utf-8") as fh:
        cities = yaml.safe_load(fh) or {}
    return sorted(cities)


def city_data_dir(city_key: str | None = None) -> Path:
    """Return (and create) the data directory for a city: ``data/{city}/``."""
    key = active_city_key(city_key)
    d = _data_root() / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    return d


if __name__ == "__main__":
    for ck in list_cities():
        cfg = load_city(ck)
        print(f"{cfg.key}: {cfg.display_name} ({cfg.country}) bbox={cfg.bbox} epsg={cfg.utm_epsg}")
    print(f"data root: {_data_root()}")
