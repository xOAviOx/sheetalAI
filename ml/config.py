"""Configuration loading for the SheetalAI ML stage.

Reads the same single source of truth used across the monorepo:
``data-pipeline/config/cities.yaml``. Kept as a small standalone module so the
ml/ uv environment stays independent of data-pipeline/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CITIES_YAML = REPO_ROOT / "data-pipeline" / "config" / "cities.yaml"
MODELS_DIR = PKG_DIR / "models"

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class CityConfig:
    key: str
    display_name: str
    country: str
    bbox: tuple[float, float, float, float]
    utm_epsg: int
    grid_size_m: int
    zone_aggregation: str
    zone_grid_size_m: int
    date_range: tuple[str, str]
    cloud_cover_max: float
    ward_boundary: str | None = field(default=None)


def _data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data"))).expanduser().resolve()


def active_city_key(override: str | None = None) -> str:
    return (override or os.environ.get("CITY") or "ahmedabad").strip().lower()


def load_city(city_key: str | None = None) -> CityConfig:
    key = active_city_key(city_key)
    with CITIES_YAML.open("r", encoding="utf-8") as fh:
        cities = yaml.safe_load(fh) or {}
    if key not in cities:
        available = ", ".join(sorted(cities)) or "(none)"
        raise KeyError(f"City '{key}' not found. Available: {available}")
    c = cities[key]
    return CityConfig(
        key=key,
        display_name=c.get("display_name", key.title()),
        country=c.get("country", ""),
        bbox=tuple(float(x) for x in c["bbox"]),  # type: ignore[arg-type]
        utm_epsg=int(c["utm_epsg"]),
        grid_size_m=int(c.get("grid_size_m", 30)),
        zone_aggregation=str(c.get("zone_aggregation", "grid")),
        zone_grid_size_m=int(c.get("zone_grid_size_m", 750)),
        date_range=tuple(str(x) for x in c["date_range"]),  # type: ignore[arg-type]
        cloud_cover_max=float(c.get("cloud_cover_max", 20)),
        ward_boundary=c.get("ward_boundary"),
    )


def list_cities() -> list[str]:
    with CITIES_YAML.open("r", encoding="utf-8") as fh:
        cities = yaml.safe_load(fh) or {}
    return sorted(cities)


def city_data_dir(city_key: str | None = None) -> Path:
    key = active_city_key(city_key)
    d = _data_root() / key
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    for ck in list_cities():
        cfg = load_city(ck)
        print(f"{cfg.key}: {cfg.display_name} epsg={cfg.utm_epsg} zones={cfg.zone_aggregation}")
