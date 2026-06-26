"""Configuration loading for the SheetalAI API.

Reads the shared ``data-pipeline/config/cities.yaml`` and exposes API runtime
settings (CORS, data root, advisory flag) sourced from environment / .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CITIES_YAML = REPO_ROOT / "data-pipeline" / "config" / "cities.yaml"

load_dotenv(REPO_ROOT / ".env")


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data"))).expanduser().resolve()


def city_data_dir(city_key: str) -> Path:
    return data_root() / city_key.strip().lower()


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def advisory_enabled() -> bool:
    return os.environ.get("ENABLE_ADVISORY", "false").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class CityMeta:
    key: str
    display_name: str
    country: str
    bbox: list[float]
    utm_epsg: int


def load_cities() -> dict[str, CityMeta]:
    with CITIES_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, CityMeta] = {}
    for key, c in raw.items():
        out[key] = CityMeta(
            key=key,
            display_name=c.get("display_name", key.title()),
            country=c.get("country", ""),
            bbox=[float(x) for x in c["bbox"]],
            utm_epsg=int(c["utm_epsg"]),
        )
    return out
