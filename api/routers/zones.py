"""Zone endpoints — Phase 6.

Serves the priority zone GeoJSON produced by ml/prioritize.py and
ml/explain.py. The full FeatureCollection is small enough (~800 KB) to
return in a single response; the frontend's GeoJsonLayer streams it once and
caches locally.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as FPath
from fastapi.responses import JSONResponse, Response

from config import city_data_dir, load_cities

router = APIRouter(prefix="/cities/{city}", tags=["zones"])


# ── helpers ──────────────────────────────────────────────────────────────────

def _validate_city(city: str) -> str:
    cities = load_cities()
    key = city.strip().lower()
    if key not in cities:
        raise HTTPException(404, f"City '{city}' not found. Available: {sorted(cities)}")
    return key


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise HTTPException(
            503,
            f"{label} not found at {path}. Run the ML pipeline first.",
        )
    return path


@lru_cache(maxsize=8)
def _load_priority_fc(city: str, _mtime: float) -> dict:
    """Load and cache priority_zones.geojson (cache-busted by file mtime)."""
    p = city_data_dir(city) / "priority_zones.geojson"
    return json.loads(p.read_text())


def _priority_fc(city: str) -> dict:
    p = _require_file(city_data_dir(city) / "priority_zones.geojson", "priority_zones.geojson")
    mtime = p.stat().st_mtime
    return _load_priority_fc(city, mtime)


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/zones", summary="Priority zones — full FeatureCollection")
def get_zones(city: Annotated[str, FPath(description="City key, e.g. ahmedabad")]) -> Response:
    """Return all 750-m priority zones as a GeoJSON FeatureCollection.

    Each feature carries: equity_score, equity_rank, heat/pop/vuln_score,
    lst_c, pop_density, vulnerability, best_intervention, best_delta_lst_c,
    dominant_driver, and per-driver SHAP values from Phase 3.

    The file is served as raw bytes to avoid a Python JSON round-trip that
    would reject any NaN values that escaped serialisation in the ML pipeline.
    """
    key = _validate_city(city)
    p = _require_file(city_data_dir(key) / "priority_zones.geojson", "priority_zones.geojson")
    return Response(
        content=p.read_bytes(),
        media_type="application/geo+json",
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get("/zones/{zone_id}", summary="Single zone detail")
def get_zone(
    city: Annotated[str, FPath(description="City key")],
    zone_id: Annotated[int, FPath(description="Zone ID (0-based)")],
) -> dict:
    """Return properties + geometry for a single zone (for click-to-inspect panel)."""
    key = _validate_city(city)
    fc = _priority_fc(key)
    for feat in fc["features"]:
        if feat["properties"]["zone_id"] == zone_id:
            return feat
    raise HTTPException(404, f"Zone {zone_id} not found in city '{key}'.")


@router.get("/zones/rank/{rank}", summary="Zone by equity rank")
def get_zone_by_rank(
    city: Annotated[str, FPath(description="City key")],
    rank: Annotated[int, FPath(description="Equity rank (1 = highest priority)", ge=1)],
) -> dict:
    """Return the zone at a given equity rank (1 = most urgent)."""
    key = _validate_city(city)
    fc = _priority_fc(key)
    for feat in fc["features"]:
        if feat["properties"]["equity_rank"] == rank:
            return feat
    raise HTTPException(404, f"No zone with rank {rank} in city '{key}'.")
