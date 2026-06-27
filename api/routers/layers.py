"""Raster layer endpoints — Phase 6.

Serves pre-rendered PNG layers for deck.gl BitmapLayer. Each layer endpoint
returns the PNG file directly; a companion metadata endpoint gives the
geographic bounds needed to position the BitmapLayer on the map.

Layer catalogue
---------------
  hotspots       Getis-Ord Gi* significance classes (Phase 2)
  shap_dominant  Per-pixel dominant warming driver (Phase 3)
  simulation     Best-intervention central ΔLST (Phase 4)
  priority       Equity priority score choropleth (Phase 5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as FPath
from fastapi.responses import FileResponse

from config import city_data_dir, load_cities

router = APIRouter(prefix="/cities/{city}/layers", tags=["layers"])

# Map layer name → filename in data/{city}/
_LAYER_FILES: dict[str, dict] = {
    "hotspots": {
        "file": "hotspots.png",
        "label": "Urban heat hotspots (Getis-Ord Gi*)",
        "description": "Hot/cold/not-significant zones at 150 m neighbourhood, BH-corrected FDR.",
        "phase": 2,
    },
    "shap_dominant": {
        "file": "shap_dominant.png",
        "label": "Dominant warming driver (SHAP)",
        "description": "Per-pixel driver with the largest positive SHAP contribution to LST.",
        "phase": 3,
    },
    "simulation": {
        "file": "simulation_best.png",
        "label": "Best-intervention cooling ΔLST (°C)",
        "description": "Central ΔLST for the strongest cooling intervention. Blue = cooling.",
        "phase": 4,
    },
    "priority": {
        "file": "priority_map.png",
        "label": "Equity priority score",
        "description": "0.4·heat + 0.3·pop + 0.3·vuln, percentile-rank normalised. Dark red = highest.",
        "phase": 5,
    },
}


def _validate_city(city: str) -> str:
    cities = load_cities()
    key = city.strip().lower()
    if key not in cities:
        raise HTTPException(404, f"City '{city}' not found. Available: {sorted(cities)}")
    return key


def _city_bbox(city: str) -> list[float]:
    """Return [minLon, minLat, maxLon, maxLat] — deck.gl BitmapLayer bounds order."""
    meta = load_cities()[city]
    return list(meta.bbox)  # already [minLon, minLat, maxLon, maxLat] from cities.yaml


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("", summary="List available layers with metadata")
def list_layers(
    city: Annotated[str, FPath(description="City key, e.g. ahmedabad")],
) -> list[dict]:
    """List available raster layers, their metadata, and geographic bounds.

    The ``bounds`` field matches deck.gl BitmapLayer's expected format:
    ``[minLon, minLat, maxLon, maxLat]``.  Use ``png_url`` as the
    ``image`` prop of BitmapLayer.
    """
    key = _validate_city(city)
    ddir = city_data_dir(key)
    bbox = _city_bbox(key)
    base_url = f"/cities/{key}/layers"
    result = []
    for name, spec in _LAYER_FILES.items():
        path = ddir / spec["file"]
        result.append({
            "name": name,
            "label": spec["label"],
            "description": spec["description"],
            "phase": spec["phase"],
            "available": path.exists(),
            "png_url": f"{base_url}/{name}.png",
            "bounds": bbox,
        })
    return result


@router.get("/{name}.png", summary="Serve PNG raster layer", response_class=FileResponse)
def get_layer_png(
    city: Annotated[str, FPath(description="City key")],
    name: Annotated[str, FPath(description="Layer name (hotspots | shap_dominant | simulation | priority)")],
) -> FileResponse:
    """Serve a pre-rendered PNG for use as a deck.gl BitmapLayer image.

    The image is georeferenced to the city bbox (see ``GET /cities/{city}/layers``).
    """
    key = _validate_city(city)
    if name not in _LAYER_FILES:
        raise HTTPException(
            404,
            f"Unknown layer '{name}'. Available: {list(_LAYER_FILES)}",
        )
    spec = _LAYER_FILES[name]
    path = city_data_dir(key) / spec["file"]
    if not path.exists():
        raise HTTPException(
            503,
            f"Layer '{name}' ({spec['file']}) not found. Run the ML pipeline through Phase {spec['phase']}.",
        )
    return FileResponse(
        path=str(path),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{name}/bounds", summary="Geographic bounds for a layer")
def get_layer_bounds(
    city: Annotated[str, FPath(description="City key")],
    name: Annotated[str, FPath(description="Layer name")],
) -> dict:
    """Return geographic bounds for deck.gl BitmapLayer positioning.

    ``bounds`` is ``[minLon, minLat, maxLon, maxLat]``.
    """
    key = _validate_city(city)
    if name not in _LAYER_FILES:
        raise HTTPException(404, f"Unknown layer '{name}'. Available: {list(_LAYER_FILES)}")
    return {"name": name, "bounds": _city_bbox(key)}
