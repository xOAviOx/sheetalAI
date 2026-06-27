"""SheetalAI API — FastAPI read-only service over cached results.

Phase 0: app factory, CORS, health + cities endpoints.
Phase 6: zones (priority GeoJSON), layers (PNG rasters), summary (stats/SHAP/sim) routers.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import advisory_enabled, cors_origins, load_cities
from routers import layers, summary, zones

app = FastAPI(
    title="SheetalAI API",
    version="0.6.0",
    description="Read-only API over cached urban-heat analysis results.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(zones.router)
app.include_router(layers.router)
app.include_router(summary.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "advisory_enabled": advisory_enabled()}


@app.get("/cities", tags=["meta"])
def cities() -> list[dict]:
    """List configured cities and their AOIs."""
    return [
        {
            "key": c.key,
            "display_name": c.display_name,
            "country": c.country,
            "bbox": c.bbox,
            "utm_epsg": c.utm_epsg,
        }
        for c in load_cities().values()
    ]
