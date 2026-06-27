"""Summary / analytics endpoints — Phase 6.

Aggregates the JSON summaries written by each ML phase into dashboard-ready
responses: city-level stats panel, global SHAP bar chart, simulation details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as FPath

from config import city_data_dir, load_cities

router = APIRouter(prefix="/cities/{city}", tags=["summary"])


def _validate_city(city: str) -> str:
    cities = load_cities()
    key = city.strip().lower()
    if key not in cities:
        raise HTTPException(404, f"City '{city}' not found. Available: {sorted(cities)}")
    return key


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise HTTPException(503, f"{label} not found. Run the ML pipeline first.")
    return json.loads(path.read_text())


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.get("/summary", summary="Aggregated city summary (all phases)")
def get_summary(
    city: Annotated[str, FPath(description="City key, e.g. ahmedabad")],
) -> dict:
    """Return a merged summary across all ML phases for the city stats panel.

    Pulls metrics from hotspots_summary, train_summary, shap_summary,
    simulation_summary, and priority_summary. Missing phases return null
    values so the dashboard degrades gracefully.
    """
    key = _validate_city(city)
    ddir = city_data_dir(key)

    def _try(fname: str) -> dict:
        try:
            return json.loads((ddir / fname).read_text())
        except FileNotFoundError:
            return {}

    features = _try("features_summary.json")
    hotspots = _try("hotspots_summary.json")
    train    = _try("train_summary.json")
    shap     = _try("shap_summary.json")
    sim      = _try("simulation_summary.json")
    priority = _try("priority_summary.json")

    # hotspots: sum hot classes (99+95+90) and cold classes
    hs_cls = hotspots.get("classes", {})
    pct_hot  = sum(hs_cls.get(k, {}).get("pct", 0) for k in ("hot_99", "hot_95", "hot_90"))
    pct_cold = sum(hs_cls.get(k, {}).get("pct", 0) for k in ("cold_99", "cold_95", "cold_90"))

    # Flatten into a single dashboard-friendly object
    lst_stats = features.get("lst_c_stats", {})
    grid_meta = features.get("grid", {})
    return {
        "city": key,
        "data": {
            "n_pixels":    features.get("n_pixels"),
            "lst_mean_c":  lst_stats.get("mean"),
            "lst_min_c":   lst_stats.get("min"),
            "lst_max_c":   lst_stats.get("max"),
            "grid_size_m": grid_meta.get("res_m"),
        },
        "hotspots": {
            "pct_hot":         round(pct_hot, 2) if pct_hot else None,
            "pct_cold":        round(pct_cold, 2) if pct_cold else None,
            "pct_ns":          hs_cls.get("not_sig", {}).get("pct"),
            "lst_hot_mean_c":  hotspots.get("sanity", {}).get("mean_lst_hot95plus"),
            "lst_cold_mean_c": hotspots.get("sanity", {}).get("mean_lst_cold95plus"),
        },
        "model": {
            "spatial_cv_r2":   train.get("metrics", {}).get("spatial_cv", {}).get("r2"),
            "spatial_cv_rmse": train.get("metrics", {}).get("spatial_cv", {}).get("rmse"),
            "spatial_cv_mae":  train.get("metrics", {}).get("spatial_cv", {}).get("mae"),
        },
        "shap": {
            "n_zones": shap.get("n_zones"),
            "top_driver": (
                max(
                    shap.get("zone_dominant_driver_counts", {}).items(),
                    key=lambda kv: kv[1],
                    default=(None, None),
                )[0]
            ),
            "driver_zone_counts": shap.get("zone_dominant_driver_counts"),
        },
        "simulation": {
            "strongest_intervention": sim.get("strongest"),
            "model_rmse_c": sim.get("model_spatial_cv_rmse_c"),
            "interventions": {
                k: {
                    "label": v["label"],
                    "pct_city": v["pct_city"],
                    "central_median_cooling_c": v["band_c"].get("central_median"),
                    "band_low_c":    v["band_c"].get("low_median"),
                    "band_high_c":   v["band_c"].get("high_median"),
                    "in_literature_range": v["central_in_literature_range"],
                    "clamp_limited": v["clamp_limited"],
                }
                for k, v in sim.get("interventions", {}).items()
            },
        },
        "priority": {
            "n_zones": priority.get("n_zones"),
            "weights": priority.get("weights"),
            "equity_score_range": priority.get("equity_score_range"),
            "best_intervention_distribution": priority.get("best_intervention_distribution"),
        },
    }


@router.get("/shap/global", summary="Global SHAP driver importances")
def get_shap_global(
    city: Annotated[str, FPath(description="City key")],
) -> dict:
    """Return mean |SHAP| per driver for the global importance bar chart.

    Drivers sorted by importance descending. Values are in °C.
    """
    key = _validate_city(city)
    ddir = city_data_dir(key)
    data = _read_json(ddir / "shap_global.json", "shap_global.json")

    # shap_global.json schema: {city, base_value_c, mean_abs_shap_c: {driver: value}}
    importances: dict = data.get("mean_abs_shap_c", data.get("mean_abs_shap", {}))
    sorted_drivers = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "city": key,
        "unit": "°C",
        "note": "Mean absolute SHAP value per driver across all valid pixels.",
        "importances": [{"driver": d, "mean_abs_shap": v} for d, v in sorted_drivers],
    }


@router.get("/simulate", summary="Simulation results (Phase 4 cached)")
def get_simulate(
    city: Annotated[str, FPath(description="City key")],
) -> dict:
    """Return cached Phase 4 simulation summary.

    Includes per-intervention applicable pixel count, cooling band
    (conservative/central/ambitious), literature range, and clamp flags.
    """
    key = _validate_city(city)
    ddir = city_data_dir(key)
    return _read_json(ddir / "simulation_summary.json", "simulation_summary.json")


@router.get("/priority", summary="Priority ranking summary (Phase 5)")
def get_priority_summary(
    city: Annotated[str, FPath(description="City key")],
) -> dict:
    """Return Phase 5 priority summary including top-10 zones."""
    key = _validate_city(city)
    ddir = city_data_dir(key)
    return _read_json(ddir / "priority_summary.json", "priority_summary.json")
