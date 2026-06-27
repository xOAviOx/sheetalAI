"""Groq advisory endpoint — Phase 8.

Generates a 2–3 sentence natural-language advisory for a zone using the
Groq free-tier LLM (llama-3.1-8b-instant by default).

Gated behind ENABLE_ADVISORY=true (env / .env). When the flag is off the
endpoint returns 503 immediately so the dashboard can hide the button.
Responses are cached in-process so repeated clicks on the same zone cost
nothing.

Requires GROQ_API_KEY in environment / .env.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated

import requests as req
from fastapi import APIRouter, HTTPException, Path as FPath

from config import advisory_enabled, city_data_dir, load_cities

router = APIRouter(prefix="/cities/{city}", tags=["advisory"])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

DRIVER_LABELS: dict[str, str] = {
    "ndvi":            "low vegetation (NDVI)",
    "ndbi":            "high built-up index (NDBI)",
    "mndwi":           "low water presence (MNDWI)",
    "albedo":          "low surface albedo",
    "impervious_frac": "high impervious surface cover",
    "dist_to_water":   "distance from water bodies",
    "elevation":       "elevation",
    "slope":           "slope",
    "pop_density":     "population density",
    "vulnerability":   "social vulnerability index",
}

INTERVENTION_LABELS: dict[str, str] = {
    "urban_greening": "converting built-up land to green space",
    "tree_canopy":    "planting street trees and canopy cover",
    "cool_roofs":     "installing high-reflectivity cool roofs",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_city(city: str) -> str:
    key = city.strip().lower()
    if key not in load_cities():
        raise HTTPException(404, f"City '{city}' not found.")
    return key


@lru_cache(maxsize=1)
def _load_zones(city: str, _mtime: float) -> dict[int, dict]:
    """Return {zone_id: properties} dict, cached until file changes."""
    p = city_data_dir(city) / "priority_zones.geojson"
    if not p.exists():
        raise HTTPException(503, "priority_zones.geojson not found. Run the ML pipeline first.")
    fc = json.loads(p.read_text())
    return {f["properties"]["zone_id"]: f["properties"] for f in fc["features"]}


def _zone_props(city: str, zone_id: int) -> dict:
    p = city_data_dir(city) / "priority_zones.geojson"
    mtime = p.stat().st_mtime if p.exists() else 0.0
    zones = _load_zones(city, mtime)
    if zone_id not in zones:
        raise HTTPException(404, f"Zone {zone_id} not found in city '{city}'.")
    return zones[zone_id]


def _city_context(city: str) -> dict:
    """Pull city-level stats needed for the prompt."""
    ddir = city_data_dir(city)
    ctx: dict = {}
    try:
        fs = json.loads((ddir / "features_summary.json").read_text())
        ctx["lst_mean"] = round(fs["lst_c_stats"]["mean"], 1)
    except Exception:
        ctx["lst_mean"] = None
    try:
        ps = json.loads((ddir / "priority_summary.json").read_text())
        ctx["n_zones"] = ps["n_zones"]
    except Exception:
        ctx["n_zones"] = None
    try:
        ts = json.loads((ddir / "train_summary.json").read_text())
        ctx["rmse"] = round(ts["metrics"]["spatial_cv"]["rmse"], 2)
    except Exception:
        ctx["rmse"] = None
    return ctx


def _build_prompt(city: str, props: dict, ctx: dict) -> tuple[str, str]:
    """Build the LLM prompt from zone properties and city context."""
    shap_drivers = sorted(
        [
            (k.replace("shap_", ""), v)
            for k, v in props.items()
            if k.startswith("shap_") and isinstance(v, float) and v > 0
        ],
        key=lambda x: -x[1],
    )[:3]
    shap_text = ", ".join(
        f"{DRIVER_LABELS.get(d, d)} (+{v:.2f}°C)" for d, v in shap_drivers
    ) or "no dominant single driver"

    interv = props.get("best_intervention")
    interv_label = INTERVENTION_LABELS.get(interv or "", interv or "no clear intervention")
    delta = props.get("best_delta_lst_c")
    cooling_text = (
        f"estimated {abs(delta):.1f}°C surface cooling" if delta is not None else "uncertain cooling"
    )
    rmse = ctx.get("rmse")
    if rmse:
        cooling_text += f" (±{rmse:.2f}°C model uncertainty)"

    rank_of = (
        f"#{props['equity_rank']} of {ctx['n_zones']}"
        if ctx.get("n_zones") else f"#{props['equity_rank']}"
    )
    city_mean = f"{ctx['lst_mean']}°C" if ctx.get("lst_mean") else "city mean"

    system = (
        "You are a concise urban heat analyst for SheetalAI, a city heat intelligence platform. "
        "Write 2–3 sentences in plain English for a city planner. "
        "Be specific and quantitative. No bullet points. No hedging phrases like 'it is worth noting'. "
        "Do not repeat the zone ID or rank in every sentence. "
        "End with one concrete, actionable recommendation."
    )

    user = f"""Zone data for {city.title()}, priority rank {rank_of}:

- Surface temperature: {props['lst_c']:.1f}°C  (city mean {city_mean})
- Population density:  {props['pop_density']:,.0f} people/km²
- Vulnerability index: {props['vulnerability']:.3f}  (0 = none, 1 = extreme)
- Equity score:        {props['equity_score']:.3f}  (heat 40% + pop 30% + vuln 30%)
- Main warming causes: {shap_text}
- Best intervention:   {interv_label} → {cooling_text}

Write the advisory now."""

    return system, user


def _call_groq(system: str, user: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            503,
            "GROQ_API_KEY not set. Add it to your .env file to enable advisories.",
        )
    try:
        resp = req.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 220,
                "temperature": 0.35,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except req.exceptions.Timeout:
        raise HTTPException(504, "Groq API timed out — try again.")
    except req.exceptions.HTTPError as e:
        raise HTTPException(502, f"Groq API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(502, f"Groq call failed: {e}")


# In-process cache: (city, zone_id) → advisory text
_advisory_cache: dict[tuple[str, int], str] = {}


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get("/zones/{zone_id}/advisory", summary="AI advisory for a zone (Groq)")
def get_advisory(
    city: Annotated[str, FPath(description="City key, e.g. ahmedabad")],
    zone_id: Annotated[int, FPath(description="Zone ID")],
) -> dict:
    """Generate a plain-English advisory for the zone using the Groq LLM.

    Requires ``ENABLE_ADVISORY=true`` and ``GROQ_API_KEY`` in the environment.
    Responses are cached in-process; restart the server to clear the cache.
    """
    if not advisory_enabled():
        raise HTTPException(
            503,
            "Advisory layer is disabled. Set ENABLE_ADVISORY=true in your .env to enable it.",
        )

    key = _validate_city(city)
    cache_key = (key, zone_id)

    if cache_key in _advisory_cache:
        return {
            "zone_id": zone_id,
            "city": key,
            "advisory": _advisory_cache[cache_key],
            "model": GROQ_MODEL,
            "cached": True,
            "generated_at": None,
        }

    props = _zone_props(key, zone_id)
    ctx = _city_context(key)
    system, user = _build_prompt(key, props, ctx)
    text = _call_groq(system, user)

    _advisory_cache[cache_key] = text

    return {
        "zone_id": zone_id,
        "city": key,
        "advisory": text,
        "model": GROQ_MODEL,
        "cached": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
