# SheetalAI

> AI platform that ingests satellite data for a city, maps urban heat, explains **why** each area is hot, simulates the cooling effect of interventions, and ranks zones by human impact — surfaced in an interactive dark-themed dashboard.

**Pilot city:** Ahmedabad (India's first official Heat Action Plan — strong validation story).
**Cost:** ₹0. Free & open-source only. Runs entirely on a laptop — no hosting, no database server, no tile server.

---

## Architecture

```
DATA (GEE → cached COGs + local GeoParquet/GeoJSON)
  → FEATURES (common 30m grid, feature stack)
    → ML (hotspots · driver model+SHAP · intervention sim · prioritisation)
      → API (FastAPI, read-only over cached results)
        → FRONTEND (Next.js + MapLibre/deck.gl dashboard)
          → ADVISORY (optional Groq LLM, describes outputs only)
```

Heavy compute runs **offline** in the pipeline and is cached as analysis-ready layers. The API and frontend only read pre-computed results.

## Monorepo layout

```
sheetalai/
├── data-pipeline/   # GEE export + feature stack  (Python)
├── ml/              # hotspots, driver model+SHAP, simulation, prioritisation
├── api/             # FastAPI read-only service
├── web/             # Next.js dashboard
├── notebooks/       # EDA / validation
├── data/            # cached COGs + parquet  (gitignored)
├── run_local.sh     # pipeline → start api → start web
└── docker-compose.yml  # OPTIONAL PostGIS upgrade (not required)
```

---

## Prerequisites

- **Python 3.11+** (this repo's envs are pinned to 3.12 via `uv` for geospatial wheel compatibility)
- **Node 18+** (tested on Node 22)
- [`uv`](https://docs.astral.sh/uv/) for Python env management
- A free **Google Earth Engine** account (Phase 1 data pipeline only)
- **macOS only:** `brew install libomp` (free OpenMP runtime required by `xgboost`)

## One-time Google Earth Engine setup

GEE's free tier is **non-commercial**. The data layer is abstracted so GEE can later be
swapped for direct USGS/Copernicus/ISRO downloads without touching downstream code.

1. Sign up (free, non-commercial): https://earthengine.google.com/signup/
2. Create / note a Google Cloud project id and set `GEE_PROJECT` in your `.env`.
3. Authenticate once (opens a browser):
   ```bash
   cd data-pipeline
   uv run earthengine authenticate
   ```
   Credentials are cached by the `earthengine-api` library in your home dir (gitignored here via `.ee_credentials/`).
4. Verify:
   ```bash
   uv run python gee_auth.py
   ```

---

## Quick start (local)

```bash
# 0. configure
cp .env.example .env        # edit GEE_PROJECT (only needed for the pipeline)

# 1. run the data pipeline once for the city (Phase 1+)
#    (produces cached layers under data/ahmedabad/)
cd data-pipeline && uv sync && uv run python gee_export.py && uv run python features.py
cd ../ml && uv sync && uv run python hotspots.py && uv run python train.py && \
            uv run python explain.py && uv run python simulate.py && uv run python prioritize.py

# 2. start the API  (localhost:8000)
cd ../api && uv sync && uv run uvicorn main:app --reload

# 3. start the web app  (localhost:3000)
cd ../web && npm install && npm run dev
```

Or chain everything with the helper:

```bash
./run_local.sh            # full pipeline + api + web
./run_local.sh --skip-pipeline   # just api + web (uses cached data)
```

---

## Design principles

- **Local-first.** No hosting, no DB server, no tile server. Results are local files
  (GeoParquet zones/sims/plan, GeoJSON hotspots, COG/PNG rasters). PostGIS / TiTiler /
  Docker are optional later upgrades.
- **Config-driven cities.** Adding a city is a change to `data-pipeline/config/cities.yaml`,
  not code.
- **Swappable data source.** The data-access layer hides GEE behind an interface so a future
  commercial deployment can pull from USGS/Copernicus/ISRO directly.
- **Honest outputs.** All cooling numbers are **estimates with uncertainty bands**, never
  guarantees — surfaced in both API responses and UI copy.

---

## CHANGELOG

- **Phase 0 — Foundation.** Monorepo scaffold, git + `.gitignore`, Python envs
  (`data-pipeline`, `ml`, `api`) via `uv` pinned to 3.12, Next.js app in `web/`,
  `.env.example`, `run_local.sh`, optional `docker-compose.yml`, `gee_auth.py`,
  `config/cities.yaml` (Ahmedabad).
