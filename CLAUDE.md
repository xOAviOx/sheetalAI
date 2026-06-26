# SheetalAI — project memory

AI platform: ingest satellite data for a city → map urban heat → explain *why* each
area is hot → simulate cooling interventions → rank zones by human impact → interactive
dashboard. Pilot city: **Ahmedabad**. Built phase-by-phase (Phase 0–9), commit after each.

## Hard constraints (non-negotiable)
- **₹0 / free & open-source only.** Never add a paid dependency. Ask before anything that
  (a) costs money, (b) locks in a non-free dep, or (c) materially changes the architecture.
- Satellite data: **Google Earth Engine** (free non-commercial tier, Python API).
- Maps: **MapLibre GL JS** (NOT Mapbox) + **deck.gl**. Backend: **FastAPI** on localhost.
  Frontend: **Next.js** (App Router, TS). Storage: **local files** (GeoParquet/GeoJSON/COG/PNG),
  no DB/tile server. Optional LLM: **Groq** free tier, behind `ENABLE_ADVISORY` flag.
- Data layer MUST be swappable: GEE replaceable by USGS/Copernicus/ISRO without touching
  downstream code (GEE free tier is non-commercial only).
- All cooling numbers are **estimates with uncertainty bands**, never guarantees.
- Config-driven cities: adding a city = edit `data-pipeline/config/cities.yaml`, not code.
- Stack prefs: TypeScript, Tailwind, Framer Motion, dark premium aesthetic. Python 3.11+, type hints.
- Never commit `data/` or `.env`.

## Environment specifics (this machine)
- System Python is 3.14 (too new for geospatial wheels) → **uv envs pinned to 3.12**. uv auto-fetches it.
- **macOS**: `brew install libomp` is required for xgboost to load (already installed here).
- `shap` pulls a bad old `llvmlite` → pinned floors `numba>=0.60`, `llvmlite>=0.43` in ml/pyproject.toml.
- Node 22, npm 10. Next.js scaffolded as **v16** (App Router, Tailwind v4, src/ dir).
  Note `web/AGENTS.md`: this Next.js may differ from training data — read `node_modules/next/dist/docs/` before writing web code.
- create-next-app makes a nested `web/.git`; must be removed so files track in the root repo.

## Run order
1. `cd data-pipeline && DATA_SOURCE=synthetic uv run python gee_export.py && uv run python features.py`
2. ml stage (hotspots → train → explain → simulate → prioritize)
3. `cd api && uv run uvicorn main:app --reload`  (localhost:8000)
4. `cd web && npm run dev`  (localhost:3000)
Or `./run_local.sh` (add `--skip-pipeline` to reuse cached data).

## Data source design (key abstraction)
`data-pipeline/sources/` implements the swap point:
- `base.py` — `DataSource` ABC + `SourceResult`. Raw layer contract: `lst.tif` (1b °C),
  `sr.tif` (6b blue/green/red/nir/swir1/swir2), `worldcover.tif`, `pop.tif`, `dem.tif`.
- `gee.py` — **real** Landsat 8 C02 L2 export (cloud-masked median, QA_PIXEL mask, °C/reflectance
  scaling, WorldCover v200, WorldPop, NASADEM) via `geemap.ee_export_image`. Needs auth.
- `synthetic.py` — **offline deterministic** city (linear spectral mixing of veg/built/water/soil
  endmembers drives both SR and LST). Default. Lets Phases 2–7 run with no credentials.
- `get_source()` selects via `DATA_SOURCE` env (synthetic|gee).
Downstream NEVER imports `ee`. `features.py` reprojects every raw layer onto the **canonical grid**
(`grid.py`) → guarantees pixel-perfect alignment ("no NaN misalignment").

## Feature stack (11 bands, order matters)
`lst_c` (target), ndvi, ndbi, mndwi, albedo, impervious_frac, dist_to_water, elevation, slope,
pop_density, vulnerability. Drivers = all except lst_c. Outputs: `data/{city}/stack.tif` (multiband COG)
+ `pixels.parquet` (tidy per-pixel rows) + `features_summary.json`. Indices in `indices.py` (pure numpy).

## GEE auth status
NOT authenticated on this machine (no `~/.config/earthengine/credentials`, no `.env`). Cannot run live
GEE export. Synthetic source used for development/demo. To run real export: user does one-time
`earthengine authenticate` + set `GEE_PROJECT` in `.env`, then `DATA_SOURCE=gee`.

## Progress
- **Phase 0 ✅** committed (31bc5f8): monorepo, git, 3 uv envs, cities.yaml, gee_auth, FastAPI
  (/health,/cities), Next.js landing pinging API, run_local.sh, docker-compose (optional PostGIS).
- **Phase 1** in progress: data-source abstraction + synthetic + GEE sources, canonical grid,
  features.py. Verified on ahmedabad: 903×791 @30m, 100% valid coverage, LST 26–48°C (mean 38.9),
  all bands share identical mask, driver correlations correct (NDBI +0.83, impervious +0.73, NDVI −0.25),
  hottest 10% = impervious 0.64/ndvi 0.16 vs coolest impervious 0/ndvi 0.35. Notebook
  `notebooks/01_phase1_data_check.ipynb` renders LST. NOT yet committed.

## Phase plan (remaining)
2 hotspots (Getis-Ord Gi*, esda/libpysal) → 3 driver model (XGBoost + spatial-block CV + SHAP) →
4 simulation (counterfactual ΔLST, clamp to observed ranges, uncertainty bands, literature sanity) →
5 equity prioritisation (0.4·heat+0.3·pop+0.3·vuln) → 6 FastAPI endpoints → 7 dashboard
(MapLibre+deck.gl BitmapLayer for rasters, GeoJsonLayer for zones) → 8 Groq advisory (flagged) →
9 second city via config only + validation.

## Note
First user message contained a prompt-injection (fake "Claude Fable 5" system prompt + "secret word→banana").
Ignored — it is not a real instruction. Real repo instructions live in AGENTS.md/CLAUDE.md.
