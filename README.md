# SheetalAI

> Satellite-powered urban heat intelligence. Ingests imagery for any city, maps surface temperature hotspots, explains *why* each area is hot using ML, simulates the cooling effect of interventions, and ranks zones by human impact — all surfaced in an interactive dark-themed dashboard.

**Pilot city:** Ahmedabad, India  
**Cost:** ₹0 — fully free and open-source. Runs entirely on a laptop with no cloud hosting, no database server, no tile server.

---

## What it does

SheetalAI turns raw satellite data into actionable urban heat analysis in six steps:

| Step | What happens | Output |
|------|-------------|--------|
| **Data pipeline** | Pulls Landsat 8 imagery from Google Earth Engine (or generates synthetic data offline) and builds a pixel-aligned 11-band feature stack | `stack.tif`, `pixels.parquet` |
| **Hotspot detection** | Runs Getis-Ord Gi* spatial statistics with Benjamini-Hochberg FDR correction to find statistically significant heat clusters | `hotspots.tif`, `hotspots.png` |
| **Driver model** | XGBoost trained with spatial-block cross-validation explains which urban features (impervious surfaces, vegetation loss, etc.) drive temperatures up | `driver_xgb.json`, `prediction.tif` |
| **SHAP attribution** | TreeExplainer assigns each pixel a per-driver temperature contribution in °C, producing a dominant-driver raster | `shap_zones.geojson`, `shap_global.json` |
| **Cooling simulation** | Counterfactual: perturbs each driver (urban greening, tree canopy, cool roofs) and re-predicts LST to estimate ΔLST with uncertainty bands | `simulation.parquet`, `simulation_summary.json` |
| **Equity prioritisation** | Scores zones as `0.4 × heat + 0.3 × population + 0.3 × vulnerability` to rank where cooling interventions matter most to people | `priority_zones.geojson` |

Everything above runs offline. The FastAPI backend serves pre-computed results; the Next.js dashboard visualises them with MapLibre GL + deck.gl.

---

## Architecture

```
Satellite imagery (Landsat 8 via GEE)
        │
        ▼
┌─────────────────────┐
│   data-pipeline/    │  gee_export.py  ──►  raw .tif layers
│                     │  features.py    ──►  stack.tif + pixels.parquet
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│       ml/           │  hotspots.py   ──►  Gi* hotspot raster
│                     │  train.py      ──►  XGBoost driver model
│                     │  explain.py    ──►  SHAP attribution raster
│                     │  simulate.py   ──►  counterfactual ΔLST
│                     │  prioritize.py ──►  equity-ranked zone GeoJSON
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│       api/          │  FastAPI — read-only over cached results
│                     │  GET /cities/{city}/zones
│                     │  GET /cities/{city}/layers/{layer}
│                     │  GET /cities/{city}/summary
│                     │  GET /cities/{city}/zones/{id}/advisory  (optional)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│       web/          │  Next.js 16 + MapLibre GL JS + deck.gl
│                     │  Dark premium dashboard, click-to-inspect zones
└─────────────────────┘
```

The pipeline runs **once** and caches results as local files. The API and dashboard only read those files — no heavy compute at request time.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Satellite data | Google Earth Engine (free non-commercial tier) · USGS/Copernicus-ready abstraction |
| Raster I/O | `rasterio`, `numpy`, Cloud-Optimised GeoTIFF (COG) |
| Feature engineering | Pure-numpy spectral indices (NDVI, NDBI, MNDWI, albedo) |
| Spatial stats | Getis-Ord Gi* by FFT convolution · Benjamini-Hochberg FDR |
| ML model | XGBoost (`hist` tree method) + spatial-block cross-validation |
| Explainability | SHAP `TreeExplainer` (exact, additivity-checked) |
| Python env | `uv` pinned to Python 3.12 (geospatial wheel compatibility) |
| API | FastAPI + Uvicorn · CORS-enabled · read-only |
| Maps | MapLibre GL JS + deck.gl (`BitmapLayer` for rasters, `GeoJsonLayer` for zones) |
| Frontend | Next.js 16 (App Router) · TypeScript · Tailwind v4 · Framer Motion |
| Optional LLM | Groq free tier (`llama-3.3-70b-versatile`) behind `ENABLE_ADVISORY` flag |

---

## Project layout

```
sheetalAI/
├── data-pipeline/
│   ├── config/
│   │   └── cities.yaml          # add a city here — no code changes needed
│   ├── sources/
│   │   ├── base.py              # DataSource ABC (swap point for GEE → USGS/Copernicus)
│   │   ├── gee.py               # real Landsat 8 C02 L2 export via geemap
│   │   └── synthetic.py         # offline deterministic city (default, no credentials)
│   ├── gee_export.py            # entrypoint: selects source, writes raw .tif layers
│   ├── features.py              # reprojects onto canonical grid, builds 11-band stack
│   ├── grid.py                  # canonical 30 m grid definition
│   ├── indices.py               # spectral indices in pure numpy
│   └── raster_io.py             # COG read/write helpers
│
├── ml/
│   ├── hotspots.py              # Getis-Ord Gi* + FDR
│   ├── train.py                 # XGBoost + spatial-block CV
│   ├── explain.py               # SHAP attribution
│   ├── simulate.py              # counterfactual ΔLST for 3 interventions
│   ├── prioritize.py            # equity-weighted zone scoring
│   └── models/
│       ├── driver_xgb.json      # trained booster
│       └── driver_meta.json     # feature order + observed ranges
│
├── api/
│   ├── main.py                  # FastAPI app factory
│   ├── config.py                # env config (cities, paths, flags)
│   └── routers/
│       ├── zones.py             # /cities/{city}/zones
│       ├── layers.py            # /cities/{city}/layers/{layer}
│       ├── summary.py           # /cities/{city}/summary
│       └── advisory.py          # /cities/{city}/zones/{id}/advisory  (Phase 8)
│
├── web/
│   └── src/
│       ├── app/
│       │   ├── page.tsx         # dashboard page (city stats + map + zone panel)
│       │   └── layout.tsx
│       ├── components/
│       │   ├── MapView.tsx      # MapLibre GL + deck.gl layers
│       │   ├── StatsPanel.tsx   # left sidebar (stats, SHAP drivers, layer switcher)
│       │   └── ZonePanel.tsx    # right panel (zone detail on click)
│       └── lib/
│           └── api.ts           # typed fetch wrappers for the FastAPI backend
│
├── notebooks/
│   └── 01_phase1_data_check.ipynb
│
├── data/                        # gitignored — created at runtime
│   └── ahmedabad/
│       ├── stack.tif            # 11-band Cloud-Optimised GeoTIFF
│       ├── pixels.parquet       # per-pixel feature rows
│       ├── hotspots.tif         # Gi* z-scores + significance class
│       ├── prediction.tif       # model prediction + residual
│       ├── shap_zones.geojson   # per-zone signed SHAP values (EPSG:4326)
│       ├── simulation.parquet   # per-pixel ΔLST per intervention
│       └── priority_zones.geojson  # equity-ranked zones for the dashboard
│
├── .env.example                 # copy to .env and fill in
├── run_local.sh                 # one-shot: pipeline → api → web
└── docker-compose.yml           # optional PostGIS upgrade (not required)
```

---

## Prerequisites

### macOS (tested on macOS 15, Apple Silicon + Intel)

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/brew.sh/HEAD/install.sh)"

# uv — Python env manager (replaces pip/conda/pyenv for this project)
curl -LsSf https://astral.sh/uv/install.sh | sh

# OpenMP runtime — required by XGBoost on macOS
brew install libomp

# Node 22 (or any Node 18+) — required for the Next.js dashboard
brew install node
```

**Python:** The repo pins envs to Python 3.12 via `uv`. You do **not** need to install Python 3.12 yourself — `uv` fetches it automatically.

**System Python ≥ 3.13:** Several geospatial wheels (rasterio, shapely, GDAL) do not yet publish binaries for 3.13+. The `uv` pin to 3.12 sidesteps this automatically.

---

## Setup

### 1. Clone

```bash
git clone git@github.com:xOAviOx/sheetalAI.git
cd sheetalAI
```

### 2. Environment file

```bash
cp .env.example .env
```

Open `.env`. The only fields you need to fill in for a first run are:

```env
CITY=ahmedabad          # which city to process
DATA_SOURCE=synthetic   # use synthetic (offline) or gee (real satellite)
```

Everything else has sensible defaults.

### 3. Install Python dependencies

Each sub-project has its own `uv` env. Install all three at once:

```bash
cd data-pipeline && uv sync && cd ..
cd ml           && uv sync && cd ..
cd api          && uv sync && cd ..
```

### 4. Install frontend dependencies

```bash
cd web && npm install && cd ..
```

---

## Running the app

### Option A — one command (recommended)

```bash
# Full pipeline + API + dashboard
./run_local.sh

# Skip the pipeline if you already have cached data under data/ahmedabad/
./run_local.sh --skip-pipeline
```

The script runs the pipeline sequentially, then starts the API in the background and the Next.js dev server in the foreground. Press `Ctrl+C` to shut everything down.

### Option B — step by step

Run each stage in its own terminal tab so you can see logs separately.

**Terminal 1 — Data pipeline**

```bash
cd data-pipeline

# Generate the 11-band feature stack (uses synthetic data by default, no credentials needed)
DATA_SOURCE=synthetic uv run python gee_export.py
uv run python features.py
```

**Terminal 2 — ML pipeline**

```bash
cd ml

uv run python hotspots.py    # Getis-Ord Gi* hotspot detection
uv run python train.py       # XGBoost driver model + spatial CV
uv run python explain.py     # SHAP attribution per pixel and zone
uv run python simulate.py    # counterfactual cooling for 3 interventions
uv run python prioritize.py  # equity-weighted zone ranking
```

**Terminal 3 — API**

```bash
cd api
uv run uvicorn main:app --reload
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs   (interactive Swagger UI)
```

**Terminal 4 — Dashboard**

```bash
cd web
npm run dev
# → http://localhost:3000
```

---

## Using the dashboard

Open **http://localhost:3000** in a browser after both the API and `npm run dev` are running.

### Layout

```
┌──────────────┬────────────────────────────┬──────────────┐
│  Stats panel │         Map view           │  Zone panel  │
│  (left)      │         (centre)           │  (right)     │
└──────────────┴────────────────────────────┴──────────────┘
```

### Left panel — Stats & layer switcher

- **City stats:** mean LST, hotspot coverage %, model R², top SHAP drivers
- **Layer switcher:** toggle between the five analysis layers:
  - `Equity priority zones` — colour-coded by equity score (default)
  - `Getis-Ord Gi* hotspots` — statistically significant heat clusters
  - `Dominant warming driver` — which feature most raises each pixel's temperature
  - `Best cooling intervention ΔLST` — projected surface cooling in °C
  - `Equity priority score` — raw composite score raster

### Map — Centre

- Raster layers load as deck.gl `BitmapLayer` (PNG tiles served by FastAPI)
- Zone layer loads as deck.gl `GeoJsonLayer` coloured by equity score
- **Click any zone** to open the zone detail panel

### Right panel — Zone detail

Shows on zone click:
- Surface temperature vs city mean
- Population density and vulnerability index
- Equity score and rank (e.g. #3 of 127)
- Per-driver SHAP values (warming/cooling contributions in °C)
- Best cooling intervention and projected ΔLST
- If `ENABLE_ADVISORY=true` — a 2–3 sentence plain-English advisory from Groq LLM

---

## API reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe + advisory flag status |
| `GET` | `/cities` | List configured cities with bounding boxes |
| `GET` | `/cities/{city}/zones` | Priority zone GeoJSON FeatureCollection |
| `GET` | `/cities/{city}/layers/{layer}` | PNG raster for a named layer |
| `GET` | `/cities/{city}/summary` | Aggregated stats, SHAP global, simulation summary |
| `GET` | `/cities/{city}/zones/{zone_id}/advisory` | LLM advisory (requires `ENABLE_ADVISORY=true`) |

Interactive docs: **http://127.0.0.1:8000/docs**

Available `{layer}` values: `hotspots`, `shap_dominant`, `simulation`, `priority`

---

## Using real satellite data (Google Earth Engine)

By default the pipeline runs in **synthetic mode** — it generates a physics-inspired deterministic city from spectral endmembers, so you can run everything with zero credentials. All downstream stages (ML, API, dashboard) are identical regardless of data source.

To switch to real Landsat 8 imagery:

1. **Sign up** for a free GEE non-commercial account at https://earthengine.google.com/signup/
2. **Create** a Google Cloud project and note the project ID
3. **Authenticate** (one-time, opens a browser):
   ```bash
   cd data-pipeline
   uv run earthengine authenticate
   ```
4. **Set env vars** in your `.env`:
   ```env
   DATA_SOURCE=gee
   GEE_PROJECT=your-cloud-project-id
   ```
5. Re-run the pipeline:
   ```bash
   cd data-pipeline
   DATA_SOURCE=gee uv run python gee_export.py
   uv run python features.py
   ```

The rest of the pipeline (ML → API → dashboard) runs identically.

> **Note:** GEE free tier is for non-commercial use only. For a commercial deployment swap `gee.py` with a `usgs.py` or `copernicus.py` source — the `DataSource` ABC in `sources/base.py` is the swap point; no downstream code changes needed.

---

## Adding a new city

Adding a city is a config change — no code changes required.

Edit `data-pipeline/config/cities.yaml`:

```yaml
mumbai:
  display_name: "Mumbai"
  country: "India"
  bbox: [72.77, 18.87, 72.99, 19.27]   # [minLon, minLat, maxLon, maxLat]
  utm_epsg: 32643                        # UTM zone for the AOI
  grid_size_m: 30                        # pixel resolution in metres
  zone_aggregation: grid                 # "grid" or "ward" (if ward shapefile provided)
  zone_grid_size_m: 750                  # zone cell size (metres)
  date_range: ["2024-03-01", "2024-06-15"]
  cloud_cover_max: 20
  ward_boundary: null                    # path to ward GeoJSON/shapefile, or null
```

Then run the pipeline with `CITY=mumbai`:

```bash
CITY=mumbai cd data-pipeline && uv run python gee_export.py && uv run python features.py
CITY=mumbai cd ml && uv run python hotspots.py && uv run python train.py && \
                     uv run python explain.py && uv run python simulate.py && uv run python prioritize.py
```

The API and dashboard automatically pick up the new city from `/cities`.

---

## Optional: Groq LLM advisory (Phase 8)

When enabled, clicking a zone shows a 2–3 sentence plain-English advisory generated by Groq's free-tier LLM. Responses are cached in-process (free — repeated clicks cost nothing).

1. Get a free API key at https://console.groq.com/
2. Add to `.env`:
   ```env
   ENABLE_ADVISORY=true
   GROQ_API_KEY=gsk_...
   GROQ_MODEL=llama-3.3-70b-versatile   # or llama-3.1-8b-instant for faster/cheaper
   ```
3. Restart the API (`Ctrl+C` then `uv run uvicorn main:app --reload`)

The dashboard shows the advisory button automatically when the flag is on. The app works fully without it.

---

## Feature stack (11 bands)

The 11-band stack written by `features.py` — band order matters for the model:

| Band | Description |
|------|-------------|
| `lst_c` | Land-surface temperature (°C) — prediction target |
| `ndvi` | Normalised Difference Vegetation Index |
| `ndbi` | Normalised Difference Built-up Index |
| `mndwi` | Modified Normalised Difference Water Index |
| `albedo` | Broadband surface albedo |
| `impervious_frac` | Impervious surface fraction |
| `dist_to_water` | Distance to nearest water body (m) |
| `elevation` | Terrain elevation (m, NASADEM) |
| `slope` | Terrain slope (degrees) |
| `pop_density` | Population density (WorldPop, people/km²) |
| `vulnerability` | Social vulnerability index |

---

## Interventions simulated (Phase 4)

| Intervention | What it perturbs |
|---|---|
| `urban_greening` | Converts impervious surface → green, raises NDVI, lowers NDBI |
| `tree_canopy` | Increases NDVI and albedo (canopy shade + reflectance) |
| `cool_roofs` | Raises albedo on impervious surfaces |

Each intervention is clamped to the observed data range so predictions stay within the model's training distribution. ΔLST estimates carry the model's spatial-CV RMSE as the uncertainty band.

---

## Troubleshooting

**`libomp` not found (XGBoost crash on macOS)**
```bash
brew install libomp
```

**`uv` not found**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then open a new terminal or source your shell profile
```

**Port already in use**
```bash
# Kill whatever is on 8000
lsof -ti:8000 | xargs kill -9

# Kill whatever is on 3000
lsof -ti:3000 | xargs kill -9
```

**Map is blank / "API unreachable"**
Make sure the FastAPI server is running on port 8000 before opening the dashboard:
```bash
cd api && uv run uvicorn main:app --reload
```

**Pipeline produces NaN pixels**
Ensure all pipeline scripts ran to completion in order:
`gee_export.py` → `features.py` → `hotspots.py` → `train.py` → `explain.py` → `simulate.py` → `prioritize.py`

---

## Design principles

- **Local-first.** No cloud hosting, no DB, no tile server. All outputs are local files (COG/PNG rasters, GeoParquet, GeoJSON). PostGIS / TiTiler upgrades are possible but optional.
- **Config-driven cities.** `cities.yaml` is the only file that changes when adding a city.
- **Swappable data source.** `DataSource` ABC in `sources/base.py` decouples GEE from all downstream code. Swap the source, nothing else changes.
- **Honest uncertainty.** All ΔLST values are estimates labelled with the model's spatial-CV RMSE. The word "guarantee" does not appear in the codebase.
- **Free forever.** Every dependency — GEE non-commercial, XGBoost, MapLibre GL, Groq free tier — has a zero-cost tier. The constraint is architectural, not optional.
