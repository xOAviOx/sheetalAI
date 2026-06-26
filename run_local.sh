#!/usr/bin/env bash
# ============================================================
# SheetalAI — one-shot local runner
#   (1) run the data pipeline + ml for the city
#   (2) start the FastAPI server (localhost:8000)
#   (3) start the Next.js web app (localhost:3000)
#
# Usage:
#   ./run_local.sh                 # full pipeline + api + web
#   ./run_local.sh --skip-pipeline # api + web only (use cached data)
#   CITY=ahmedabad ./run_local.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# load .env if present (for CITY, ports, etc.)
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi

CITY="${CITY:-ahmedabad}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
SKIP_PIPELINE=false

for arg in "$@"; do
  case "$arg" in
    --skip-pipeline) SKIP_PIPELINE=true ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m[run_local]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------
# 1. Data pipeline + ML  (offline, heavy compute, cached)
# ------------------------------------------------------------
if [[ "$SKIP_PIPELINE" == "false" ]]; then
  log "Running data pipeline for city: $CITY"
  ( cd data-pipeline && uv sync --quiet \
      && CITY="$CITY" uv run python gee_export.py \
      && CITY="$CITY" uv run python features.py )

  log "Running ML stage for city: $CITY"
  ( cd ml && uv sync --quiet \
      && CITY="$CITY" uv run python hotspots.py \
      && CITY="$CITY" uv run python train.py \
      && CITY="$CITY" uv run python explain.py \
      && CITY="$CITY" uv run python simulate.py \
      && CITY="$CITY" uv run python prioritize.py )
else
  log "Skipping pipeline (using cached data under data/$CITY/)"
fi

# ------------------------------------------------------------
# 2. API  (background)
# ------------------------------------------------------------
log "Starting API at http://$API_HOST:$API_PORT (docs at /docs)"
( cd api && uv sync --quiet \
    && uv run uvicorn main:app --host "$API_HOST" --port "$API_PORT" ) &
API_PID=$!

cleanup() {
  log "Shutting down..."
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------
# 3. Web  (foreground)
# ------------------------------------------------------------
log "Starting web app at http://localhost:3000"
( cd web && npm install --silent && npm run dev )
