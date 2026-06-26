"""GEE export — Phase 1. Exports cloud-masked median composites (LST + SR bands,
WorldCover, WorldPop, DEM) as Cloud-Optimized GeoTIFFs to data/{city}/raw/.

Implemented in Phase 1.
"""

from __future__ import annotations

from config import load_city


def main() -> None:
    cfg = load_city()
    raise SystemExit(
        f"[gee_export] Phase 1 not yet implemented (city={cfg.key}). "
        "Scaffold only — see build plan Phase 1."
    )


if __name__ == "__main__":
    main()
