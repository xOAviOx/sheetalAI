"""Feature stack — Phase 1. Computes NDVI/NDBI/MNDWI/albedo/impervious/
dist_to_water/elevation/slope/pop_density/vulnerability and writes the aligned
multiband stack + tidy parquet.

Implemented in Phase 1.
"""

from __future__ import annotations

from config import load_city


def main() -> None:
    cfg = load_city()
    raise SystemExit(
        f"[features] Phase 1 not yet implemented (city={cfg.key}). Scaffold only."
    )


if __name__ == "__main__":
    main()
