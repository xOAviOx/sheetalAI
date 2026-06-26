"""Raw-layer export orchestrator (Phase 1).

Despite the name (kept to match the project layout), this is source-agnostic:
it asks the configured :class:`DataSource` to materialise the raw COGs for the
city, then records provenance. Select the source with ``DATA_SOURCE=synthetic``
(default, offline) or ``DATA_SOURCE=gee`` (real satellite, needs auth).

    DATA_SOURCE=synthetic uv run python gee_export.py
    CITY=ahmedabad DATA_SOURCE=gee uv run python gee_export.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from config import city_data_dir, load_city
from grid import build_grid
from sources import get_source


def main() -> None:
    cfg = load_city()
    source = get_source()
    data_dir = city_data_dir(cfg.key)
    raw_dir = data_dir / "raw"
    grid = build_grid(cfg)

    print(f"[export] city={cfg.key} source={source.name} grid={grid.width}x{grid.height}@{grid.res}m")
    result = source.export(cfg, raw_dir)

    manifest = {
        "city": cfg.key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "epsg": grid.epsg,
            "res_m": grid.res,
            "width": grid.width,
            "height": grid.height,
            "bounds_utm": list(grid.bounds),
        },
        "date_range": list(cfg.date_range),
        "layers": result.as_dict(),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export] wrote {len(result.as_dict()) - 2} raw COGs to {raw_dir}")
    print(f"[export] provenance: {result.note}")


if __name__ == "__main__":
    main()
