"""Source factory — selects the data source from config/env.

Set ``DATA_SOURCE`` in the environment (or pass ``name``):
  - ``synthetic`` (default): offline deterministic city, no credentials needed.
  - ``gee``: real Google Earth Engine export (needs ``earthengine authenticate``).

Downstream code only ever calls :func:`get_source` and the :class:`DataSource`
interface, never a concrete source — that is the swap point for a future
USGS/Copernicus/ISRO implementation.
"""

from __future__ import annotations

import os

from sources.base import DataSource


def get_source(name: str | None = None) -> DataSource:
    key = (name or os.environ.get("DATA_SOURCE") or "synthetic").strip().lower()
    if key == "synthetic":
        from sources.synthetic import SyntheticSource

        return SyntheticSource()
    if key in {"gee", "earthengine"}:
        from sources.gee import GEESource

        return GEESource()
    raise ValueError(f"Unknown DATA_SOURCE '{key}'. Use 'synthetic' or 'gee'.")


__all__ = ["get_source", "DataSource"]
