"""Swappable data-source interface.

Hard constraint #2: GEE must be replaceable by direct USGS/Copernicus/ISRO
downloads *without touching downstream code*. Downstream code therefore only
ever talks to this interface — it asks a source to materialise a fixed set of
raw COGs on disk, and never imports ``ee`` or any source-specific library.

Each source writes these COGs into ``data/{city}/raw/`` on the city's native
projection; ``features.py`` reprojects them onto the canonical grid.

Raw layer contract (filename -> meaning):
  lst.tif         1 band   land-surface temperature, °C
  sr.tif          6 bands  surface reflectance: blue,green,red,nir,swir1,swir2 (0-1)
  worldcover.tif  1 band   ESA WorldCover class codes (10..100)
  pop.tif         1 band   population count per native pixel
  dem.tif         1 band   elevation, metres
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from config import CityConfig

# Canonical surface-reflectance band order used everywhere downstream.
SR_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")

# ESA WorldCover class codes of interest.
WORLDCOVER_BUILTUP = 50
WORLDCOVER_WATER = 80
WORLDCOVER_TREES = 10


@dataclass(frozen=True)
class SourceResult:
    """Paths to the raw COGs a source produced, plus provenance."""

    source_name: str
    lst: Path
    sr: Path
    worldcover: Path
    pop: Path
    dem: Path
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source_name,
            "lst": str(self.lst),
            "sr": str(self.sr),
            "worldcover": str(self.worldcover),
            "pop": str(self.pop),
            "dem": str(self.dem),
            "note": self.note,
        }


class DataSource(ABC):
    """Materialise raw analysis layers for a city as COGs on disk."""

    name: str = "base"

    @abstractmethod
    def export(self, cfg: CityConfig, raw_dir: Path) -> SourceResult:
        """Write raw COGs into ``raw_dir`` and return their paths."""
        raise NotImplementedError
