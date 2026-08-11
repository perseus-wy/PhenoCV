"""Data adapters for PhenoCV.

An adapter translates a user's dataset layout into the
:class:`~phenocv.engine.PlantSequence` objects the core engine consumes
(sequences + anchor masks).

* :class:`CsvManifestAdapter` — the default, data-source-agnostic path: write
  one manifest CSV/JSON describing your sequences and sparse anchor masks.
* :class:`PlantPhenotypingAdapter` — a worked example for potted soybean
  temporal data (frame-index CSV + per-plant manual mask directories).
* :class:`BaseAdapter` — subclass this to support a new dataset format.
"""

from .base import BaseAdapter
from .csv_manifest import CsvManifestAdapter
from .plant_phenotyping import PlantPhenotypingAdapter

__all__ = ["BaseAdapter", "CsvManifestAdapter", "PlantPhenotypingAdapter"]
