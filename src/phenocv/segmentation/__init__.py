# -*- coding: utf-8 -*-
"""Temporal canopy segmentation module (SAM 2 video propagation).

This is one of PhenoCV's composable modules. It ships the data-source-agnostic
engine and pluggable adapters that turn a dataset layout into per-plant,
per-frame canopy masks with temporal consistency.

Re-exports
----------
* :func:`load_config` — load propagation config + presets
* :class:`PlantSequence`, :class:`TemporalPropagationConfig`,
  :func:`run_sam2_video_temporal` — the engine API
* :class:`BaseAdapter`, :class:`CsvManifestAdapter`,
  :class:`PlantPhenotypingAdapter` — dataset adapters
"""

from .config import load_config
from .engine import (
    PlantSequence, TemporalPropagationConfig, run_sam2_video_temporal,
)
from .adapters import (
    BaseAdapter, CsvManifestAdapter, PlantPhenotypingAdapter,
)

__all__ = [
    "load_config",
    "PlantSequence", "TemporalPropagationConfig", "run_sam2_video_temporal",
    "BaseAdapter", "CsvManifestAdapter", "PlantPhenotypingAdapter",
]
