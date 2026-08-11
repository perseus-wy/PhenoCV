# -*- coding: utf-8 -*-
"""PhenoCV phenotype computation — a pluggable, 4-tier plant-trait toolkit.

Four tiers of trait extractors, each declaring its input dependencies and a
tier number. ``compute_traits`` runs every extractor whose requirements you
satisfy and merges the columns:

* Tier 1 — mask-only 2D shape (area, bbox, centroid, solidity, ...)
* Tier 2 — mask + RGB normalized-RGB vegetation indices (ExG/ExR/VARI/...)
* Tier 3 — mask + depth(mm) + intrinsics → plant height / area / volume (mm)
* Tier 4 — mask + multispectral reflectance → 12 indices + reflectance stats

Extending PhenoCV
-----------------
Write a :class:`~phenocv.phenotypes.base.TraitExtractor` subclass, set its
``name`` / ``description`` / ``requires`` / ``tier``, implement ``extract``
(returning a flat column dict; use ``float('nan')`` for unobservable values),
and decorate it with ``@register``. Import the module once (this package does
that for you) — no engine change required.
"""

from __future__ import annotations

from .base import (
    TraitExtractor, register, get_extractor, all_extractors,
    available_for, clear_registry,
    INPUT_MASK, INPUT_RGB, INPUT_DEPTH, INPUT_CALIB, INPUT_MULTISPECTRAL,
)
from .calib import CameraIntrinsics, load_rgb_intrinsics
from .shape2d import compute_2d_traits, Shape2DExtractor
from .rgb_indices import (
    compute_reference_index_images, summarize_indices_over_mask,
    RgbVegetationIndexExtractor,
)
from .geometry3d import (
    deproject, fit_soil_plane, soil_candidate_points,
    compute_canopy_geometry, compute_plant_height,
    FixedGroundFrame, CanopyMeshConfig, CanopyGeometryTraits,
    CanopyGeometryExtractor,
)
from .multispectral import (
    compute_index_images, summarize_reflectance,
    empirical_line_gains, apply_gains, remove_pot_rim_false_positive,
    compute_multispectral_traits, MultispectralExtractor,
    BANDS, INDEX_NAMES, STATS, PANEL_REFLECTANCE,
)
from .compute_traits import compute_traits

__all__ = [
    "TraitExtractor", "register", "get_extractor", "all_extractors",
    "available_for", "clear_registry",
    "INPUT_MASK", "INPUT_RGB", "INPUT_DEPTH", "INPUT_CALIB", "INPUT_MULTISPECTRAL",
    "CameraIntrinsics", "load_rgb_intrinsics",
    "compute_2d_traits", "Shape2DExtractor",
    "compute_reference_index_images", "summarize_indices_over_mask",
    "RgbVegetationIndexExtractor",
    "deproject", "fit_soil_plane", "soil_candidate_points",
    "compute_canopy_geometry", "compute_plant_height",
    "FixedGroundFrame", "CanopyMeshConfig", "CanopyGeometryTraits",
    "CanopyGeometryExtractor",
    "compute_index_images", "summarize_reflectance",
    "empirical_line_gains", "apply_gains", "remove_pot_rim_false_positive",
    "compute_multispectral_traits", "MultispectralExtractor",
    "BANDS", "INDEX_NAMES", "STATS", "PANEL_REFLECTANCE",
    "compute_traits",
]
