# -*- coding: utf-8 -*-
"""PhenoCV shared core: the trait-extractor registry and IO helpers.

Every module (segmentation, phenotypes, future counting / disease) builds on
this common base so the toolkit stays composable and the extension points are
shared. Agents and downstream code should prefer ``import phenocv.core`` over
reaching into a specific module's internals.
"""

from .registry import (
    TraitExtractor, INPUT_MASK, INPUT_RGB, INPUT_DEPTH, INPUT_CALIB,
    INPUT_MULTISPECTRAL, register, get_extractor, all_extractors,
    available_for, clear_registry,
)
from .io import (
    read_gray_mask, read_rgb, read_depth_mm, read_ms_band, first_match,
)
from .modalities import (
    ModalityReader, register_modality, get_modality, all_modalities,
    available_modalities,
)

__all__ = [
    "TraitExtractor",
    "INPUT_MASK", "INPUT_RGB", "INPUT_DEPTH", "INPUT_CALIB", "INPUT_MULTISPECTRAL",
    "register", "get_extractor", "all_extractors", "available_for", "clear_registry",
    "read_gray_mask", "read_rgb", "read_depth_mm", "read_ms_band", "first_match",
    "ModalityReader", "register_modality", "get_modality", "all_modalities",
    "available_modalities",
]
