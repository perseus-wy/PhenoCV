# -*- coding: utf-8 -*-
"""Backwards-compatible re-export of the shared trait-extractor registry.

The registry used to live here; it now lives in :mod:`phenocv.core.registry`
so segmentation, counting, and other modules share one extension point and one
4-tier input-dependency model. This module re-exports everything so existing
``from phenocv.phenotypes.base import TraitExtractor`` imports keep working.

See :mod:`phenocv.core.registry` for the full design notes.
"""

from phenocv.core.registry import (
    TraitExtractor, INPUT_MASK, INPUT_RGB, INPUT_DEPTH, INPUT_CALIB,
    INPUT_MULTISPECTRAL, register, get_extractor, all_extractors,
    available_for, clear_registry,
)

__all__ = [
    "TraitExtractor",
    "INPUT_MASK", "INPUT_RGB", "INPUT_DEPTH", "INPUT_CALIB", "INPUT_MULTISPECTRAL",
    "register", "get_extractor", "all_extractors", "available_for", "clear_registry",
]
