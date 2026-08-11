# -*- coding: utf-8 -*-
"""Shared trait-extractor registry — the common extension point of PhenoCV.

This module used to live at ``phenocv.phenotypes.base``. It was lifted into
``phenocv.core`` so that **every** PhenoCV module — segmentation, phenotypes,
and any future counting / disease / flowering module — registers its
compute tools against one registry and one 4-tier input-dependency model.
``phenocv.phenotypes.base`` now re-exports it for backwards compatibility.

Design
------
Every phenotype "tool" is a :class:`TraitExtractor` that:

* declares its **input dependencies** (``mask`` / ``rgb`` / ``depth`` /
  ``calibration`` / ``multispectral``) and a **tier** (1=mask-only,
  2=mask+rgb, 3=depth+calib, 4=multispectral);
* implements :meth:`TraitExtractor.extract`, returning a flat ``dict`` of
  named columns (``NaN`` for unobservable, never a fabricated value).

The orchestrator (``phenocv.phenotypes.compute_traits``) runs every registered
extractor whose ``requires`` are a subset of the inputs you actually have, in
tier order, and merges the result into one row. Adding a new tool is therefore
**just**: write one extractor, decorate it with ``@register`` — no engine
changes. This is the extension point that lets PhenoCV grow into a general
plant-phenotyping CV toolkit.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Set

# --- Input dependency tags (what an extractor may ask for) ----------------
INPUT_MASK = "mask"
INPUT_RGB = "rgb"
INPUT_DEPTH = "depth"
INPUT_CALIB = "calibration"
INPUT_MULTISPECTRAL = "multispectral"


class TraitExtractor(abc.ABC):
    """Base class for a single phenotype computation.

    Subclasses set class attributes ``name``, ``description``, ``requires``
    and ``tier``, then implement :meth:`extract`.
    """

    name: str = ""
    description: str = ""
    requires: List[str] = []
    tier: int = 1

    @abc.abstractmethod
    def extract(self, *, mask=None, rgb=None, depth=None, calibration=None,
                multispectral=None, **ctx) -> Dict[str, Any]:
        """Compute traits; return a flat column dict (use ``float('nan')``
        for unobservable outputs). ``**ctx`` carries optional config such as
        ``soil_plane``, ``fixed_ground``, ``mesh_config``."""
        raise NotImplementedError


_REGISTRY: Dict[str, TraitExtractor] = {}


def register(extractor: TraitExtractor) -> TraitExtractor:
    """Register an extractor under its ``name``.

    Accepts either an instance or a subclass (e.g. when used as a class
    decorator ``@register``); a subclass is instantiated automatically.
    """
    original = extractor
    if isinstance(extractor, type) and issubclass(extractor, TraitExtractor):
        instance = extractor()  # class decorator: instantiate for the registry
    elif isinstance(extractor, TraitExtractor):
        instance = extractor
    else:
        raise TypeError("register() expects a TraitExtractor instance or subclass")
    if not instance.name:
        raise ValueError("TraitExtractor.name must be set before registration")
    if instance.name in _REGISTRY:
        raise ValueError("Duplicate trait extractor name: %r" % instance.name)
    _REGISTRY[instance.name] = instance
    return original  # keep the class/instance name binding intact for callers


def get_extractor(name: str) -> TraitExtractor:
    return _REGISTRY[name]


def all_extractors() -> Dict[str, TraitExtractor]:
    return dict(_REGISTRY)


def available_for(inputs: Set[str]) -> List[TraitExtractor]:
    """Return registered extractors whose ``requires`` ⊆ ``inputs``."""
    have = set(inputs)
    out = [e for e in _REGISTRY.values() if set(e.requires).issubset(have)]
    return sorted(out, key=lambda e: (e.tier, e.name))


def clear_registry() -> None:  # pragma: no cover - test helper
    _REGISTRY.clear()
