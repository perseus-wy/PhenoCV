# -*- coding: utf-8 -*-
"""Modality / sensor adapter registry — the sensor-extension point of PhenoCV.

PhenoCV already ships readers for the four reference modalities (mask, RGB,
depth, multispectral) in :mod:`phenocv.core.io`. This module adds a *registry*
so that new sensor types — hyperspectral cubes, LiDAR point clouds, thermal,
fluorescence — can plug in alongside them without anyone editing the existing
readers.

Design (first principles, minimal)
----------------------------------
* A modality is anything that turns a path into a ``numpy`` array via
  :meth:`ModalityReader.read`. ``__init__`` never imports ``cv2``/``torch`` —
  readers do that lazily inside ``read``, so ``import phenocv.core`` stays
  pure-CPU.
* :func:`register_modality` records a reader under a stable name. Built-in
  readers are pre-registered; contributors register their own once.
* :func:`get_modality` / :func:`all_modalities` / :func:`available_modalities`
  are the lookup surface the rest of the toolkit uses.

Adding a new sensor (future contributor recipe)
-----------------------------------------------
::

    @register_modality
    class HyperspectralReader(ModalityReader):
        name = "hyperspectral"
        description = "Load an ENVI/numpy spectral cube."
        def read(self, path):
            import numpy as np
            return np.load(path)  # [H, W, bands]

    get_modality("hyperspectral").read("cube.npy")
"""

from __future__ import annotations

import abc
from typing import Callable, Dict, List


class ModalityReader(abc.ABC):
    """Abstract sensor reader: path -> numpy array. 抽象传感器读取器。

    Subclasses set ``name`` / ``description`` and implement :meth:`read`. The
    heavy IO imports (``cv2``/``torch``) belong inside :meth:`read` so importing
    this module never pulls in a GPU/plotting backend.
    """

    #: Stable modality identifier used by :func:`register_modality`.
    name: str = ""
    #: Human-readable description of what the reader loads.
    description: str = ""

    @abc.abstractmethod
    def read(self, path) -> "object":
        """Load the sensor artifact at ``path`` and return a numpy array."""
        raise NotImplementedError


# Built-in readers reuse the minimal IO helpers in ``phenocv.core.io``; those
# import cv2 lazily inside each function, so registration is import-safe.
from .io import (  # noqa: E402  (import after the ABC to avoid churn)
    read_gray_mask, read_rgb, read_depth_mm, read_ms_band,
)


class _IoModalityReader(ModalityReader):
    """Thin ModalityReader wrapping a core IO helper function."""

    def __init__(self, name: str, description: str,
                 fn: Callable) -> None:
        self.name = name
        self.description = description
        self._fn = fn

    def read(self, path):
        return self._fn(path)


class LidarReader(ModalityReader):
    """Example depth-like LiDAR reader (demonstrates how to plug in a sensor).

    Reuses the 16-bit-PNG / ``.npy`` depth loader; a real LiDAR backend would
    parse ``.ply``/``.las`` here instead.
    """

    name = "lidar"
    description = "LiDAR / depth-like point-cloud raster (mm, float32)."

    def read(self, path):
        return read_depth_mm(path)


_MODALITIES: Dict[str, ModalityReader] = {}


def register_modality(reader) -> "object":
    """Register a modality reader under its ``name``.

    注册一种模态读取器。 Accepts a :class:`ModalityReader` instance or subclass
    (class decorator form instantiates it automatically).
    """
    original = reader
    if isinstance(reader, type) and issubclass(reader, ModalityReader):
        instance = reader()  # class decorator: instantiate for the registry
    elif isinstance(reader, ModalityReader):
        instance = reader
    else:
        raise TypeError("register_modality expects a ModalityReader instance or subclass")
    if not instance.name:
        raise ValueError("ModalityReader.name must be set before registration")
    if instance.name in _MODALITIES:
        raise ValueError("Duplicate modality name: %r" % instance.name)
    _MODALITIES[instance.name] = instance
    return original  # keep the class/instance binding intact for callers


def get_modality(name: str) -> ModalityReader:
    """Return the registered reader for ``name``."""
    return _MODALITIES[name]


def all_modalities() -> Dict[str, ModalityReader]:
    """Return a copy of the full name -> reader mapping."""
    return dict(_MODALITIES)


def available_modalities() -> List[str]:
    """Return the sorted list of registered modality names."""
    return sorted(_MODALITIES)


# -- Pre-register the reference modalities (idempotent) ----------------------
for _name, _desc, _fn in (
    ("mask", "Binary canopy/object mask (bool [H,W]).", read_gray_mask),
    ("rgb", "RGB image (uint8 [H,W,3], R,G,B).", read_rgb),
    ("depth", "Depth image in millimetres (float32 [H,W]).", read_depth_mm),
    ("multispectral", "One multispectral band (float32 reflectance).", read_ms_band),
):
    if _name not in _MODALITIES:
        register_modality(_IoModalityReader(_name, _desc, _fn))

if "lidar" not in _MODALITIES:
    register_modality(LidarReader())
