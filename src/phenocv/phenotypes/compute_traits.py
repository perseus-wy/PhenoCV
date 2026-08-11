# -*- coding: utf-8 -*-
"""Tier-orchestrated phenotype computation.

``compute_traits`` runs every registered :class:`TraitExtractor` whose
``requires`` ⊆ the inputs you actually pass, in tier order (mask-only →
mask+RGB → depth+calib → multispectral), and merges the results into a single
flat column dictionary. It is **fail-closed**: a thrown extractor is recorded
under ``<name>_error`` rather than aborting the whole row, and unobservable
outputs are returned as ``NaN`` (+ ``missing_reason`` where the module
provides one).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import (
    INPUT_MASK, INPUT_RGB, INPUT_DEPTH, INPUT_CALIB, INPUT_MULTISPECTRAL,
    available_for,
)
# Importing the layer modules registers their extractors as a side effect.
from . import shape2d  # noqa: F401
from . import rgb_indices  # noqa: F401
from . import geometry3d  # noqa: F401
from . import multispectral  # noqa: F401


def compute_traits(*, mask=None, rgb=None, depth=None, calibration=None,
                   multispectral=None, **ctx) -> Dict[str, Any]:
    """Compute every applicable phenotype trait for one (plant, frame).

    Parameters
    ----------
    mask : np.ndarray
        Canopy/plant boolean (or 0/1) mask in image coordinates.
    rgb : np.ndarray, optional
        uint8 RGB image (``[H,W,3]``) for Tier-2 normalized-RGB indices.
    depth : np.ndarray, optional
        Depth image in **millimetres** (``[H,W]``) for Tier-3 geometry.
    calibration : str | pathlib.Path | CameraIntrinsics, optional
        Camera intrinsics source for Tier-3 (path to JSON, or a
        :class:`~phenocv.phenotypes.calib.CameraIntrinsics` instance).
    multispectral : dict[int, np.ndarray], optional
        Calibrated reflectance bands keyed by wavelength (555/660/720/840)
        for Tier-4 indices.
    **ctx : extra config
        Passed through to every extractor. Useful keys: ``soil_plane`` (3-vector
        for Tier-3), ``fixed_ground`` (:class:`FixedGroundFrame`), ``mesh_config``
        (:class:`CanopyMeshConfig`), ``intrinsics`` (pre-built
        :class:`CameraIntrinsics`), ``pot_circle`` / ``rgb_support`` (Tier-4 rim
        filter).

    Returns
    -------
    dict
        Merged trait columns + ``_inputs`` (sorted available input tags) and
        ``_extractors_run`` (names).
    """
    available = set()
    if mask is not None:
        available.add(INPUT_MASK)
    if rgb is not None:
        available.add(INPUT_RGB)
    if depth is not None:
        available.add(INPUT_DEPTH)
    if calibration is not None:
        available.add(INPUT_CALIB)
    if multispectral is not None:
        available.add(INPUT_MULTISPECTRAL)

    extractors = available_for(available)
    row: Dict[str, Any] = {}
    for ext in extractors:
        try:
            out = ext.extract(
                mask=mask, rgb=rgb, depth=depth, calibration=calibration,
                multispectral=multispectral, **ctx)
            row.update(out)
        except Exception as exc:  # fail-closed per-extractor
            row["%s_error" % ext.name] = str(exc)
    row["_inputs"] = sorted(available)
    row["_extractors_run"] = [e.name for e in extractors]
    return row
