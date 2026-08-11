# -*- coding: utf-8 -*-
"""Shared image / mask IO helpers for PhenoCV modules.

These readers are intentionally minimal (``cv2`` + ``numpy`` only) so that any
module — segmentation masks, phenotype rasters, multispectral bands — can load
the same on-disk artifacts without re-implementing ``cvtColor`` / ``.npy`` /
16-bit normalization. All functions accept ``str`` or :class:`pathlib.Path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def read_gray_mask(path) -> np.ndarray:
    """Load a mask image as a boolean ``[H,W]`` array (``>0`` = foreground)."""
    import cv2
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("cannot read mask: %s" % p)
    return img > 0


def read_rgb(path) -> np.ndarray:
    """Load an RGB image (``[H,W,3]`` uint8, channel order R,G,B)."""
    import cv2
    p = Path(path)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("cannot read rgb: %s" % p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_mm(path) -> np.ndarray:
    """Load a depth image in **millimetres** as ``[H,W]`` float32.

    Accepts a ``.npy`` array or a 16-bit PNG (``IMREAD_UNCHANGED``).
    """
    import cv2
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(str(p)).astype(np.float32)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("cannot read depth: %s" % p)
    return img.astype(np.float32)


def read_ms_band(path) -> np.ndarray:
    """Load one multispectral band as float32 reflectance (~unit interval).

    If the source is 16-bit (>1), normalize by 65535. Calibration against a
    reference panel is the caller's responsibility (see ``empirical_line_gains``).
    """
    import cv2
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p)).astype(np.float32)
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError("cannot read ms band: %s" % p)
        arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:
        arr = arr / 65535.0
    return arr


def first_match(directory, stem) -> Optional[Path]:
    """Return the first existing ``<directory>/<stem>.<ext>`` or ``None``.

    Tries ``.png .jpg .jpeg .tif .tiff .npy`` in order.
    """
    if not directory:
        return None
    d = Path(directory)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"):
        p = d / (stem + ext)
        if p.exists():
            return p
    return None
