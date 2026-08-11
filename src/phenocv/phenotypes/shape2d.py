# -*- coding: utf-8 -*-
"""Tier-1 (mask-only) 2D canopy shape descriptors.

Pure ``cv2`` + ``numpy``; no depth, no RGB, no calibration. Safe to run the
moment a mask exists (e.g. straight out of segmentation). Mirrors
``pheno_extract.mask_phenotypes.compute_2d_traits``.
"""

from __future__ import annotations

import numpy as np
import cv2

from .base import TraitExtractor, register, INPUT_MASK


def compute_2d_traits(mask: np.ndarray) -> dict:
    """2D shape descriptors from a boolean/0-1 mask (pixel units, image coords).

    Returns area, bbox, centroid, convex-hull area, perimeter, solidity
    (=area/hull), circularity (=4πA/P²), aspect_ratio (=w/h). Empty mask -> NaNs.
    """
    m = (np.asarray(mask) > 0)
    area_px = int(m.sum())
    out = {
        "area_px": area_px,
        "bbox_x": np.nan, "bbox_y": np.nan, "bbox_w": np.nan, "bbox_h": np.nan,
        "centroid_x": np.nan, "centroid_y": np.nan,
        "convex_hull_area_px": np.nan, "perimeter_px": np.nan,
        "solidity": np.nan, "circularity": np.nan, "aspect_ratio": np.nan,
    }
    if area_px == 0:
        return out

    ys, xs = np.where(m)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    out["bbox_x"] = x0
    out["bbox_y"] = y0
    out["bbox_w"] = x1 - x0 + 1
    out["bbox_h"] = y1 - y0 + 1
    out["centroid_x"] = float(xs.mean())
    out["centroid_y"] = float(ys.mean())

    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return out
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    out["convex_hull_area_px"] = float(cv2.contourArea(hull))
    out["perimeter_px"] = float(cv2.arcLength(cnt, True))
    if out["perimeter_px"] > 0:
        out["circularity"] = float(4.0 * np.pi * area_px) / (out["perimeter_px"] ** 2)
    if out["convex_hull_area_px"] > 0:
        out["solidity"] = float(area_px) / out["convex_hull_area_px"]
    if out["bbox_h"] > 0:
        out["aspect_ratio"] = float(out["bbox_w"]) / out["bbox_h"]
    return out


@register
class Shape2DExtractor(TraitExtractor):
    name = "shape2d"
    description = "2D mask shape descriptors (area, bbox, centroid, solidity, circularity, aspect)."
    requires = [INPUT_MASK]
    tier = 1

    def extract(self, *, mask=None, **ctx):
        return compute_2d_traits(mask)
