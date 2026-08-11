"""Minimal, runnable reference for a pluggable, fail-closed trait engine.

Pure numpy — no domain code. Demonstrates the four-tier input model,
`@register` extractor, `available_for()` selection, and per-extractor
fail-closed orchestration. Adapt the `extract()` bodies to your pipeline.

Run:  python references/registry_pattern.py
"""
from __future__ import annotations

from typing import Dict, List

# --- input tags -----------------------------------------------------------
INPUT_MASK = "mask"
INPUT_RGB = "rgb"
INPUT_DEPTH = "depth"
INPUT_CALIB = "calibration"
INPUT_MULTISPECTRAL = "multispectral"


class TraitExtractor:
    name: str = ""
    requires: List[str] = []
    tier: int = 0
    description: str = ""

    def extract(self, *, mask=None, rgb=None, depth=None,
                calibration=None, multispectral=None, **ctx) -> Dict:
        raise NotImplementedError


_REGISTRY: Dict[str, TraitExtractor] = {}


def register(extractor):
    """Class or instance decorator. Stores an *instance* so `extract()` is bound."""
    original = extractor
    if isinstance(extractor, type) and issubclass(extractor, TraitExtractor):
        instance = extractor()
    elif isinstance(extractor, TraitExtractor):
        instance = extractor
    else:
        raise TypeError("register() expects a TraitExtractor instance or subclass")
    if not instance.name:
        raise ValueError("TraitExtractor.name must be set")
    if instance.name in _REGISTRY:
        raise ValueError("duplicate extractor name: %s" % instance.name)
    _REGISTRY[instance.name] = instance
    return original


def available_for(available):
    """Return registered extractors whose `requires` ⊆ `available`, by tier."""
    out = [e for e in _REGISTRY.values() if set(e.requires) <= set(available)]
    return sorted(out, key=lambda e: e.tier)


def compute_traits(*, mask=None, rgb=None, depth=None,
                   calibration=None, multispectral=None, **ctx) -> Dict:
    """Run every applicable extractor and merge into one fail-closed row."""
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

    row: Dict = {}
    for ext in available_for(available):
        try:
            row.update(ext.extract(mask=mask, rgb=rgb, depth=depth,
                                   calibration=calibration,
                                   multispectral=multispectral, **ctx))
        except Exception as exc:                      # per-extractor fail-closed
            row["%s_error" % ext.name] = str(exc)
    row["_inputs"] = sorted(available)
    row["_extractors_run"] = [e.name for e in available_for(available)]
    return row


# --- example extractors (replace with real math) --------------------------
@register
class Shape2D(TraitExtractor):
    name = "shape2d"
    requires = [INPUT_MASK]
    tier = 1

    def extract(self, *, mask=None, **ctx):
        if not mask.any():
            return {"area_px": float("nan"), "missing_reason": "empty_mask"}
        return {"area_px": int(mask.sum())}


@register
class RGBIndices(TraitExtractor):
    name = "rgb_vegetation_indices"
    requires = [INPUT_MASK, INPUT_RGB]
    tier = 2

    def extract(self, *, mask=None, rgb=None, **ctx):
        # sketch only: compute a normalized-RGB index over the masked pixels
        px = rgb[mask]
        r, g, b = px[:, 0].astype(float), px[:, 1].astype(float), px[:, 2].astype(float)
        denom = (r + g + b)
        exg = (2 * g - r - b) / denom
        return {"ExG_mean": float(exg.mean())}


if __name__ == "__main__":
    import numpy as np
    mask = np.zeros((8, 8), bool); mask[2:6, 2:6] = True
    rgb = np.zeros((8, 8, 3), np.uint8); rgb[mask] = (30, 120, 20)
    row = compute_traits(mask=mask, rgb=rgb)
    print("L1+L2:", row)

    # missing RGB -> L2 skipped, no fabricated values
    row2 = compute_traits(mask=mask)
    print("L1 only:", row2)
    assert "rgb_vegetation_indices" not in row2.get("_extractors_run", [])
    print("OK: fail-closed, no fabricated L2 columns")
