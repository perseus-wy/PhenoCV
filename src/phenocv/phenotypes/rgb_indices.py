# -*- coding: utf-8 -*-
"""Tier-2 (mask + RGB) normalized-RGB vegetation indices.

Ports ``pheno_extract.reference_indices`` (source: ima knowledge base
"颜色校正" note). All 16 named indices + 10 experience features. Division is
**sign-preserving** (``signed_safe_divide``): VARI/WI have legitimately negative
denominators, so we must NOT clamp the denominator to +epsilon — instead we
mark singular pixels NaN and report a valid-fraction column.

Typical use: compute the index map once, then aggregate statistics over the
canopy mask (mean / median / std / p10 / p90) so a whole-plant frame yields a
few robust scalars.
"""

from __future__ import annotations

import numpy as np

from .base import TraitExtractor, register, INPUT_MASK, INPUT_RGB

DIVISION_EPSILON = 1.0e-6

REFERENCE_INDEX_NAMES = (
    "NDI", "CIVE", "MNDVI", "GRVI", "IKAW", "ExG", "ExR", "ExGR",
    "VARI", "GLA", "GLI", "RGBVI", "MGRVI", "WI", "PSRI", "ARI",
)
EXPERIENCE_FEATURE_NAMES = (
    "exp_r", "exp_g", "exp_b", "exp_r_over_g", "exp_g_over_b",
    "exp_b_over_g", "exp_r_plus_g", "exp_r_plus_b", "exp_g_plus_b",
    "exp_gmb_over_gpb",
)


def signed_safe_divide(numerator, denominator, epsilon: float = DIVISION_EPSILON):
    """Divide preserving denominator sign; singular pixels -> NaN.

    ``np.maximum(denominator, eps)`` would wrongly rewrite valid negative
    denominators (breaks VARI, WI). We keep the sign and NaN the singular spots;
    downstream stats ignore NaNs and report a valid fraction.
    """
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape),
                  np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > epsilon)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


# --- 16 named indices (r,g,b are normalized-RGB in [0,1]) -------------------
def ndi(r, g, b):      return (r - g) / (r + g + 0.01)
def cive(r, g, b):     return 0.441 * r - 0.811 * g + 0.3856 * b + 18.78745
def mndvi(r, g, b):    return signed_safe_divide(r - g - b, r + g)
def grvi(r, g, b):     return signed_safe_divide(g - r, g + r)
def ikaw(r, g, b):     return signed_safe_divide(r - b, r + b)
def exg(r, g, b):      return 2.0 * g - r - b
def exr(r, g, b):      return 1.4 * r - g
def exgr(r, g, b):     return 3.0 * g - 2.4 * r - b
def vari(r, g, b):     return signed_safe_divide(g - r, g + r - b)
def gla(r, g, b):      return signed_safe_divide(2.0 * g - r + b, 2.0 * g + r + b)
def gli(r, g, b):      return signed_safe_divide(2.0 * g - r - b, 2.0 * g + r + b)
def rgbvi(r, g, b):
    g2, br = g * g, b * r
    return signed_safe_divide(g2 - br, g2 + br)
def mgrvi(r, g, b):
    g2, r2 = g * g, r * r
    return signed_safe_divide(g2 - r2, g2 + r2)
def wi(r, g, b):       return signed_safe_divide(g - b, r - g)
def psri(r, g, b):     return signed_safe_divide(r - b, r)
def ari(r, g, b):      return signed_safe_divide(np.ones_like(r), 1.0 - r)


def experience_features(r, g, b) -> dict:
    return {
        "exp_r": r, "exp_g": g, "exp_b": b,
        "exp_r_over_g": signed_safe_divide(r, g),
        "exp_g_over_b": signed_safe_divide(g, b),
        "exp_b_over_g": signed_safe_divide(b, g),
        "exp_r_plus_g": r + g, "exp_r_plus_b": r + b, "exp_g_plus_b": g + b,
        "exp_gmb_over_gpb": signed_safe_divide(g - b, g + b),
    }


def _normalized_rgb(rgb: np.ndarray):
    """uint8 RGB -> normalized r,g,b (sum=1), shape [H,W] each."""
    values = rgb.astype(np.float32) / 255.0
    total = values.sum(axis=-1)
    r = signed_safe_divide(values[..., 0], total)
    g = signed_safe_divide(values[..., 1], total)
    b = signed_safe_divide(values[..., 2], total)
    return r, g, b


def compute_reference_index_images(rgb: np.ndarray) -> dict:
    """All 16 index maps over an RGB uint8 image [H,W,3]."""
    r, g, b = _normalized_rgb(rgb)
    return {
        "NDI": ndi(r, g, b), "CIVE": cive(r, g, b), "MNDVI": mndvi(r, g, b),
        "GRVI": grvi(r, g, b), "IKAW": ikaw(r, g, b), "ExG": exg(r, g, b),
        "ExR": exr(r, g, b), "ExGR": exgr(r, g, b), "VARI": vari(r, g, b),
        "GLA": gla(r, g, b), "GLI": gli(r, g, b), "RGBVI": rgbvi(r, g, b),
        "MGRVI": mgrvi(r, g, b), "WI": wi(r, g, b), "PSRI": psri(r, g, b),
        "ARI": ari(r, g, b),
    }


def compute_reference_feature_images(rgb: np.ndarray) -> dict:
    """16 index maps + 10 experience-feature maps."""
    out = compute_reference_index_images(rgb)
    r, g, b = _normalized_rgb(rgb)
    out.update(experience_features(r, g, b))
    return out


def summarize_indices_over_mask(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Per-index robust statistics over the canopy mask.

    Returns ``<IDX>_mean / _median / _std / _p10 / _p90`` for each of the 16
    named indices, plus ``<IDX>_valid_frac`` (fraction of mask pixels with a
    finite index value). Empty mask -> all NaN.
    """
    images = compute_reference_index_images(rgb)
    m = (np.asarray(mask) > 0)
    if not m.any():
        cols = {}
        for name in REFERENCE_INDEX_NAMES:
            for stat in ("mean", "median", "std", "p10", "p90", "valid_frac"):
                cols["%s_%s" % (name, stat)] = np.nan
        return cols

    cols = {}
    for name in REFERENCE_INDEX_NAMES:
        vals = images[name][m]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            cols["%s_mean" % name] = np.nan
            cols["%s_median" % name] = np.nan
            cols["%s_std" % name] = np.nan
            cols["%s_p10" % name] = np.nan
            cols["%s_p90" % name] = np.nan
            cols["%s_valid_frac" % name] = 0.0
            continue
        cols["%s_mean" % name] = float(np.mean(finite))
        cols["%s_median" % name] = float(np.median(finite))
        cols["%s_std" % name] = float(np.std(finite))
        cols["%s_p10" % name] = float(np.percentile(finite, 10))
        cols["%s_p90" % name] = float(np.percentile(finite, 90))
        cols["%s_valid_frac" % name] = float(finite.size / m.sum())
    return cols


@register
class RgbVegetationIndexExtractor(TraitExtractor):
    name = "rgb_vegetation_indices"
    description = "Normalized-RGB vegetation indices (ExG/ExR/ExGR/GLI/VARI/...) aggregated over the mask."
    requires = [INPUT_MASK, INPUT_RGB]
    tier = 2

    def extract(self, *, mask=None, rgb=None, **ctx):
        return summarize_indices_over_mask(rgb, mask)
