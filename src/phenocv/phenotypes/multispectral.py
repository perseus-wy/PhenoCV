# -*- coding: utf-8 -*-
"""Tier-4 (mask + multispectral reflectance) vegetation indices.

Ports ``pheno_extract.multispectral``.

Input contract
--------------
The ``multispectral`` input is a ``dict[int, np.ndarray]`` of **calibrated
reflectance** arrays (≈ unit interval), keyed by wavelength in nm::

    {555: green, 660: red, 720: red_edge, 840: nir}

Every band must be the same ``[H, W]`` shape and live in the **mask's
coordinate frame** (the 555 nm / "green" frame — band-to-green registration
is upstream data-prep and intentionally out of scope here).

Calibration
-----------
Radiometric calibration (empirical-line against the reference panel) is
provided as a helper: :func:`empirical_line_gains` turns per-band panel
medians into gains, :func:`apply_gains` turns raw signals into reflectance.
The heavier upstream steps (panel detection, vignetting correction, ECC
band registration) live in the data-ingest pipeline, not this trait module.
"""

from __future__ import annotations

import numpy as np
import cv2

from .base import TraitExtractor, register, INPUT_MASK, INPUT_MULTISPECTRAL

# --- Band layout (MS400: green / red / red-edge / NIR) ---------------------
BANDS = (555, 660, 720, 840)
GREEN, RED, RED_EDGE, NIR = BANDS

INDEX_NAMES = (
    "NDVI", "NDRE", "GNDVI", "SAVI", "OSAVI", "RVI", "DVI",
    "CIgreen", "CIrededge", "MTCI", "MCARI", "TCARI",
)

STATS = ("mean", "median", "std", "p10", "p90")

PANEL_SERIAL = "CA320233044"
PANEL_REFLECTANCE = {555: 0.61, 660: 0.60, 720: 0.61, 840: 0.60}

# --- Pot-rim false-positive filter constants (faithful port) ---------------
POT_RIM_PROTECTED_CORE_RATIO = 0.68
POT_RIM_MINIMUM_RADIAL_MEDIAN = 0.72
POT_RIM_MAXIMUM_RGB_SUPPORT = 0.25
POT_RIM_NDVI_SUPPORT_THRESHOLD = 0.25
POT_RIM_MAXIMUM_NDVI_SUPPORT = 0.55
POT_RIM_EVIDENCE_NDVI_THRESHOLD = 0.15
POT_RIM_MINIMUM_COMPONENT_AREA = 200
POT_RIM_MINIMUM_COMPONENT_FRACTION = 0.008
POT_RIM_EVIDENCE_DILATION_PX = 11


def _safe_divide(numerator, denominator):
    """Numerator/denominator with NaN where the denominator is singular."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape),
                     np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    np.divide(numerator, denominator, out=output, where=valid)
    return output


def compute_index_images(bands: dict) -> dict:
    """12 vegetation-index maps from calibrated reflectance bands.

    ``bands`` must contain all of :data:`BANDS`. Mirrors
    ``pheno_extract.multispectral.compute_index_images`` exactly.
    """
    green, red, red_edge, nir = (np.asarray(bands[band], dtype=np.float32) for band in BANDS)
    ndvi = _safe_divide(nir - red, nir + red)
    return {
        "NDVI": ndvi,
        "NDRE": _safe_divide(nir - red_edge, nir + red_edge),
        "GNDVI": _safe_divide(nir - green, nir + green),
        "SAVI": 1.5 * _safe_divide(nir - red, nir + red + 0.5),
        "OSAVI": 1.16 * _safe_divide(nir - red, nir + red + 0.16),
        "RVI": _safe_divide(nir, red),
        "DVI": nir - red,
        "CIgreen": _safe_divide(nir, green) - 1.0,
        "CIrededge": _safe_divide(nir, red_edge) - 1.0,
        "MTCI": _safe_divide(nir - red_edge, red_edge - red),
        "MCARI": (red_edge - red - 0.2 * (red_edge - green)) * _safe_divide(red_edge, red),
        "TCARI": 3.0 * (
            red_edge - red - 0.2 * (red_edge - green) * _safe_divide(red_edge, red)
        ),
    }


def summarize_indices_over_mask(indices: dict, mask: np.ndarray) -> dict:
    """Per-index robust statistics over the canopy mask.

    Returns ``<IDX>_mean / _median / _std / _p10 / _p90 / _valid_frac`` for
    each of the 12 indices. Empty mask -> NaN.
    """
    m = (np.asarray(mask) > 0)
    cols = {}
    for name in INDEX_NAMES:
        vals = indices[name][m]
        finite = vals[np.isfinite(vals)]
        cols["%s_valid_frac" % name] = float(len(finite) / max(vals.size, 1)) if vals.size else np.nan
        if finite.size == 0:
            for stat in STATS:
                cols["%s_%s" % (name, stat)] = np.nan
            continue
        cols["%s_mean" % name] = float(np.mean(finite))
        cols["%s_median" % name] = float(np.median(finite))
        cols["%s_std" % name] = float(np.std(finite))
        cols["%s_p10" % name] = float(np.percentile(finite, 10))
        cols["%s_p90" % name] = float(np.percentile(finite, 90))
    return cols


def summarize_reflectance(bands: dict, mask: np.ndarray) -> dict:
    """Per-band reflectance statistics over the canopy mask.

    Returns ``R<band>_mean / _median / _std / _p10 / _p90 / _valid_fraction``
    for each of the 4 bands.
    """
    m = (np.asarray(mask) > 0)
    row = {}
    for wavelength in BANDS:
        selected = np.asarray(bands[wavelength], dtype=np.float32)[m]
        finite = selected[np.isfinite(selected)]
        row["R%d_valid_fraction" % wavelength] = (
            float(len(finite) / max(selected.size, 1)) if selected.size else np.nan
        )
        if finite.size == 0:
            for stat in STATS:
                row["R%d_%s" % (wavelength, stat)] = np.nan
            continue
        row["R%d_mean" % wavelength] = float(np.mean(finite))
        row["R%d_median" % wavelength] = float(np.median(finite))
        row["R%d_std" % wavelength] = float(np.std(finite))
        row["R%d_p10" % wavelength] = float(np.percentile(finite, 10))
        row["R%d_p90" % wavelength] = float(np.percentile(finite, 90))
    return row


def empirical_line_gains(panel_medians: dict, panel_reflectance: dict = PANEL_REFLECTANCE) -> dict:
    """Empirical-line calibration gains from per-band panel medians.

    ``gain[band] = PANEL_REFLECTANCE[band] / panel_median[band]`` — the exact
    rule used in ``pheno_extract.multispectral._date_calibration``.

    ``panel_medians`` maps each band in :data:`BANDS` to the median corrected
    signal over the reference panel region (a robust, finite, positive value).
    """
    gains = {}
    for band in BANDS:
        med = float(panel_medians[band])
        if not np.isfinite(med) or med <= 0:
            raise ValueError("panel_median_not_positive_for_band_%d" % band)
        gains[band] = float(panel_reflectance[band] / med)
    return gains


def apply_gains(signal_bands: dict, gains: dict) -> dict:
    """Multiply raw per-band signals by the empirical-line gains -> reflectance."""
    return {band: np.asarray(signal_bands[band], dtype=np.float32) * gains[band] for band in BANDS}


def remove_pot_rim_false_positive(mask, rgb_support, ndvi, circle):
    """Strip only significant pot-rim arcs lacking RGB + spectral support.

    Faithful port of ``pheno_extract.multispectral._remove_pot_rim_false_positive``.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Candidate plant mask in the 555 nm / green frame.
    rgb_support : np.ndarray
        Boolean/0-1 RGB-vegetation support map (same frame).
    ndvi : np.ndarray
        NDVI map (same frame), used as a *diagnostic* (never a decision gate).
    circle : (cx, cy, radius)
        Pot circle in pixel coordinates.

    Returns
    -------
    (cleaned_mask, audit) : (np.ndarray bool, dict)
        ``cleaned_mask`` removes only suspicious outer arcs; ``audit`` records
        how many components were inspected / removed.
    """
    if mask.shape != rgb_support.shape or mask.shape != ndvi.shape:
        raise ValueError("pot_rim_filter_shape_mismatch")
    cx, cy, radius = circle
    if radius <= 0:
        raise ValueError("invalid_pot_radius_for_rim_filter")

    yy, xx = np.indices(mask.shape, dtype=np.float32)
    normalized_radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    protected_core = normalized_radius < POT_RIM_PROTECTED_CORE_RATIO
    outer_mask = mask & ~protected_core
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        outer_mask.astype(np.uint8), connectivity=8
    )
    cleaned = mask.copy()
    removed_pixels = 0
    preserved_evidence_pixels = 0
    suspicious_components = 0
    minimum_component_area = max(
        POT_RIM_MINIMUM_COMPONENT_AREA,
        int(round(float(mask.sum()) * POT_RIM_MINIMUM_COMPONENT_FRACTION)),
    )
    evidence = (
        np.asarray(rgb_support, dtype=bool)
        & np.isfinite(ndvi)
        & (ndvi >= POT_RIM_EVIDENCE_NDVI_THRESHOLD)
    )
    evidence_kernel = np.ones(
        (POT_RIM_EVIDENCE_DILATION_PX, POT_RIM_EVIDENCE_DILATION_PX), dtype=np.uint8
    )
    core_mask = mask & protected_core
    adjacency_kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        component = labels == label
        radii = normalized_radius[component]
        radial_median = float(np.median(radii))
        rgb_fraction = float(np.mean(rgb_support[component]))
        ndvi_fraction = float(
            np.mean(
                np.isfinite(ndvi[component])
                & (ndvi[component] >= POT_RIM_NDVI_SUPPORT_THRESHOLD)
            )
        )
        suspicious = (
            radial_median >= POT_RIM_MINIMUM_RADIAL_MEDIAN
            and rgb_fraction < POT_RIM_MAXIMUM_RGB_SUPPORT
            and ndvi_fraction < POT_RIM_MAXIMUM_NDVI_SUPPORT
        )
        if not suspicious:
            continue
        suspicious_components += 1
        touches_core = bool(
            np.any(
                cv2.dilate(
                    component.astype(np.uint8),
                    adjacency_kernel,
                    iterations=1,
                ).astype(bool)
                & core_mask
            )
        )
        if touches_core:
            evidence_seed = (component & evidence).astype(np.uint8)
            preserve = (
                cv2.dilate(
                    evidence_seed,
                    evidence_kernel,
                    iterations=1,
                ).astype(bool)
                & component
            )
        else:
            preserve = np.zeros_like(component)
        remove = component & ~preserve
        if int(remove.sum()) < POT_RIM_MINIMUM_COMPONENT_AREA:
            continue
        cleaned[remove] = False
        removed_pixels += int(remove.sum())
        preserved_evidence_pixels += int(preserve.sum())

    audit = {
        "pot_rim_components_inspected": count - 1,
        "pot_rim_suspicious_components": suspicious_components,
        "pot_rim_removed_pixels": removed_pixels,
        "pot_rim_preserved_evidence_pixels": preserved_evidence_pixels,
    }
    return cleaned, audit


def compute_multispectral_traits(bands: dict, mask: np.ndarray, **ctx) -> dict:
    """Full Tier-4 trait row.

    Optional ``ctx``:

    * ``pot_circle`` : (cx, cy, radius) — if given **and** ``rgb_support`` is
      also given, the mask is refined by :func:`remove_pot_rim_false_positive`
      before aggregation.
    * ``rgb_support`` : 0/1 RGB-vegetation support map (required for the rim
      filter to run).

    Returns index + reflectance summaries plus ``mask_area_px`` and, when the
    rim filter runs, ``pot_rim_*`` audit columns.
    """
    indices = compute_index_images(bands)
    ndvi = indices["NDVI"]
    m = (np.asarray(mask) > 0)
    audit = {}
    pot_circle = ctx.get("pot_circle")
    rgb_support = ctx.get("rgb_support")
    if pot_circle is not None and rgb_support is not None:
        m, audit = remove_pot_rim_false_positive(m, rgb_support, ndvi, pot_circle)

    row = summarize_indices_over_mask(indices, m)
    row.update(summarize_reflectance(bands, m))
    row["mask_area_px"] = int(m.sum())
    row.update(audit)
    return row


@register
class MultispectralExtractor(TraitExtractor):
    name = "multispectral_vegetation_indices"
    description = "Multispectral (MS400 4-band) vegetation indices + reflectance stats over the mask; optional pot-rim false-positive filter."
    requires = [INPUT_MASK, INPUT_MULTISPECTRAL]
    tier = 4

    def extract(self, *, mask=None, multispectral=None, **ctx):
        if multispectral is None:
            raise ValueError("multispectral input (dict of calibrated reflectance bands) is required")
        return compute_multispectral_traits(multispectral, mask, **ctx)
