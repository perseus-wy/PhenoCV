# -*- coding: utf-8 -*-
"""CPU-only tests for the 4-tier phenotyping package (phenocv.phenotypes).

No torch / GPU; synthetic masks, RGB, depth and multispectral arrays only.
"""

import numpy as np
import pytest

from phenocv import phenotypes as P
from phenocv.phenotypes.calib import CameraIntrinsics
from phenocv.phenotypes.base import (
    INPUT_MASK, INPUT_RGB, INPUT_DEPTH, INPUT_CALIB, INPUT_MULTISPECTRAL,
)


# --------------------------------------------------------------------------
# Fixtures / synthetic data
# --------------------------------------------------------------------------
def _square_mask(size=120, lo=40, hi=80):
    m = np.zeros((size, size), bool)
    m[lo:hi, lo:hi] = True
    return m


def _disk_mask(size=120, cx=60, cy=60, r=50):
    yy, xx = np.ogrid[:size, :size]
    m = np.zeros((size, size), bool)
    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = True
    return m


def _flat_scene(h=200, w=200, soil=1000.0, canopy=900.0, cxy=(100, 100, 100)):
    depth = np.full((h, w), float(soil), np.float32)
    mask = np.zeros((h, w), bool)
    mask[cxy[1] - 20:cxy[1] + 20, cxy[0] - 20:cxy[0] + 20] = True
    depth[mask] = float(canopy)
    return depth, mask


# --------------------------------------------------------------------------
# Registry / routing
# --------------------------------------------------------------------------
def test_registry_has_four_tiers():
    names = set(P.all_extractors())
    assert {"shape2d", "rgb_vegetation_indices", "canopy_3d_geometry",
            "multispectral_vegetation_indices"} <= names


def test_available_for_routes_by_inputs():
    assert [e.name for e in P.available_for({INPUT_MASK})] == ["shape2d"]
    avail = P.available_for({INPUT_MASK, INPUT_RGB})
    assert "shape2d" in [e.name for e in avail]
    assert "rgb_vegetation_indices" in [e.name for e in avail]
    # depth alone does not satisfy the 3-input L3 requirement
    assert P.available_for({INPUT_DEPTH}) == []


# --------------------------------------------------------------------------
# Tier 1 — mask-only 2D shape
# --------------------------------------------------------------------------
def test_l1_square_metrics():
    m = _square_mask()
    r = P.compute_traits(mask=m)
    assert r["area_px"] == 40 * 40
    assert r["bbox_w"] == 40 and r["bbox_h"] == 40
    assert abs(r["aspect_ratio"] - 1.0) < 1e-6
    assert r["_extractors_run"] == ["shape2d"]


def test_l1_empty_mask_is_fail_closed():
    r = P.compute_traits(mask=np.zeros((50, 50), bool))
    assert r["area_px"] == 0
    assert np.isnan(r["solidity"]) and np.isnan(r["circularity"])


# --------------------------------------------------------------------------
# Tier 2 — mask + RGB normalized-RGB indices
# --------------------------------------------------------------------------
def test_l2_exg_spotlights_green_plant():
    m = _square_mask()
    rgb = np.full((120, 120, 3), 128, np.uint8)
    rgb[m] = [30, 200, 40]  # strong green vegetation
    r = P.compute_traits(mask=m, rgb=rgb)
    assert r["ExG_mean"] > 1.0
    assert r["GRVI_mean"] > 0.5
    assert r["ExG_valid_frac"] == 1.0
    # VARI/WI keep their (legitimately negative) denominator sign
    assert np.isfinite(r["VARI_mean"])


# --------------------------------------------------------------------------
# Tier 3 — depth + intrinsics -> plant height (mm)
# --------------------------------------------------------------------------
def test_l3_plant_height_on_flat_scene():
    depth, mask = _flat_scene()
    intr = CameraIntrinsics(200, 200, 600.0, 600.0, 100.0, 100.0)
    traits = P.compute_plant_height(depth, mask, intr,
                                    soil_plane=np.array([0.0, 0.0, 1000.0]))
    assert traits.missing_reason == ""
    assert abs(traits.mean_canopy_height_mm - 100.0) < 1e-3
    assert abs(traits.canopy_height_p95_mm - 100.0) < 1e-3
    assert traits.geometry_qc_pass
    assert traits.canopy_projected_area_mm2 > 0


def test_l3_orchestrator_autofits_soil_plane():
    depth, mask = _flat_scene()
    intr = CameraIntrinsics(200, 200, 600.0, 600.0, 100.0, 100.0)
    r = P.compute_traits(mask=mask, depth=depth, calibration=intr)
    assert "canopy_3d_geometry" in r["_extractors_run"]
    assert abs(r["mean_canopy_height_mm"] - 100.0) < 1.0
    assert r["missing_reason"] == ""


def test_l3_fit_soil_plane_recovers_tilted_ground():
    # Synthetic soil points on Z = 0.05*X + 0.02*Y + 500, with noise + outliers.
    rng = np.random.default_rng(0)
    n = 4000
    xy = rng.uniform(-300, 300, size=(n, 2))
    z = 0.05 * xy[:, 0] + 0.02 * xy[:, 1] + 500.0 + rng.normal(0, 1.0, n)
    pts = np.column_stack([xy, z])
    # sprinkle gross outliers (e.g. plant leaves poking into the soil annulus)
    pts[:400, 2] += rng.uniform(50, 200, 400)
    coeff, holdout_med, audit = P.fit_soil_plane(pts)
    assert coeff is not None and audit["status"] == "accepted"
    assert abs(coeff[0] - 0.05) < 0.02 and abs(coeff[1] - 0.02) < 0.02
    assert abs(coeff[2] - 500.0) < 5.0


def test_l3_fail_closed_on_degenerate_input():
    intr = CameraIntrinsics(200, 200, 600.0, 600.0, 100.0, 100.0)
    r = P.compute_traits(mask=np.zeros((200, 200), bool), depth=np.full((200, 200), 1000.0, np.float32),
                         calibration=intr)
    # empty mask -> soil plane unavailable -> fail-closed, no crash
    assert "canopy_3d_geometry" in r["_extractors_run"]
    assert "soil_plane" in (r.get("missing_reason") or "")


# --------------------------------------------------------------------------
# Tier 4 — multispectral indices + calibration + pot-rim filter
# --------------------------------------------------------------------------
def _uniform_bands(g=0.1, r=0.05, re=0.1, nir=0.5):
    return {
        555: np.full((120, 120), float(g), np.float32),
        660: np.full((120, 120), float(r), np.float32),
        720: np.full((120, 120), float(re), np.float32),
        840: np.full((120, 120), float(nir), np.float32),
    }


def test_l4_index_formulas():
    bands = _uniform_bands()
    m = _square_mask()
    r = P.compute_traits(mask=m, multispectral=bands)
    assert abs(r["NDVI_mean"] - 0.8182) < 1e-3       # (0.5-0.05)/(0.5+0.05)
    assert abs(r["NDRE_mean"] - 0.6667) < 1e-3       # (0.5-0.1)/(0.5+0.1)
    assert abs(r["GNDVI_mean"] - 0.6667) < 1e-3
    assert abs(r["MCARI_mean"] - 0.10) < 1e-6
    assert abs(r["TCARI_mean"] - 0.15) < 1e-6
    assert abs(r["R840_mean"] - 0.5) < 1e-6
    assert r["mask_area_px"] == 40 * 40


def test_l4_empirical_line_calibration():
    panel_medians = {555: 0.61, 660: 0.30, 720: 0.62, 840: 0.30}
    gains = P.empirical_line_gains(panel_medians)
    assert abs(gains[660] - 2.0) < 1e-6               # 0.60 / 0.30
    signals = {b: np.full((10, 10), 0.3, np.float32) for b in P.BANDS}
    refl = P.apply_gains(signals, gains)
    assert abs(float(np.nanmedian(refl[660])) - 0.6) < 1e-6


def test_l4_pot_rim_removes_unsupported_arcs():
    mask = _disk_mask()
    circle = (60, 60, 50)
    rgb_support = np.zeros((120, 120), bool)          # no RGB support anywhere
    ndvi = np.zeros((120, 120), np.float32)           # no NDVI support
    cleaned, audit = P.remove_pot_rim_false_positive(mask, rgb_support, ndvi, circle)
    assert audit["pot_rim_removed_pixels"] > 0
    # the protected core (strict < 0.68*radius) must survive untouched
    yy, xx = np.ogrid[:120, :120]
    core = (yy - 60) ** 2 + (xx - 60) ** 2 < (0.68 * 50) ** 2
    assert np.array_equal(cleaned, core)


# --------------------------------------------------------------------------
# Full orchestration
# --------------------------------------------------------------------------
def test_full_orchestrator_runs_all_four_tiers():
    m = _square_mask()
    rgb = np.full((120, 120, 3), 128, np.uint8)
    rgb[m] = [30, 200, 40]
    depth, _ = _flat_scene(h=120, w=120)
    mask120 = _square_mask(120)
    intr = CameraIntrinsics(120, 120, 600.0, 600.0, 60.0, 60.0)
    bands = _uniform_bands()
    r = P.compute_traits(mask=mask120, rgb=rgb, depth=depth, calibration=intr,
                         multispectral=bands)
    assert set(r["_extractors_run"]) == {
        "shape2d", "rgb_vegetation_indices", "canopy_3d_geometry",
        "multispectral_vegetation_indices",
    }
    # no extractor crashed
    assert all(not k.endswith("_error") for k in r)


def test_per_extractor_errors_are_recorded_not_fatal():
    # multispectral input with a missing band should not abort the whole row
    m = _square_mask(120)
    bad = {555: np.full((120, 120), 0.1, np.float32)}  # missing 660/720/840
    r = P.compute_traits(mask=m, multispectral=bad)
    assert "multispectral_vegetation_indices_error" in r
    # L1 still computed
    assert r["area_px"] == int(m.sum())
