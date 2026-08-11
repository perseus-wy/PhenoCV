# -*- coding: utf-8 -*-
"""CPU-only unit tests for the pure-logic layer (no torch / GPU)."""

import numpy as np
import pytest

from phenocv.segmentation import engine as e


def test_compute_roi_pads_and_stays_in_bounds():
    masks = [np.zeros((720, 1280), bool) for _ in range(3)]
    m = np.zeros((720, 1280), bool)
    m[100:140, 200:260] = True
    masks.append(m)
    roi = e.compute_roi(masks, (720, 1280))
    x0, y0, x1, y1 = roi
    assert 0 <= x0 and x1 <= 1280 and 0 <= y0 and y1 <= 720
    assert x0 <= 200 and y0 <= 100  # ROI must enclose the mask


def test_compute_roi_all_empty_degrades_to_full_image():
    masks = [np.zeros((100, 100), bool) for _ in range(3)]
    assert e.compute_roi(masks, (100, 100)) == (0, 0, 100, 100)


def test_threshold_ladder_base_and_fallback():
    lg = np.zeros((50, 50), np.float32)
    lg[10:30, 10:30] = 0.3
    mk, thr, fb = e.threshold_ladder(lg)
    assert int(mk.sum()) == 400 and fb is False

    lg3 = np.zeros((50, 50), np.float32)
    lg3[10:30, 10:30] = -0.2
    mk3, thr3, fb3 = e.threshold_ladder(lg3)
    assert int(mk3.sum()) == 2500 and thr3 == -0.5 and fb3 is True


def test_mask_iou():
    a = np.zeros((10, 10), bool)
    a[0:5, 0:5] = True
    b = a.copy()
    assert e.mask_iou(a, b) == 1.0
    assert e.mask_iou(a, np.zeros((10, 10), bool)) == 0.0
    assert e.mask_iou(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == 1.0


def test_boundary_f1():
    a = np.zeros((10, 10), bool)
    a[0:5, 0:5] = True
    b = a.copy()
    assert e.boundary_f1(a, b) == 1.0
    assert e.boundary_f1(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == 1.0


def test_mask_to_isat_objects():
    a = np.zeros((10, 10), bool)
    a[0:5, 0:5] = True
    objs = e.mask_to_isat_objects(a, min_area=1)
    assert len(objs) == 1 and objs[0]["category"] == "plant"
    assert round(objs[0]["area"], 1) == 16.0


def test_constrain_logits_to_box():
    lg = np.ones((20, 20), np.float32)
    out = e.constrain_logits_to_box(lg, (5, 5, 10, 10))
    assert out[7, 7] == 1.0       # inside box kept
    assert out[0, 0] == -1e9       # outside box pushed down


def test_plant_sequence_bounds_check():
    a = np.zeros((10, 10), bool)
    a[0:5, 0:5] = True
    with pytest.raises(ValueError):
        e.PlantSequence(key="p1", frame_paths=["a.png"],
                        anchors=[e.AnchorFrame(frame_idx=5, mask=a)])
