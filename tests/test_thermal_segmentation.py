# -*- coding: utf-8 -*-
"""CPU-only tests for phenocv.thermal.segmentation.

All tests run without torch / sam2 (those are imported lazily only inside the
SAM 2 propagation functions). 全部测试无需 torch/sam2（仅在 SAM2 传播函数内惰性
导入），可在 CPU 环境通过。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Importing this module must NOT pull in torch/sam2.
import phenocv.thermal  # noqa: F401  (registers the package)
from phenocv.thermal import segmentation as ts
from phenocv.thermal.io import (
    thermal_feature_image,
    resolve_layer_overlap,
    load_temperature,
)
from phenocv.thermal.traits import (
    partition_canopy_by_relative_height,
    summarize_masked_temperature,
)


def test_import_without_torch_sam2():
    """Module imports even when torch/sam2 are not installed (lazy import)."""
    assert ts is not None
    assert callable(ts.segment_video_with_sam2)
    assert callable(ts.clean_target_mask)
    assert callable(ts.ThermalVideoSegmenter)
    # torch must not be imported merely by importing the module.
    assert "torch" not in sys.modules


def test_thermal_feature_image_shape_range():
    rng = np.random.default_rng(0)
    temp = rng.normal(27.0, 1.5, size=(48, 64))
    feat = thermal_feature_image(temp)
    assert feat.shape == (48, 64, 3)
    assert feat.dtype == np.uint8
    assert feat.min() >= 0 and feat.max() <= 255


def test_partition_canopy_disjoint_union():
    # A tall solid rectangle spanning many rows -> three non-empty layers.
    whole = np.zeros((60, 20), dtype=bool)
    whole[5:55, 5:15] = True
    layers = partition_canopy_by_relative_height(whole)
    assert set(layers) == {"upper", "middle", "lower"}
    for name, m in layers.items():
        assert m.any(), f"{name} should be non-empty"
    # mutually exclusive
    stacked = np.stack([layers["upper"], layers["middle"], layers["lower"]])
    assert int((stacked.sum(axis=0) > 1).sum()) == 0
    # union equals whole
    union = np.logical_or.reduce(list(stacked))
    assert int(np.logical_xor(union, whole).sum()) == 0


def test_partition_canopy_rejects_empty():
    with pytest.raises(ValueError):
        partition_canopy_by_relative_height(np.zeros((40, 20), dtype=bool))


def test_resolve_layer_overlap_mutual_exclusion():
    # Realistic contract: overlapping candidate masks + mutually-exclusive
    # identity seeds (a true partition). Overlap is assigned by nearest seed.
    upper_seed = np.zeros((20, 20), dtype=bool); upper_seed[:, :8] = True
    lower_seed = np.zeros((20, 20), dtype=bool); lower_seed[:, 12:] = True
    upper_cand = np.zeros((20, 20), dtype=bool); upper_cand[:, :12] = True
    lower_cand = np.zeros((20, 20), dtype=bool); lower_cand[:, 8:] = True
    masks = {"upper": upper_cand, "lower": lower_cand}
    seeds = {"upper": upper_seed, "lower": lower_seed}
    resolved, overlap_count = resolve_layer_overlap(masks, seeds)
    assert overlap_count == int((upper_cand & lower_cand).sum())
    assert overlap_count > 0
    # disjoint
    assert int((resolved["upper"] & resolved["lower"]).sum()) == 0
    # union stays within the original support
    support = upper_cand | lower_cand
    out_union = resolved["upper"] | resolved["lower"]
    assert int(np.logical_xor(out_union, support).sum()) == 0


def test_clean_target_mask_reference_keeps_positive_component():
    # Reference frame: two disconnected components; a positive point sits in A.
    raw = np.zeros((30, 30), dtype=bool)
    a = raw.copy()
    a[5:12, 5:12] = True
    b = raw.copy()
    b[20:27, 20:27] = True
    raw = a | b
    cleaned, info = ts.clean_target_mask(
        raw, reference_points=[(8.0, 8.0)], reference_labels=[1],
        is_reference=True,
    )
    assert int(cleaned.sum()) == int(a.sum())  # only A retained
    assert not info["hard_fail"]


def test_clean_target_mask_reference_negative_in_target_is_hard_fail():
    # A component that contains BOTH a positive and a negative point -> hard fail.
    raw = np.zeros((30, 30), dtype=bool)
    raw[5:20, 5:20] = True
    cleaned, info = ts.clean_target_mask(
        raw,
        reference_points=[(8.0, 8.0), (18.0, 18.0)],
        reference_labels=[1, 0],
        is_reference=True,
    )
    assert info["hard_fail"] is True


def test_clean_target_mask_propagated_uses_temporal_anchor():
    # Propagated frame: two components; only the one overlapping the (dilated)
    # anchor is retained.
    raw = np.zeros((40, 40), dtype=bool)
    a = raw.copy()
    a[5:12, 5:12] = True
    b = raw.copy()
    b[30:37, 30:37] = True
    raw = a | b
    anchor = np.zeros_like(raw)
    anchor[5:12, 5:12] = True  # same place as A
    cleaned, info = ts.clean_target_mask(
        raw, prev_mask=anchor, is_reference=False, dilate_px=2,
    )
    assert int(cleaned.sum()) == int(a.sum())
    assert info["support_mode"] == "anchor_overlap"


def test_clean_target_mask_box_only_keeps_largest():
    raw = np.zeros((30, 30), dtype=bool)
    a = raw.copy()
    a[2:6, 2:6] = True  # small
    b = raw.copy()
    b[15:25, 15:25] = True  # large
    raw = a | b
    cleaned, info = ts.clean_target_mask(raw, is_reference=True)  # no points
    assert info["support_mode"] == "box_only"
    assert int(cleaned.sum()) == int(b.sum())


def test_merge_bidirectional_fwd_priority_and_fill():
    shape = (4, 4)
    mask_a = np.zeros(shape, bool); mask_a[0, 0] = True
    mask_b = np.zeros(shape, bool); mask_b[1, 1] = True
    mask_c = np.zeros(shape, bool); mask_c[2, 2] = True
    fwd = {0: mask_a, 1: mask_b}
    bwd = {1: ~mask_b, 2: mask_c}  # bwd[1] differs but fwd wins
    merged = ts.merge_bidirectional(fwd, bwd, n=3, shape=shape)
    assert int(merged[0].sum()) == 1 and merged[0][0, 0]
    assert int(merged[1].sum()) == 1 and merged[1][1, 1]  # fwd priority
    assert int(merged[2].sum()) == 1 and merged[2][2, 2]  # bwd fill
    assert merged[2].dtype == bool


def test_segment_video_with_sam2_lazy_call_requires_sam2(tmp_path):
    """Calling without torch/sam2 installed must fail lazily (no top-level
    import of torch)."""
    with pytest.raises(Exception):
        ts.segment_video_with_sam2(
            temperature_frames=[np.zeros((8, 8))],
            reference_index=0,
            config_path="does_not_exist.yaml",  # skip auto-locate
            cache_dir=str(tmp_path / "cache"),
        )


def test_load_prompt_config_validation(tmp_path):
    # Valid box within 640x480.
    ok = tmp_path / "p_ok.yaml"
    ok.write_text(
        "prompt_type: box\nbox: [10, 20, 100, 200]\n", encoding="utf-8")
    cfg = ts.load_prompt_config(ok)
    assert cfg["prompt_type"] == "box"
    assert cfg["box"] == [10, 20, 100, 200]

    # Box out of range -> ValueError.
    bad = tmp_path / "p_bad.yaml"
    bad.write_text(
        "prompt_type: box\nbox: [10, 20, 700, 200]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ts.load_prompt_config(bad)


def test_summarize_masked_temperature_basic():
    temp = np.full((10, 10), 25.0)
    temp[0, 0] = 35.0  # one hot pixel
    mask = np.ones((10, 10), dtype=bool)
    stats = summarize_masked_temperature(temp, mask)
    assert stats["temp_median_c"] == 25.0
    assert stats["temp_p10_c"] == 25.0
    assert stats["pixel_count"] == 100


def _make_fake_segmenter(tmp_path):
    """Inject a fake propagation so run_segment runs without torch/sam2."""
    seg = ts.ThermalVideoSegmenter(checkpoint_path="ckpt.pt", device="cpu")
    # Precomputed whole-plant masks for 5 frames (a growing vertical bar).
    def _fake(temps, ref_idx, box, points, labels):
        n = len(temps)
        out = {}
        for gi in range(n):
            m = np.zeros(temps[gi].shape, dtype=bool)
            m[10:30 + gi * 2, 20:30] = True
            out[gi] = m
        return out

    seg._segment_fn = _fake
    return seg


def test_run_segment_full_pipeline_no_torch(tmp_path):
    """run_segment produces masks/CSV/review and disjoint layers without torch."""
    seg = _make_fake_segmenter(tmp_path)
    stems = ["f%03d" % i for i in range(5)]
    # Synthetic true-temperature frames (no NaN issues).
    temps = [np.full((60, 50), 26.0 + 0.1 * i) for i in range(5)]
    prompt_cfg = {"prompt_type": "box", "box": [18, 8, 32, 50]}
    out = tmp_path / "seg_out"
    summary = seg.run_segment(
        segment_id="segment_01",
        stems=stems,
        temperature_frames=temps,
        reference_stem="f000",
        prompt_cfg=prompt_cfg,
        output_dir=out,
    )
    assert summary["status"] == "OK"
    assert (out / "segment_complete.json").is_file()
    assert (out / "segment_01_thermal_metrics.csv").is_file()
    assert (out / "masks" / "whole" / "f000.png").is_file()
    for layer in ("upper", "middle", "lower"):
        assert (out / "masks" / layer / "f000.png").is_file()
    # Disjoint layers per frame.
    assert summary["all_layers_disjoint"] is True
    assert summary["all_unions_exact"] is True


def test_run_segment_fail_closed_on_hard_fail(tmp_path):
    """A cleanup hard fail must produce segment_failed_qc.json and raise."""
    seg = _make_fake_segmenter(tmp_path)

    def _fake_bad(temps, ref_idx, box, points, labels):
        n = len(temps)
        out = {}
        for gi in range(n):
            m = np.zeros(temps[gi].shape, dtype=bool)
            if gi == ref_idx:
                # reference: a component containing a positive AND a negative point
                m[10:40, 10:40] = True
            else:
                m[10:30, 20:30] = True
            out[gi] = m
        return out

    seg._segment_fn = _fake_bad
    stems = ["f%03d" % i for i in range(5)]
    temps = [np.full((60, 50), 26.0) for i in range(5)]
    # reference with a positive + negative point -> hard fail
    prompt_cfg = {
        "prompt_type": "points_and_box",
        "box": [8, 8, 42, 42],
        "points": [[12.0, 12.0], [38.0, 38.0]],
        "point_labels": [1, 0],
    }
    out = tmp_path / "seg_fail"
    with pytest.raises(RuntimeError):
        seg.run_segment(
            segment_id="segment_01",
            stems=stems,
            temperature_frames=temps,
            reference_stem="f000",
            prompt_cfg=prompt_cfg,
            output_dir=out,
        )
    assert (out / "segment_failed_qc.json").is_file()
    assert not (out / "segment_complete.json").is_file()
