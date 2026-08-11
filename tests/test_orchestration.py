# -*- coding: utf-8 -*-
"""End-to-end orchestration tests using a mock propagator (no torch / GPU).

IMPORTANT: the engine assumes the propagator returns ROI-space logits, because
SAM 2 runs on the ROI-cropped frame stack. So the mock must return ROI-space
logits (per the ROI used), and comparisons happen in ROI space.
"""

import numpy as np

from phenocv import engine as e


class MockPropagator:
    """Returns 10 * ground-truth-mask (ROI space) as logits (binary -> exact)."""

    def __init__(self, gt_masks_roi):
        self.gt = gt_masks_roi
        self.cfg = e.TemporalPropagationConfig()

    def bidirectional(self, frame_dir, prompts):
        return {i: (self.gt[i].astype(np.float32) * 10.0)
                for i in range(len(self.gt))}

    def point_rescue(self, frame_dir, frame_idx, anchor_mask):
        return None


def _make_seq(n=6, h=240, w=320):
    frames, anchors, gt = [], [], []
    for i in range(n):
        r = 18 + i * 6
        mask = np.zeros((h, w), bool)
        yy, xx = np.ogrid[:h, :w]
        mask[(xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= r * r] = True
        gt.append(mask)
        frames.append("f%d.png" % i)
        if i in (0, 3, 5):
            anchors.append(e.AnchorFrame(frame_idx=i, mask=mask.copy(), source="manual"))
    seq = e.PlantSequence(
        key="demo_01", frame_paths=frames, anchors=anchors,
        frame_labels=["T%d" % i for i in range(n)],
        frame_extras=[{"das": i + 1} for i in range(n)])
    return seq, gt


def test_run_full_propagation_perfect_mock():
    seq, gt = _make_seq()
    roi = e.compute_roi([a.mask for a in seq.anchors], (240, 320))
    gt_roi = [e.crop_to_roi(g, roi) for g in gt]
    masks, rows = e.run_full_propagation(seq, MockPropagator(gt_roi), "fake_dir", roi)
    assert len(masks) == seq.n_frames
    for i in range(seq.n_frames):
        assert e.mask_iou(masks[i], gt_roi[i]) == 1.0, ("frame", i)
    src = {r["frame_idx"]: r["pred_source"] for r in rows}
    assert src[0] == "manual"
    assert src[5] == "manual"
    assert src[1] == "propagated"


def test_run_loo_validation_perfect_mock():
    seq, gt = _make_seq()
    roi = e.compute_roi([a.mask for a in seq.anchors], (240, 320))
    gt_roi = [e.crop_to_roi(g, roi) for g in gt]
    rows = e.run_loo_validation(seq, MockPropagator(gt_roi), "fake_dir", roi)
    assert len(rows) == 3  # one fold per anchor
    for r in rows:
        assert r["iou"] >= 0.99
        assert r["pred_source"] in ("propagated", "propagated_lowthr")


def test_summarize_quality_interior_only():
    seq, gt = _make_seq()
    roi = e.compute_roi([a.mask for a in seq.anchors], (240, 320))
    gt_roi = [e.crop_to_roi(g, roi) for g in gt]
    rows = e.run_loo_validation(seq, MockPropagator(gt_roi), "fake_dir", roi)
    s = e.summarize_quality(rows, interior_only=True)
    # anchors [0,3,5]; 0 and 5 are endpoints -> dropped -> only idx 3 remains
    assert s["n"] == 1
    assert s["iou_median"] >= 0.99


def test_frame_row_carries_extras_and_roi():
    seq, gt = _make_seq()
    roi = (10, 20, 110, 120)
    gt_roi = [e.crop_to_roi(g, roi) for g in gt]
    _, rows = e.run_full_propagation(seq, MockPropagator(gt_roi), "fake_dir", roi)
    r0 = next(r for r in rows if r["frame_idx"] == 0)
    assert r0["das"] == 1            # frame_extras passthrough (int from _make_seq)
    assert r0["roi_x"] == 10 and r0["roi_w"] == 100
