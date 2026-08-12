# -*- coding: utf-8 -*-
"""Temporal canopy segmentation via SAM 2 video propagation.

Sparse keyframe masks -> full-sequence masks, with leave-one-out (LOO)
quality reporting and ISAT annotation export.

Pipeline layers
---------------
1. Pure-CPU logic layer (no torch dependency; directly testable with pytest)
   - :func:`compute_roi` / :func:`crop_to_roi` / :func:`paste_from_roi`
   - :func:`threshold_ladder` / :func:`constrain_logits_to_box`
   - :func:`mask_iou` / :func:`boundary_f1`
   - :func:`mask_to_isat_objects` / :func:`build_isat_document`
2. GPU propagation layer (torch/sam2 imported lazily)
   - :class:`Sam2VideoPropagator`
3. Orchestration layer
   - :func:`run_loo_validation` / :func:`run_full_propagation`
   - :func:`run_sam2_video_temporal` (programmatic entry point)

Key engineering notes (hard-won, do not "simplify" away)
-------------------------------------------------------
* **ROI cropping is mandatory.** SAM 2 internally resizes every frame to
  1024px. Feeding a full 1280x720 frame crushes early-stage seedlings
  (hundreds of pixels) out of existence. We crop a square ROI around the
  union bbox of the anchor masks (padded) before propagation, which lifts
  a small seedling's effective resolution by roughly an order of magnitude.
* **Bidirectional propagation.** Run forward and reverse, then average the
  logits before thresholding. Unidirectional propagation drifts near the
  sequence endpoints.
* **Threshold ladder fallback.** When the base threshold (0.0) yields an
  empty mask, step down (-0.5 / -1.0 / -2.0 / -4.0). This recovers tiny
  seedlings that a hard 0.0 cutoff would otherwise discard as "empty".
* **Point-rescue MUST be box-constrained.** An unconstrained centroid point
  prompt makes SAM 2 grab background in a far corner; we expand the nearest
  anchor bbox and push out-of-box logits to -1e9.
* torch/sam2 are imported lazily: importing this module must NOT pull in CUDA.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # cv2 is a hard dependency, but keep the import error legible
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("phenocv.segmentation.engine requires opencv-python") from exc


__all__ = [
    "TemporalPropagationConfig",
    "AnchorFrame",
    "PlantSequence",
    "compute_roi",
    "crop_to_roi",
    "paste_from_roi",
    "threshold_ladder",
    "constrain_logits_to_box",
    "rescue_box_from_anchor",
    "nearest_anchor_index",
    "mask_iou",
    "boundary_f1",
    "mask_to_isat_objects",
    "build_isat_document",
    "render_qa_grid",
    "Sam2VideoPropagator",
    "run_loo_validation",
    "run_full_propagation",
    "summarize_quality",
    "run_sam2_video_temporal",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class TemporalPropagationConfig:
    """All tunable parameters for temporal propagation.

    The defaults are the production values used for the reference soybean
    run; changing them affects reproducibility.
    """

    # -- ROI cropping --
    roi_pad_ratio: float = 1.9
    """Padding multiplier applied to the union bbox of anchor masks."""
    roi_min_size: int = 384
    """Minimum ROI side length (px), guards against over-small seedling ROIs."""
    roi_max_size: Optional[int] = None
    """Maximum ROI side length (px); None means bounded by image height."""

    # -- Propagation --
    bidirectional: bool = True
    """Average logits of forward + reverse passes when True."""
    base_threshold: float = 0.0
    """Base binarization threshold on the propagated logits."""
    threshold_ladder: Tuple[float, ...] = (-0.5, -1.0, -2.0, -4.0)
    """Fallback thresholds tried in order when the base yields an empty mask."""
    min_valid_area: int = 1
    """Minimum pixel count to count as a non-empty mask."""

    # -- Seedling point-rescue --
    rescue_enabled: bool = True
    rescue_box_ratio: float = 0.65
    """Box expansion factor relative to the nearest anchor bbox half-extent."""
    rescue_min_box: float = 8.0
    """Minimum box width/height (px) for the rescue prompt."""

    # -- Frame-stack export --
    jpeg_quality: int = 95

    # -- ISAT export --
    isat_category: str = "plant"
    isat_simplify_eps: float = 2.0
    isat_min_area: float = 10.0

    # -- QA --
    qa_grid_cols: int = 6
    qa_grid_tile: int = 200

    # -- Backend selection --
    backend: str = "sam2"
    """Segmentation backend identifier (``"sam2"`` default). Forwarded to the
    backend factory so a single config selects the algorithm; ``"sam2"`` keeps
    the historical behavior."""

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "TemporalPropagationConfig":
        """Build from a YAML/dict, ignoring unknown keys (forward-compatible)."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "threshold_ladder" in kwargs and kwargs["threshold_ladder"] is not None:
            kwargs["threshold_ladder"] = tuple(float(x) for x in kwargs["threshold_ladder"])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["threshold_ladder"] = list(self.threshold_ladder)
        return out


@dataclass
class AnchorFrame:
    """One human-labeled anchor frame."""

    frame_idx: int
    """0-based index within the full temporal sequence."""
    mask: np.ndarray
    """Boolean mask in full-image coordinates."""
    source: str = "manual"


@dataclass
class PlantSequence:
    """One plant's full temporal sequence + sparse anchors."""

    key: str
    """Sequence identifier, e.g. "plant_13"."""
    frame_paths: List[str]
    """Time-ordered full-sequence RGB paths."""
    anchors: List[AnchorFrame]
    """Sparse human-labeled anchors (frame_idx must fall within frame_paths)."""
    frame_labels: List[str] = field(default_factory=list)
    """Human-readable per-frame label (date/DAS); defaults to index."""
    frame_extras: List[Dict[str, Any]] = field(default_factory=list)
    """Per-frame extra metadata columns, passed through verbatim into the
    area.csv / frame_manifest.csv / loo_quality.csv rows.

    This is the data-source-agnostic extension point: a plant-phenotyping
    scenario can pass ``{"date":..., "das":..., "ts":...}``; any other domain
    can pass arbitrary keys. Defaults to a list of empty dicts.
    """
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.frame_paths)
        for a in self.anchors:
            if not 0 <= a.frame_idx < n:
                raise ValueError(
                    "%s: anchor frame_idx=%d out of range (n_frames=%d)"
                    % (self.key, a.frame_idx, n))
        if not self.frame_labels:
            self.frame_labels = [str(i) for i in range(n)]
        if len(self.frame_labels) != n:
            raise ValueError("%s: frame_labels length != frame_paths" % self.key)
        if not self.frame_extras:
            self.frame_extras = [{} for _ in range(n)]
        if len(self.frame_extras) != n:
            raise ValueError("%s: frame_extras length != frame_paths" % self.key)

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    @property
    def anchor_indices(self) -> List[int]:
        return [a.frame_idx for a in self.anchors]

    def extras_for(self, idx: int) -> Dict[str, Any]:
        """Extra metadata for frame ``idx`` (always a safely mutable copy)."""
        try:
            return dict(self.frame_extras[idx] or {})
        except IndexError:
            return {}


# --------------------------------------------------------------------------
# Pure-CPU logic layer
# --------------------------------------------------------------------------

def compute_roi(masks: Sequence[np.ndarray],
                image_shape: Tuple[int, int],
                cfg: Optional[TemporalPropagationConfig] = None) -> Tuple[int, int, int, int]:
    """Compute a square ROI from the union bbox of the masks.

    Parameters
    ----------
    masks : full-image-coordinate masks (empty masks allowed).
    image_shape : ``(height, width)``.
    cfg : parameters; defaults when None.

    Returns
    -------
    ``(x0, y0, x1, y1)`` (right/bottom open), guaranteed inside the image and
    as square as possible. Degrades to the whole image when all masks empty.
    """
    cfg = cfg or TemporalPropagationConfig()
    h, w = int(image_shape[0]), int(image_shape[1])

    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    for m in masks:
        if m is None:
            continue
        arr = np.asarray(m)
        if arr.size == 0 or not arr.any():
            continue
        ys, xs = np.where(arr)
        xs_min.append(int(xs.min()))
        xs_max.append(int(xs.max()))
        ys_min.append(int(ys.min()))
        ys_max.append(int(ys.max()))

    if not xs_min:
        return (0, 0, w, h)

    x0, x1 = min(xs_min), max(xs_max)
    y0, y1 = min(ys_min), max(ys_max)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    side = max(x1 - x0 + 1, y1 - y0 + 1) * float(cfg.roi_pad_ratio)
    side = max(side, float(cfg.roi_min_size))
    cap = float(cfg.roi_max_size) if cfg.roi_max_size else float(min(h, w))
    side = min(side, cap)
    half = side / 2.0

    rx0 = int(round(cx - half))
    ry0 = int(round(cy - half))
    size = int(round(side))
    rx0 = max(0, min(rx0, w - size)) if size <= w else 0
    ry0 = max(0, min(ry0, h - size)) if size <= h else 0
    rx1 = min(w, rx0 + size)
    ry1 = min(h, ry0 + size)
    return (rx0, ry0, rx1, ry1)


def crop_to_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop by ROI (handles 2D masks and 3D images)."""
    x0, y0, x1, y1 = roi
    return image[y0:y1, x0:x1]


def paste_from_roi(sub: np.ndarray,
                   roi: Tuple[int, int, int, int],
                   full_shape: Tuple[int, int]) -> np.ndarray:
    """Paste an ROI-space mask back into full-image coordinates."""
    x0, y0, x1, y1 = roi
    h, w = int(full_shape[0]), int(full_shape[1])
    full = np.zeros((h, w), dtype=bool)
    sub_b = np.asarray(sub).astype(bool)
    th, tw = y1 - y0, x1 - x0
    if sub_b.shape != (th, tw):
        sub_b = cv2.resize(sub_b.astype(np.uint8), (tw, th),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    full[y0:y1, x0:x1] = sub_b
    return full


def threshold_ladder(logits: np.ndarray,
                     cfg: Optional[TemporalPropagationConfig] = None
                     ) -> Tuple[np.ndarray, float, bool]:
    """Threshold-ladder binarization: step down on empty.

    Returns
    -------
    ``(mask, threshold_used, used_fallback)``
    """
    cfg = cfg or TemporalPropagationConfig()
    lg = np.asarray(logits, dtype=np.float32)

    mask = lg > cfg.base_threshold
    if int(mask.sum()) >= cfg.min_valid_area:
        return mask, float(cfg.base_threshold), False

    for thr in cfg.threshold_ladder:
        mask = lg > float(thr)
        if int(mask.sum()) >= cfg.min_valid_area:
            return mask, float(thr), True

    return np.zeros(lg.shape, dtype=bool), float(cfg.base_threshold), True


def constrain_logits_to_box(logits: np.ndarray,
                            box: Tuple[float, float, float, float],
                            fill: float = -1e9) -> np.ndarray:
    """Push logits outside ``box`` to ``fill``.

    The key step of point-rescue — without it, SAM 2 grabs background in a
    far corner as the target.
    """
    lg = np.asarray(logits, dtype=np.float32)
    h, w = lg.shape[:2]
    bx0, by0, bx1, by1 = box
    x0 = max(0, min(int(round(bx0)), w - 1))
    y0 = max(0, min(int(round(by0)), h - 1))
    x1 = max(x0 + 1, min(int(round(bx1)), w))
    y1 = max(y0 + 1, min(int(round(by1)), h))
    out = np.full(lg.shape, float(fill), dtype=np.float32)
    out[y0:y1, x0:x1] = lg[y0:y1, x0:x1]
    return out


def rescue_box_from_anchor(anchor_mask: np.ndarray,
                           cfg: Optional[TemporalPropagationConfig] = None
                           ) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Derive the rescue ``(cx, cy, bx0, by0, bx1, by1)`` from the nearest anchor.

    Returns None when the anchor mask is empty.
    """
    cfg = cfg or TemporalPropagationConfig()
    arr = np.asarray(anchor_mask).astype(bool)
    if not arr.any():
        return None
    ys, xs = np.where(arr)
    cx, cy = float(xs.mean()), float(ys.mean())
    bw = max(float(xs.max() - xs.min()), cfg.rescue_min_box)
    bh = max(float(ys.max() - ys.min()), cfg.rescue_min_box)
    r = float(cfg.rescue_box_ratio)
    h, w = arr.shape[:2]
    bx0 = max(0.0, cx - r * bw)
    by0 = max(0.0, cy - r * bh)
    bx1 = min(float(w - 1), cx + r * bw)
    by1 = min(float(h - 1), cy + r * bh)
    return (cx, cy, bx0, by0, bx1, by1)


def nearest_anchor_index(frame_idx: int, anchor_indices: Sequence[int]) -> Optional[int]:
    """Index of the anchor nearest to ``frame_idx``; None when no anchors."""
    if not anchor_indices:
        return None
    return min(anchor_indices, key=lambda k: (abs(k - frame_idx), k))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union; both empty -> 1.0 (agree on "no target")."""
    aa = np.asarray(a).astype(bool)
    bb = np.asarray(b).astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return 1.0
    return inter / float(union)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1 (BF1): compares contours within ``tolerance`` pixels.

    Both empty -> 1.0; one empty, one not -> 0.0.
    """
    p = np.asarray(pred).astype(np.uint8)
    g = np.asarray(gt).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    def _edge(m: np.ndarray) -> np.ndarray:
        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        return (m - er).astype(bool)

    pe, ge = _edge(p), _edge(g)
    if not pe.any() or not ge.any():
        return 0.0

    k = 2 * int(tolerance) + 1
    kernel = np.ones((k, k), np.uint8)
    pe_d = cv2.dilate(pe.astype(np.uint8), kernel, iterations=1).astype(bool)
    ge_d = cv2.dilate(ge.astype(np.uint8), kernel, iterations=1).astype(bool)

    precision = float(np.logical_and(pe, ge_d).sum()) / float(pe.sum())
    recall = float(np.logical_and(ge, pe_d).sum()) / float(ge.sum())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mask_to_isat_objects(mask: np.ndarray,
                         simplify_eps: float = 2.0,
                         min_area: float = 10.0,
                         category: str = "plant") -> List[Dict[str, Any]]:
    """Mask -> list of ISAT objects.

    External contours only (no hole rings) + Douglas-Peucker simplification;
    one object per connected component, group incrementing.
    """
    arr = np.asarray(mask)
    binary = (arr > 127).astype(np.uint8) if arr.dtype != bool else arr.astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: List[Dict[str, Any]] = []
    group = 1
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        if simplify_eps and simplify_eps > 0:
            cnt = cv2.approxPolyDP(cnt, float(simplify_eps), True)
        if len(cnt) < 3:
            continue
        seg = [[float(pt[0][0]), float(pt[0][1])] for pt in cnt]
        x, y, w, h = cv2.boundingRect(cnt)
        objects.append({
            "category": category,
            "group": group,
            "segmentation": seg,
            "area": area,
            "layer": float(group),
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "iscrowd": False,
            "note": "",
        })
        group += 1
    return objects


def build_isat_document(mask: np.ndarray,
                        image_name: str,
                        image_folder: str,
                        image_size: Tuple[int, int],
                        note: str = "",
                        cfg: Optional[TemporalPropagationConfig] = None) -> Dict[str, Any]:
    """Assemble one ISAT JSON (``{"info": ..., "objects": [...]}``)."""
    cfg = cfg or TemporalPropagationConfig()
    h, w = int(image_size[0]), int(image_size[1])
    objects = mask_to_isat_objects(mask, cfg.isat_simplify_eps,
                                   cfg.isat_min_area, cfg.isat_category)
    return {
        "info": {
            "description": "ISAT",
            "folder": image_folder,
            "name": image_name,
            "width": w,
            "height": h,
            "depth": 3,
            "note": note,
        },
        "objects": objects,
    }


def render_qa_grid(frame_dir: str,
                   masks: Sequence[np.ndarray],
                   out_png: str,
                   labels: Optional[Sequence[str]] = None,
                   cfg: Optional[TemporalPropagationConfig] = None) -> str:
    """Stitch the whole sequence's ROI frames + mask contours into one QA grid."""
    cfg = cfg or TemporalPropagationConfig()
    cols = int(cfg.qa_grid_cols)
    tile = int(cfg.qa_grid_tile)
    n = len(masks)
    rows = int(np.ceil(n / float(cols))) if n else 1
    canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)

    for i, sub in enumerate(masks):
        jp = os.path.join(frame_dir, "%05d.jpg" % i)
        bgr = cv2.imread(jp)
        if bgr is None:
            continue
        vis = bgr.copy()
        cnts, _ = cv2.findContours(np.asarray(sub).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        vis = cv2.resize(vis, (tile, tile), interpolation=cv2.INTER_AREA)
        tag = labels[i] if labels and i < len(labels) else "%02d" % i
        cv2.putText(vis, str(tag), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = vis

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, canvas)
    return out_png


# --------------------------------------------------------------------------
# GPU propagation layer (torch/sam2 imported lazily)
# --------------------------------------------------------------------------

DEFAULT_SAM2_CONFIG = "sam2.1_hiera_l.yaml"


class Sam2VideoPropagator:
    """Thin wrapper around the SAM 2 video predictor (frame-stack export +
    bidirectional propagation).

    torch / sam2 are imported on first use, so importing this module does not
    pull in CUDA.
    """

    def __init__(self,
                 checkpoint: str,
                 model_cfg: str = DEFAULT_SAM2_CONFIG,
                 device: str = "cuda",
                 cfg: Optional[TemporalPropagationConfig] = None) -> None:
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self.cfg = cfg or TemporalPropagationConfig()
        self._predictor = None
        self._torch = None

    # -- lazy init --------------------------------------------------------
    @property
    def predictor(self):
        if self._predictor is None:
            import torch  # noqa: lazy by design
            from sam2.build_sam import build_sam2_video_predictor
            self._torch = torch
            if not os.path.exists(self.checkpoint):
                raise FileNotFoundError("SAM 2 checkpoint not found: %s" % self.checkpoint)
            self._predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint, device=self.device)
        return self._predictor

    # -- frame-stack export ------------------------------------------------
    def export_frame_stack(self,
                           frame_paths: Sequence[str],
                           roi: Tuple[int, int, int, int],
                           cache_dir: str) -> str:
        """Crop the whole sequence by ROI and write it as SAM 2's ``%05d.jpg``."""
        os.makedirs(cache_dir, exist_ok=True)
        quality = int(self.cfg.jpeg_quality)
        for i, path in enumerate(frame_paths):
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError("Cannot read frame: %s" % path)
            sub = crop_to_roi(bgr, roi)
            out = os.path.join(cache_dir, "%05d.jpg" % i)
            cv2.imwrite(out, sub, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cache_dir

    def init_state(self, frame_dir: str):
        return self.predictor.init_state(video_path=frame_dir)

    # -- propagation ------------------------------------------------------
    def propagate(self,
                  frame_dir: str,
                  prompts: Dict[int, np.ndarray],
                  reverse: bool = False) -> Dict[int, np.ndarray]:
        """Build a fresh inference state on ``frame_dir``, inject anchor masks
        and propagate in one direction.

        Each call re-``init_state``s, so reusing one ``inference_state`` across
        multiple ``propagate_in_video`` calls never accumulates CUDA memory.
        The state is released and ``empty_cache`` called afterwards to bound
        peak memory.
        """
        pred = self.predictor
        state = pred.init_state(video_path=frame_dir)
        try:
            pred.reset_state(state)
            for fidx, mask in sorted(prompts.items()):
                pred.add_new_mask(state, frame_idx=int(fidx), obj_id=1,
                                  mask=np.asarray(mask).astype(bool))
            out: Dict[int, np.ndarray] = {}
            for fidx, _obj_ids, logits in pred.propagate_in_video(state, reverse=reverse):
                lg = logits[0]
                if hasattr(lg, "ndim") and lg.ndim == 3:
                    lg = lg[0]
                out[int(fidx)] = lg.float().cpu().numpy()
            return out
        finally:
            del state
            if self._torch is not None:
                self._torch.cuda.empty_cache()

    def bidirectional(self,
                      frame_dir: str,
                      prompts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Run forward + reverse (each a fresh ``init_state``), average logits."""
        fwd = self.propagate(frame_dir, prompts, reverse=False)
        if not self.cfg.bidirectional:
            return fwd
        rev = self.propagate(frame_dir, prompts, reverse=True)
        merged: Dict[int, np.ndarray] = {}
        for k in set(fwd) | set(rev):
            a, b = fwd.get(k), rev.get(k)
            if a is None:
                merged[k] = b
            elif b is None:
                merged[k] = a
            else:
                merged[k] = 0.5 * (a + b)
        return merged

    # -- per-frame point-rescue -------------------------------------------
    def point_rescue(self,
                     frame_dir: str,
                     frame_idx: int,
                     anchor_mask: np.ndarray) -> Optional[np.ndarray]:
        """Single-frame point prompt using the nearest anchor's centroid + bbox;
        returns the box-constrained logits."""
        geom = rescue_box_from_anchor(anchor_mask, self.cfg)
        if geom is None:
            return None
        cx, cy, bx0, by0, bx1, by1 = geom
        pred = self.predictor
        state = pred.init_state(video_path=frame_dir)
        try:
            pred.reset_state(state)
            _fi, _ids, out = pred.add_new_points_or_box(
                state, frame_idx=int(frame_idx), obj_id=1,
                points=np.array([[cx, cy]], np.float32),
                labels=np.array([1], np.int32),
                box=np.array([bx0, by0, bx1, by1], np.float32))
            lg = out[0]
            if hasattr(lg, "ndim") and lg.ndim == 3:
                lg = lg[0]
            lg = lg.float().cpu().numpy()
            return constrain_logits_to_box(lg, (bx0, by0, bx1, by1))
        finally:
            del state
            if self._torch is not None:
                self._torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Orchestration layer
# --------------------------------------------------------------------------

def _resolve_mask(logits: np.ndarray,
                  cfg: TemporalPropagationConfig) -> Tuple[np.ndarray, float, str]:
    """logits -> (mask, thr, source_tag)."""
    mask, thr, fell_back = threshold_ladder(logits, cfg)
    if int(mask.sum()) >= cfg.min_valid_area:
        return mask, thr, ("propagated_lowthr" if fell_back else "propagated")
    return mask, thr, "empty"


def run_loo_validation(seq: PlantSequence,
                       propagator: Sam2VideoPropagator,
                       frame_dir: str,
                       roi: Tuple[int, int, int, int],
                       cfg: Optional[TemporalPropagationConfig] = None,
                       progress=None) -> List[Dict[str, Any]]:
    """Leave-one-out validation: hide each anchor in turn, propagate with the
    rest, and compare against the hidden ground truth.

    This is the only way to quantify interpolation accuracy without adding
    any new annotations.

    Returns
    -------
    One row per fold, with iou / bf1 / gap_frames / is_endpoint, etc.
    """
    cfg = cfg or propagator.cfg
    rows: List[Dict[str, Any]] = []
    anchors = sorted(seq.anchors, key=lambda a: a.frame_idx)
    if len(anchors) < 2:
        return rows

    all_idx = [a.frame_idx for a in anchors]
    last = seq.n_frames - 1

    for held in anchors:
        prompts = {
            a.frame_idx: crop_to_roi(a.mask, roi)
            for a in anchors if a.frame_idx != held.frame_idx
        }
        t0 = time.time()
        logits_map = propagator.bidirectional(frame_dir, prompts)
        lg = logits_map.get(held.frame_idx)
        if lg is None:
            pred_sub = np.zeros((roi[3] - roi[1], roi[2] - roi[0]), bool)
            thr, src = float(cfg.base_threshold), "missing"
        else:
            pred_sub, thr, src = _resolve_mask(lg, cfg)

        gt_sub = crop_to_roi(held.mask, roi)
        others = [k for k in all_idx if k != held.frame_idx]
        gap = min(abs(k - held.frame_idx) for k in others) if others else -1

        row: Dict[str, Any] = {
            "sequence": seq.key,
            "frame_idx": held.frame_idx,
            "frame_label": seq.frame_labels[held.frame_idx],
        }
        row.update(seq.extras_for(held.frame_idx))
        row.update({
            "iou": round(mask_iou(pred_sub, gt_sub), 6),
            "bf1": round(boundary_f1(pred_sub, gt_sub), 6),
            "gt_area": int(np.asarray(gt_sub).astype(bool).sum()),
            "pred_area": int(np.asarray(pred_sub).astype(bool).sum()),
            "gap_frames": int(gap),
            "is_endpoint": int(held.frame_idx in (0, last)),
            "thr": thr,
            "pred_source": src,
            "n_prompts": len(prompts),
            "seconds": round(time.time() - t0, 2),
        })
        rows.append(row)
        if progress:
            progress("LOO %s k=%d IoU=%.3f" % (seq.key, held.frame_idx, rows[-1]["iou"]))
    return rows


def run_full_propagation(seq: PlantSequence,
                         propagator: Sam2VideoPropagator,
                         frame_dir: str,
                         roi: Tuple[int, int, int, int],
                         cfg: Optional[TemporalPropagationConfig] = None,
                         progress=None
                         ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Propagate with all anchors once, producing the full-sequence masks (ROI
    coordinates).

    Anchor frames use the human mask directly; non-anchor frames go through
    propagation + threshold ladder + point-rescue.
    """
    cfg = cfg or propagator.cfg
    anchors = sorted(seq.anchors, key=lambda a: a.frame_idx)
    anchor_map = {a.frame_idx: a for a in anchors}
    prompts = {a.frame_idx: crop_to_roi(a.mask, roi) for a in anchors}

    logits_map = propagator.bidirectional(frame_dir, prompts) if prompts else {}

    masks: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    empty = np.zeros((roi[3] - roi[1], roi[2] - roi[0]), bool)

    for i in range(seq.n_frames):
        if i in anchor_map:
            sub = crop_to_roi(anchor_map[i].mask, roi).astype(bool)
            masks.append(sub)
            rows.append(_frame_row(seq, i, sub, float(cfg.base_threshold), "manual", roi))
            continue

        lg = logits_map.get(i)
        if lg is None:
            sub, thr, src = empty.copy(), float(cfg.base_threshold), "empty"
        else:
            sub, thr, src = _resolve_mask(lg, cfg)

        if src == "empty" and cfg.rescue_enabled and anchors:
            k = nearest_anchor_index(i, list(anchor_map))
            if k is not None:
                r_lg = propagator.point_rescue(frame_dir, i, crop_to_roi(anchor_map[k].mask, roi))
                if r_lg is not None:
                    r_mask, r_thr, _ = threshold_ladder(r_lg, cfg)
                    if int(r_mask.sum()) >= cfg.min_valid_area:
                        sub, thr, src = r_mask, r_thr, "point_rescue"
        # NOTE: the rescue runs on an independent init_state, so the shared
        # `logits_map` (computed from all anchors) is already correct for every
        # subsequent frame. Re-running bidirectional here is pure waste.

        if src == "empty":
            src = "failed_empty"
        masks.append(sub.astype(bool))
        rows.append(_frame_row(seq, i, sub, thr, src, roi))
        if progress and (i % 12 == 0 or i == seq.n_frames - 1):
            progress("FULL %s %d/%d" % (seq.key, i + 1, seq.n_frames))

    return masks, rows


def _frame_row(seq: PlantSequence, idx: int, sub_mask: np.ndarray,
               thr: float, source: str,
               roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sequence": seq.key,
        "frame_idx": idx,
        "frame_label": seq.frame_labels[idx],
    }
    row.update(seq.extras_for(idx))
    row.update({
        "frame_path": seq.frame_paths[idx],
        "area_px": int(np.asarray(sub_mask).astype(bool).sum()),
        "thr": thr,
        "pred_source": source,
    })
    if roi is not None:
        row.update({
            "roi_x": int(roi[0]),
            "roi_y": int(roi[1]),
            "roi_w": int(roi[2] - roi[0]),
            "roi_h": int(roi[3] - roi[1]),
        })
    return row


def summarize_quality(loo_rows: Sequence[Dict[str, Any]],
                      interior_only: bool = True) -> Dict[str, Any]:
    """Summarize LOO metrics.

    ``interior_only=True`` drops endpoint folds — endpoint LOO is extrapolation
    rather than interpolation, and in production the endpoints are anchors
    anyway, so including them understates real accuracy.
    """
    rows = [r for r in loo_rows if not (interior_only and int(r.get("is_endpoint", 0)))]
    if not rows:
        return {"n": 0}
    iou = np.array([float(r["iou"]) for r in rows])
    bf1 = np.array([float(r["bf1"]) for r in rows])
    return {
        "n": int(len(rows)),
        "interior_only": bool(interior_only),
        "iou_median": round(float(np.median(iou)), 4),
        "iou_mean": round(float(iou.mean()), 4),
        "iou_min": round(float(iou.min()), 4),
        "iou_p10": round(float(np.percentile(iou, 10)), 4),
        "bf1_median": round(float(np.median(bf1)), 4),
        "frac_ge_085": round(float((iou >= 0.85).mean()), 4),
        "frac_ge_070": round(float((iou >= 0.70).mean()), 4),
    }


def _write_csv(rows: Sequence[Dict[str, Any]], path: str) -> Optional[str]:
    """Write CSV. Column names = union of all row keys (first-seen order),
    missing values left blank.

    Union (not first-row keys) because ``frame_extras`` may carry different
    metadata columns across sequences.
    """
    if not rows:
        return None
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_sam2_video_temporal(sequences: Sequence[PlantSequence],
                            output_root: str,
                            checkpoint: str,
                            model_cfg: str = DEFAULT_SAM2_CONFIG,
                            device: str = "cuda",
                            cache_root: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None,
                            do_loo: bool = True,
                            export_isat: bool = True,
                            export_qa: bool = True,
                            resume: bool = True,
                            image_size: Tuple[int, int] = (720, 1280),
                            progress=None) -> Dict[str, Any]:
    """Top-level orchestration: LOO validation + full propagation + exports.

    Parameters
    ----------
    sequences : sequences to process.
    output_root : output root; one sub-directory per sequence.
    cache_root : ROI frame-stack cache; defaults to ``<output_root>/_frames``.
                 Prefer a fast local disk — network storage slows SAM 2 frame
                 reads considerably.
    resume : skip sequences already marked ``DONE``.

    Returns
    -------
    Result dict with ``summary`` / ``loo_summary`` / CSV paths.
    """
    cfg = TemporalPropagationConfig.from_mapping(config)
    propagator = Sam2VideoPropagator(checkpoint, model_cfg, device, cfg)
    cache_root = cache_root or os.path.join(output_root, "_frames")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    def _log(msg: str) -> None:
        if progress:
            progress(msg)
        else:
            print(msg, flush=True)

    loo_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    seq_rows: List[Dict[str, Any]] = []
    t_start = time.time()

    for seq in sequences:
        seq_dir = os.path.join(output_root, seq.key)
        done_flag = os.path.join(seq_dir, "DONE")
        if resume and os.path.exists(done_flag):
            _log("skip %s (DONE)" % seq.key)
            continue
        if not seq.anchors:
            _log("skip %s (no anchors)" % seq.key)
            continue

        t0 = time.time()
        roi = compute_roi([a.mask for a in seq.anchors], image_size, cfg)
        frame_dir = os.path.join(cache_root, seq.key)
        propagator.export_frame_stack(seq.frame_paths, roi, frame_dir)
        _log("%s roi=%s frames=%d anchors=%d" %
             (seq.key, roi, seq.n_frames, len(seq.anchors)))

        if do_loo:
            rows = run_loo_validation(seq, propagator, frame_dir, roi, cfg, progress)
            loo_rows.extend(rows)

        masks, rows = run_full_propagation(seq, propagator, frame_dir, roi, cfg, progress)
        frame_rows.extend(rows)

        _persist_sequence(seq, masks, rows, roi, seq_dir, frame_dir,
                          image_size, cfg, export_isat, export_qa)

        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["pred_source"]] = counts.get(r["pred_source"], 0) + 1
        areas = [r["area_px"] for r in rows]
        seq_rows.append({
            "sequence": seq.key,
            "n_frames": seq.n_frames,
            "n_anchors": len(seq.anchors),
            "roi": "%d,%d,%d,%d" % roi,
            "area_first": areas[0] if areas else 0,
            "area_last": areas[-1] if areas else 0,
            "area_max": max(areas) if areas else 0,
            "n_manual": counts.get("manual", 0),
            "n_propagated": counts.get("propagated", 0),
            "n_lowthr": counts.get("propagated_lowthr", 0),
            "n_rescue": counts.get("point_rescue", 0),
            "n_failed": counts.get("failed_empty", 0),
            "seconds": round(time.time() - t0, 1),
        })
        with open(done_flag, "w", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        _log("%s done in %.1fs %s" % (seq.key, time.time() - t0, counts))

    result: Dict[str, Any] = {
        "status": "completed",
        "output_root": output_root,
        "n_sequences": len(seq_rows),
        "n_frames": len(frame_rows),
        "elapsed_min": round((time.time() - t_start) / 60.0, 1),
        "config": cfg.to_dict(),
        "loo_summary_interior": summarize_quality(loo_rows, True),
        "loo_summary_all": summarize_quality(loo_rows, False),
        "loo_quality_csv": _write_csv(loo_rows, os.path.join(output_root, "loo_quality.csv")),
        "frame_manifest_csv": _write_csv(frame_rows, os.path.join(output_root, "frame_manifest.csv")),
        "sequence_summary_csv": _write_csv(seq_rows, os.path.join(output_root, "sequence_summary.csv")),
    }
    with open(os.path.join(output_root, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def _persist_sequence(seq: PlantSequence,
                      masks: Sequence[np.ndarray],
                      rows: Sequence[Dict[str, Any]],
                      roi: Tuple[int, int, int, int],
                      seq_dir: str,
                      frame_dir: str,
                      image_size: Tuple[int, int],
                      cfg: TemporalPropagationConfig,
                      export_isat: bool,
                      export_qa: bool) -> None:
    """Persist: full-image mask PNGs + ISAT JSONs + area.csv + QA grid."""
    mask_dir = os.path.join(seq_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    json_dir = os.path.join(seq_dir, "jsons")
    if export_isat:
        os.makedirs(json_dir, exist_ok=True)

    area_rows = []
    for i, sub in enumerate(masks):
        full = paste_from_roi(sub, roi, image_size)
        stem = os.path.splitext(os.path.basename(seq.frame_paths[i]))[0]
        cv2.imwrite(os.path.join(mask_dir, "%s.png" % stem),
                    (full.astype(np.uint8) * 255))

        if export_isat:
            note = "%s | frame_idx=%d | label=%s | src=%s" % (
                seq.key, i, seq.frame_labels[i], rows[i]["pred_source"])
            doc = build_isat_document(
                full,
                image_name=os.path.basename(seq.frame_paths[i]),
                image_folder=os.path.dirname(seq.frame_paths[i]),
                image_size=image_size,
                note=note,
                cfg=cfg)
            with open(os.path.join(json_dir, "%s.json" % stem), "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)

        area_row: Dict[str, Any] = {
            "sequence": seq.key,
            "frame_idx": i,
            "frame_label": seq.frame_labels[i],
        }
        area_row.update(seq.extras_for(i))
        area_row.update({
            "stem": stem,
            "area_px": int(full.sum()),
            "pred_source": rows[i]["pred_source"],
            "thr": rows[i].get("thr"),
        })
        area_rows.append(area_row)

    _write_csv(area_rows, os.path.join(seq_dir, "area.csv"))
    if export_qa:
        render_qa_grid(frame_dir, masks, os.path.join(seq_dir, "qa_grid.png"),
                       labels=seq.frame_labels, cfg=cfg)
