# -*- coding: utf-8 -*-
"""Thermal (FLIR) temporal segmentation via SAM 2 video propagation.

热红外时序分割（基于 SAM2 视频传播 + 热特征提示 + 目标锚定清理 + 分层）。

Pipeline layers
---------------
1. Pure-CPU logic layer (no torch dependency; directly testable with pytest)
   - :func:`clean_target_mask`        target-anchored mask cleanup
   - :func:`_clean_frames_bidirectional`  bidirectional target-anchored cleanup
   - :func:`merge_bidirectional`      fwd+bwd mask merge (pure, testable)
   - :func:`_select_evidence_frames`  review-frame selection
   - :func:`load_prompt_config`       box/points prompt validation (640x480)
   - :func:`partition_canopy_by_relative_height` (from ``phenocv.thermal.traits``)
   - :func:`resolve_layer_overlap`    (from ``phenocv.thermal.io``)
   - :func:`summarize_masked_temperature` (from ``phenocv.thermal.traits``)
2. GPU propagation layer (torch/sam2 imported lazily)
   - :func:`segment_video_with_sam2`  SAM 2 fwd+bwd, fed **thermal feature
     images** built by ``thermal_feature_image`` (absolute ΔT / gradient),
     not RGB.
3. Orchestration layer
   - :class:`ThermalVideoSegmenter`   one-segment run → masks/CSV/review overlay
   - :func:`run_segment`              CLI-friendly entry

Key engineering notes (hard-won, do not "simplify" away)
-------------------------------------------------------
* **Thermal feature input.** SAM 2 eats 3-channel images. For thermal we feed
  a feature image (absolute temp / local ΔT / gradient) built from the true
  temperature, never a pseudo-color frame. Temperature is read ONLY from the
  NPY true-temperature matrix.
* **Bidirectional propagation.** Forward (reference→end) + reverse
  (reference→start, time-reversed), merged with forward priority. Unidirectional
  propagation drifts near the sequence endpoints.
* **Target-anchored cleanup (post-propagation).** SAM 2 sometimes swallows an
  adjacent pot/soil/background plane, turning the whole-plant mask into a giant
  connected component 5–10× normal area. We keep only components touching the
  reference/positive points (reference frame) or the temporally-neighbouring
  anchor (propagated frames). Components with no anchor support are dropped.
* **Fail-closed.** A QC failure (hard fail from cleanup, or area out-of-bounds)
  must NOT be published: we write ``segment_failed_qc.json`` and raise.
* torch/sam2 are imported lazily: importing this module must NOT pull in CUDA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .io import (
    load_temperature,
    save_mask,
    thermal_feature_image,
    make_layer_overlay,
    resolve_layer_overlap,
)
from .traits import partition_canopy_by_relative_height, summarize_masked_temperature


logger = logging.getLogger(__name__)


__all__ = [
    "ThermalSegmentConfig",
    "clean_target_mask",
    "merge_bidirectional",
    "load_prompt_config",
    "segment_video_with_sam2",
    "ThermalVideoSegmenter",
    "run_segment",
]


# --------------------------------------------------------------------------
# Configuration
# 配置
# --------------------------------------------------------------------------

@dataclass
class ThermalSegmentConfig:
    """All tunable parameters for thermal temporal segmentation.

    热红外时序分割的全部可调参数。
    """

    # -- Image geometry (coordinate validation) --
    image_width: int = 640
    """Thermal frame width; prompt coords are validated against this."""
    image_height: int = 480
    """Thermal frame height; prompt coords are validated against this."""

    # -- Target-anchored cleanup --
    dilate_px: int = 10
    """Dilation radius (px) applied to the temporal anchor before overlap test."""
    hard_area_max_px: int = 30000
    """Hard upper bound on cleaned whole-plant area (background-engulf guard)."""
    min_whole_area_px: int = 200
    """Minimum cleaned whole-plant area; below this a frame is flagged empty."""

    # -- Overlay / export --
    vmin: float = 23.0
    """Fixed temperature-scale lower bound (°C) for review overlays."""
    vmax: float = 30.0
    """Fixed temperature-scale upper bound (°C) for review overlays."""
    jpeg_quality: int = 95

    # -- SAM 2 propagation --
    offload_video_to_cpu: bool = True
    bidirectional: bool = True

    # -- Fail-closed --
    qc_fail_raises: bool = True
    """When True, a QC failure raises instead of only writing the QC file."""

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "ThermalSegmentConfig":
        """Build from a YAML/dict, ignoring unknown keys (forward-compatible)."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Target-anchored mask cleanup (pure-CPU)
# 目标锚定掩膜清理（纯 CPU）
# --------------------------------------------------------------------------

def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel) > 0


def _bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def clean_target_mask(
    raw: np.ndarray,
    *,
    reference_points: Optional[Sequence[Sequence[float]]] = None,
    reference_labels: Optional[Sequence[int]] = None,
    reference_clean: Optional[np.ndarray] = None,
    prev_mask: Optional[np.ndarray] = None,
    is_reference: bool = False,
    dilate_px: int = 10,
    hard_area_max_px: int = 30000,
    alert_area_ratio_to_anchor: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clean a single whole-plant mask, returning ``(cleaned_bool, info)``.

    清理单帧整株掩膜，返回 ``(cleaned_bool, info)``。

    The cleanup keeps only components that are anchored to the target identity:

    * **Reference frame** — keep components containing a positive point; if a
      retained component also covers a negative point (plant merged with
      pot/soil), this is a **hard fail** (we do not silently crop it into a
      fake plant); do not force "keep largest component".
    * **Propagated frame** — keep only components overlapping the temporal
      anchor (the previous/next valid frame's supported target, dilated); drop
      isolated pots/soil/background blobs.
    * A box-only reference (no positive/negative points) degenerates to
      "keep largest component" and is flagged for multimodal confirmation;
      it is NOT a hard fail.

    Parameters
    ----------
    raw : raw whole-plant mask (boolean).
    reference_points / reference_labels : positive(1)/negative(0) points for the
        reference frame (used to judge component support).
    reference_clean : reference-frame cleaned mask (anchor for propagated frames).
    prev_mask : temporally-neighbouring cleaned mask (anchor for propagated
        frames; takes priority over ``reference_clean``).
    is_reference : treat ``raw`` as the reference frame.
    dilate_px : anchor dilation radius.
    hard_area_max_px : hard upper bound on cleaned area (engulf guard).
    alert_area_ratio_to_anchor : if cleaned area exceeds this ratio of the
        anchor area, flag as hard fail (suspected background engulf).

    Returns
    -------
    ``(cleaned_mask, info)`` where ``info`` carries auditable component-support
    records (``component_records``, ``hard_fail``, ``support_mode``, ...).
    """
    raw = np.asarray(raw, dtype=bool)
    info: Dict[str, Any] = {
        "component_count": 0,
        "positive_components": 0,
        "negative_in_target": False,
        "hard_fail": False,
        "anchor_overlap_pixels": 0,
        "cleaned_pixel_count": 0,
        "raw_component_count": 0,
        "retained_component_count": 0,
        "supported_component_count": 0,
        "dropped_unanchored_component_count": 0,
        "dropped_unanchored_pixels": 0,
        "support_mode": "none",
        "component_records": [],
    }
    if not raw.any():
        info["cleaned_pixel_count"] = 0
        return raw.copy(), info

    n, labels = cv2.connectedComponents(raw.astype(np.uint8))
    components = [labels == i for i in range(1, n)]
    info["component_count"] = max(0, n - 1)
    info["raw_component_count"] = max(0, n - 1)

    pts = list(reference_points or [])
    labs = list(reference_labels or [])
    pos_pts = [(float(x), float(y)) for (x, y), l in zip(pts, labs) if l == 1]
    neg_pts = [(float(x), float(y)) for (x, y), l in zip(pts, labs) if l == 0]

    def _count_hits(comp: np.ndarray, points) -> int:
        c = 0
        for x, y in points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < comp.shape[0] and 0 <= xi < comp.shape[1] and comp[yi, xi]:
                c += 1
        return c

    # Temporal anchor (used for non-reference frames).
    anchor = prev_mask if (prev_mask is not None) else reference_clean
    danchor = (
        _dilate(np.asarray(anchor, dtype=bool), dilate_px)
        if (anchor is not None and np.asarray(anchor, dtype=bool).any())
        else None
    )

    records: List[Dict[str, Any]] = []
    retained_mask = np.zeros_like(raw)
    supported_count = 0
    dropped_count = 0
    dropped_pixels = 0

    for i, comp in enumerate(components, start=1):
        area = int(comp.sum())
        pos_hits = _count_hits(comp, pos_pts)
        neg_hits = _count_hits(comp, neg_pts)
        anchor_overlap = (
            int(np.logical_and(comp, danchor).sum()) if danchor is not None else 0
        )
        rec: Dict[str, Any] = {
            "component_id": i,
            "area_pixels": area,
            "bbox_xyxy": _bbox(comp),
            "positive_point_hits": pos_hits,
            "negative_point_hits": neg_hits,
            "anchor_overlap_pixels": anchor_overlap,
            "anchor_overlap_fraction": (
                round(anchor_overlap / area, 4) if area else 0.0
            ),
            "retained": False,
            "decision_reason": "",
        }

        if is_reference:
            if pos_hits >= 1:
                rec["retained"] = True
                rec["decision_reason"] = "reference_positive_hit"
                supported_count += 1
            else:
                rec["retained"] = False
                rec["decision_reason"] = "reference_no_positive_dropped"
                dropped_count += 1
                dropped_pixels += area
        else:
            if anchor_overlap >= 1:
                rec["retained"] = True
                rec["decision_reason"] = "anchor_overlap"
                supported_count += 1
            else:
                rec["retained"] = False
                rec["decision_reason"] = "anchor_no_overlap_dropped"
                dropped_count += 1
                dropped_pixels += area
        records.append(rec)
        if rec["retained"]:
            retained_mask |= comp

    # Box-only reference (no positive/negative points): fall back to keeping the
    # largest component; flag for multimodal confirmation — NOT a hard fail.
    box_only_reference = is_reference and not pos_pts and not neg_pts

    if is_reference and not box_only_reference:
        neg_in_target = any(
            (r["retained"] and r["negative_point_hits"] >= 1) for r in records
        )
        info["positive_components"] = supported_count
        info["negative_in_target"] = neg_in_target
        any_positive = any(r["retained"] for r in records)
        if neg_in_target or not any_positive:
            info["hard_fail"] = True
            if not retained_mask.any() and components:
                retained_mask = max(components, key=lambda c: int(c.sum())).copy()
        info["support_mode"] = "reference_positive"
    elif box_only_reference:
        pass
    else:
        if anchor is not None and np.asarray(anchor, dtype=bool).any():
            info["anchor_overlap_pixels"] = int(
                np.logical_and(retained_mask, danchor).sum()
            )
            info["support_mode"] = "anchor_overlap"
            if (
                retained_mask.sum() > anchor.sum() * alert_area_ratio_to_anchor
                and anchor.sum() > 0
            ):
                info["hard_fail"] = True
        elif pos_pts:
            info["support_mode"] = "positive_fallback"
            if supported_count == 0:
                info["hard_fail"] = True
        else:
            info["support_mode"] = "max_component_no_anchor"
            if components and not retained_mask.any():
                retained_mask = max(components, key=lambda c: int(c.sum())).copy()

    if box_only_reference:
        if components:
            retained_mask = max(components, key=lambda c: int(c.sum())).copy()
            supported_count = 1
            dropped_count = len(components) - 1
            dropped_pixels = int(
                sum(int(c.sum()) for c in components) - int(retained_mask.sum())
            )
            records = [{
                "component_id": 1,
                "area_pixels": int(retained_mask.sum()),
                "bbox_xyxy": _bbox(retained_mask),
                "positive_point_hits": 0,
                "negative_point_hits": 0,
                "anchor_overlap_pixels": 0,
                "anchor_overlap_fraction": 0.0,
                "retained": True,
                "decision_reason": "box_only_max_component",
            }]
        info["support_mode"] = "box_only"

    # Alert when the cleaned area far exceeds the hard upper bound.
    if int(retained_mask.sum()) > int(hard_area_max_px) and hard_area_max_px > 0:
        info["hard_fail"] = True

    info["cleaned_pixel_count"] = int(retained_mask.sum())
    info["retained_component_count"] = int(sum(1 for r in records if r["retained"]))
    info["supported_component_count"] = int(supported_count)
    info["dropped_unanchored_component_count"] = int(dropped_count)
    info["dropped_unanchored_pixels"] = int(dropped_pixels)
    info["component_records"] = records
    info["unsupported_retained_pixels"] = 0
    return retained_mask, info


# --------------------------------------------------------------------------
# Bidirectional merge (pure-CPU, testable)
# 双向合并（纯 CPU，可测试）
# --------------------------------------------------------------------------

def merge_bidirectional(
    merged_fwd: Dict[int, np.ndarray],
    merged_bwd: Dict[int, np.ndarray],
    n: int,
    shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Merge forward + reverse propagation maps into ``{global_idx: mask}``.

    正向优先；逆向段仅在正向未覆盖或正向为空时填补；未覆盖帧记为全 False。

    Parameters
    ----------
    merged_fwd / merged_bwd : maps ``global_idx -> boolean mask``.
    n : total frame count.
    shape : ``(height, width)`` for the zero fallback mask.

    Returns
    -------
    ``{gi: bool mask}`` for ``gi in range(n)``.
    """
    result: Dict[int, np.ndarray] = {}
    for gi in range(n):
        f = merged_fwd.get(gi)
        if f is not None and np.asarray(f).any():
            result[gi] = np.asarray(f, dtype=bool)
        else:
            b = merged_bwd.get(gi)
            if b is not None:
                result[gi] = np.asarray(b, dtype=bool)
            else:
                result[gi] = np.zeros(shape, dtype=bool)
    return result


# --------------------------------------------------------------------------
# Prompt config (box / points / points_and_box), 640x480 validation
# 提示配置（640x480 坐标范围校验）
# --------------------------------------------------------------------------

def load_prompt_config(path: str | Path) -> Dict[str, Any]:
    """Load a single-segment prompt config (box / points / points_and_box).

    读取单段提示配置（box / points / points_and_box），并对坐标做 640x480
    范围校验。

    - ``box``: bounding-box only
    - ``points``: positive(1)/negative(0) points only
    - ``points_and_box``: box + real positive/negative points (negative points
      are passed to SAM 2)

    Returns
    -------
    dict with at least ``prompt_type`` and (as applicable) ``box``, ``points``,
    ``point_labels``, plus ``_path``.
    """
    p = Path(path).expanduser().resolve()
    import yaml

    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"提示配置必须是映射: {p}")
    pt = cfg.get("prompt_type", "box")
    if pt not in ("box", "points", "points_and_box"):
        raise ValueError(f"prompt_type 必须是 box/points/points_and_box: {pt}")

    W_IMG = 640
    H_IMG = 480

    def _check_xy(xy, where):
        if (
            not isinstance(xy, (list, tuple))
            or len(xy) != 2
            or not all(isinstance(v, (int, float)) for v in xy)
        ):
            raise ValueError(f"{where} 必须是 [x, y] 数值: {xy}")
        x, y = float(xy[0]), float(xy[1])
        if not (0 <= x <= W_IMG and 0 <= y <= H_IMG):
            raise ValueError(f"{where} 越界 (需 0..{W_IMG}, 0..{H_IMG}): {xy}")
        return [x, y]

    if pt in ("box", "points_and_box"):
        if "box" not in cfg:
            raise ValueError(f"{pt} 提示缺少 box 字段: {p}")
        box = cfg["box"]
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            raise ValueError(f"box 必须是 4 元素: {box}")
        cfg["box"] = [int(round(v)) for v in box]
        x1, y1, x2, y2 = cfg["box"]
        if not (0 <= x1 < x2 <= W_IMG and 0 <= y1 < y2 <= H_IMG):
            raise ValueError(f"box 坐标非法或越界: {cfg['box']}")

    if pt in ("points", "points_and_box"):
        if "points" not in cfg or "point_labels" not in cfg:
            raise ValueError(f"{pt} 提示缺少 points / point_labels: {p}")
        pts = cfg["points"]
        labels = cfg["point_labels"]
        if not isinstance(pts, (list, tuple)) or not isinstance(labels, (list, tuple)):
            raise ValueError(f"points / point_labels 必须是列表: {p}")
        if len(pts) != len(labels):
            raise ValueError(
                f"points 与 point_labels 长度不一致: {len(pts)} vs {len(labels)}"
            )
        if len(pts) == 0:
            raise ValueError("points 不能为空。")
        cfg["points"] = [_check_xy(xy, "points") for xy in pts]
        cfg["point_labels"] = [int(v) for v in labels]
        if set(cfg["point_labels"]) - {0, 1}:
            raise ValueError("point_labels 只能是 0 或 1。")
        if 1 not in cfg["point_labels"]:
            raise ValueError("points 中至少需要一个正点 (label=1)。")
    return {**cfg, "_path": p}


# --------------------------------------------------------------------------
# Evidence / review-frame selection (pure-CPU)
# 证据帧 / 审阅帧选择（纯 CPU）
# --------------------------------------------------------------------------

def _biological_segment_id(seg_id: str) -> str:
    return seg_id.replace("_dbg", "")


def _select_evidence_frames(
    seg_id: str,
    n: int,
    ref_idx: int,
    qc_rows: Sequence[Dict[str, Any]],
    whole_counts: Dict[int, int],
    evidence_policy_segment_id: Optional[str] = None,
) -> List[int]:
    """Select review frames: first / reference / last + anomalies + area
    extremes + segment-specific floor.

    证据帧选择：首/参考/末 + 异常帧 + 面积极值 + 段特定下限。
    """
    bio = _biological_segment_id(evidence_policy_segment_id or seg_id)
    s = {0, ref_idx, n - 1}
    for q in qc_rows:
        if q.get("qc_status") in ("REVIEW", "FAIL", "SEMANTIC_CHECK"):
            s.add(int(q["frame_index_global_in_segment"]))
    if whole_counts:
        min_gi = min(whole_counts, key=lambda g: whole_counts[g])
        max_gi = max(whole_counts, key=lambda g: whole_counts[g])
        s.update({min_gi, max_gi})
    if bio == "segment_04":
        s.update(range(n))
    target = 13 if bio in ("segment_03", "segment_05") else 0
    if len(s) < target:
        step = max(1, n // target)
        gi = 0
        while len(s) < target and gi < n:
            s.add(gi)
            gi += step
    return sorted(s)


def _sha256(path: Path) -> str:
    """Compute a file SHA-256 (no transformers dependency needed)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# GPU propagation layer (torch/sam2 imported lazily)
# GPU 传播层（torch/sam2 惰性导入）
# --------------------------------------------------------------------------

def _locate_sam2_config(model_size: str) -> str:
    """Locate the matching SAM 2 config inside the installed ``sam2`` package.

    在已安装的 sam2 包内查找对应配置；找不到则抛出。
    """
    size_to_suffix = {
        "small": "s", "base_plus": "b+", "large": "l",
        "tiny": "t", "s": "s", "b+": "b+", "l": "l", "t": "t",
    }
    suffix = size_to_suffix.get(model_size, model_size)
    try:
        import sam2

        base = Path(sam2.__file__).resolve().parent
        candidate = base / "configs" / "sam2" / f"sam2_hiera_{suffix}.yaml"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    raise FileNotFoundError(
        f"找不到 SAM2 配置 sam2_hiera_{suffix}.yaml（请检查 sam2 安装）。"
    )


def segment_video_with_sam2(
    frame_paths: Optional[Sequence[str]] = None,
    *,
    temperature_frames: Optional[Sequence[np.ndarray]] = None,
    temperature_paths: Optional[Sequence[str]] = None,
    reference_index: int = 0,
    prompt_box: Optional[Tuple[int, int, int, int]] = None,
    checkpoint_path: Optional[str] = None,
    model_size: str = "small",
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    config_path: Optional[str] = None,
    min_whole_area_px: int = 200,
    cache_dir: Optional[str] = None,
    prompt_points: Optional[Sequence[Sequence[float]]] = None,
    prompt_labels: Optional[Sequence[int]] = None,
    jpeg_quality: int = 95,
    bidirectional: bool = True,
) -> Dict[int, np.ndarray]:
    """Segment an ordered thermal sequence with SAM 2 video propagation.

    对有序热红外帧做 SAM2 视频分割，返回 ``{全局帧下标: bool 整株掩膜}``。

    Two passes — forward (reference→end) and reverse (reference→start,
    time-reversed) — are both rooted at the reference frame. The thermal input
    to SAM 2 is a 3-channel **feature image** built from the true temperature
    (absolute / local-ΔT / gradient) via
    :func:`phenocv.thermal.io.thermal_feature_image`, not RGB.

    Parameters
    ----------
    frame_paths : thermal-frame NPY paths (aliased to ``temperature_paths``).
    temperature_frames : pre-loaded temperature arrays (alternative to paths).
    temperature_paths : thermal-frame NPY paths (true temperature matrices).
    reference_index : index of the prompted reference frame in the sequence.
    prompt_box : ``(x0, y0, x1, y1)`` box prompt.
    checkpoint_path : SAM 2 checkpoint ``.pt`` (must be passed in; never
        hard-coded). 模型权重路径由参数传入，绝不硬编码。
    model_size : SAM 2 model size (small/base_plus/large/tiny).
    device : inference device (cuda/cpu/auto).
    prompt_points / prompt_labels : positive(1)/negative(0) points.
    bidirectional : run both forward and reverse passes when True.

    Returns
    -------
    ``{global_frame_index: boolean whole-plant mask}``.
    """
    n = None
    temps: List[np.ndarray] = []
    if temperature_frames is not None:
        temps = [np.asarray(t, dtype=float) for t in temperature_frames]
        n = len(temps)
    elif temperature_paths is not None or frame_paths is not None:
        paths = list(temperature_paths if temperature_paths is not None else frame_paths)
        temps = [load_temperature(p) for p in paths]
        n = len(temps)
    else:
        raise ValueError("必须提供 temperature_frames 或 temperature_paths。")

    if n == 0:
        return {}
    cfg = config_path or _locate_sam2_config(model_size)
    h, w = temps[0].shape[:2]

    fwd_seq = list(range(reference_index, n))            # reference → end
    bwd_seq = list(range(reference_index, -1, -1))       # reference → start (reversed)
    fwd_map = {local: g for local, g in enumerate(fwd_seq)}
    bwd_map = {local: g for local, g in enumerate(bwd_seq)}

    result: Dict[int, np.ndarray] = {}
    tmp_root = Path(cache_dir) if cache_dir else (
        Path(frame_paths[0]).resolve().parent / ".sam2_video_cache"
        if frame_paths else Path.cwd() / ".sam2_video_cache"
    )
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: lazy by design

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，SAM2 回退到 CPU。")
            device = "cpu"

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor

        _cfg_dir = os.path.dirname(str(cfg))
        _cfg_name = os.path.basename(str(cfg))
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=_cfg_dir, version_base=None):
            predictor = build_sam2_video_predictor(
                _cfg_name, ckpt_path=checkpoint_path, device=device
            )

        def _write_feature_frames(seq_indices, out_dir):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for local, g in enumerate(seq_indices):
                feat = thermal_feature_image(temps[g])
                cv2.imwrite(
                    str(out_dir / f"{local:05d}.jpg"),
                    feat,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )

        def _propagate(seq_indices, mapping):
            video_dir = tmp_root / ("fwd" if seq_indices is fwd_seq else "bwd")
            if video_dir.exists():
                import shutil

                shutil.rmtree(video_dir, ignore_errors=True)
            _write_feature_frames(seq_indices, video_dir)
            inference_state = predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=0,
                obj_id=1,
                box=list(prompt_box) if prompt_box is not None else None,
                points=(
                    np.asarray(prompt_points, dtype=np.float32)
                    if prompt_points is not None else None
                ),
                labels=(
                    np.asarray(prompt_labels, dtype=np.int32)
                    if prompt_labels is not None else None
                ),
            )
            out: Dict[int, np.ndarray] = {}
            for out_local, _obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                gi = mapping.get(int(out_local))
                if gi is None:
                    continue
                mask = np.asarray(mask_logits[0][0].cpu() > 0.0, dtype=bool)
                if gi not in out or not out[gi].any():
                    out[gi] = mask
            return out

        merged_fwd = _propagate(fwd_seq, fwd_map)
        if bidirectional:
            merged_bwd = _propagate(bwd_seq, bwd_map)
        else:
            merged_bwd = {}
        result = merge_bidirectional(merged_fwd, merged_bwd, n, (h, w))
    finally:
        if not cache_dir:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)
    return result


# --------------------------------------------------------------------------
# Orchestration: one-segment run
# 编排：单段运行
# --------------------------------------------------------------------------

def _clean_frames_bidirectional(
    merged_masks: Dict[int, np.ndarray],
    n: int,
    ref_global_idx: int,
    prompt_points,
    prompt_labels,
    ref_clean_mask: np.ndarray,
    dilate_px: int = 10,
    hard_area_max_px: int = 30000,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
    """Bidirectional target-anchored cleanup.

    双向目标锚定清理：
    - 正向 ``ref+1..end``：以 ``cleaned[gi-1]``（上一帧）为时间锚；
    - 逆向 ``ref-1..0``：以 ``cleaned[gi+1]``（下一时间相邻帧）为时间锚。

    Returns ``(cleaned_dict, info_dict)`` keyed by global frame index.
    """
    cleaned: Dict[int, np.ndarray] = {ref_global_idx: ref_clean_mask}
    infos: Dict[int, Dict[str, Any]] = {}
    for gi in range(ref_global_idx + 1, n):
        raw = merged_masks.get(gi, np.zeros((480, 640), dtype=bool))
        anchor = cleaned.get(gi - 1)
        whole, cinfo = clean_target_mask(
            raw, reference_points=prompt_points, reference_labels=prompt_labels,
            reference_clean=ref_clean_mask, prev_mask=anchor,
            is_reference=False, dilate_px=dilate_px, hard_area_max_px=hard_area_max_px,
        )
        cleaned[gi] = whole
        infos[gi] = cinfo
    for gi in range(ref_global_idx - 1, -1, -1):
        raw = merged_masks.get(gi, np.zeros((480, 640), dtype=bool))
        anchor = cleaned.get(gi + 1)
        whole, cinfo = clean_target_mask(
            raw, reference_points=prompt_points, reference_labels=prompt_labels,
            reference_clean=ref_clean_mask, prev_mask=anchor,
            is_reference=False, dilate_px=dilate_px, hard_area_max_px=hard_area_max_px,
        )
        cleaned[gi] = whole
        infos[gi] = cinfo
    return cleaned, infos


class ThermalVideoSegmenter:
    """Run one thermal segment: SAM 2 fwd+bwd → target-anchored cleanup →
    layering → temperature stats → masks / CSV / review overlay.

    单段运行：SAM2 视频 fwd+bwd 分割 → 逐帧目标锚定双向清理 → 分层 →
    温度统计 → 输出掩膜/CSV/审阅叠加图。
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_size: str = "small",
        device: str = "cuda",
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size
        self.device = device
        self.config_path = config_path
        self.cache_dir = cache_dir
        base = ThermalSegmentConfig.from_mapping(config)
        for k, v in kwargs.items():
            if k in ThermalSegmentConfig.__dataclass_fields__:
                setattr(base, k, v)
        self.cfg = base
        # Test hook: inject precomputed {gi: mask} to bypass torch/sam2.
        self._segment_fn = None

    # -- internal propagation ----------------------------------------------
    def _segment_video(self, temps, reference_index, prompt_box,
                       prompt_points, prompt_labels) -> Dict[int, np.ndarray]:
        if self._segment_fn is not None:
            return self._segment_fn(
                temps, reference_index, prompt_box, prompt_points, prompt_labels)
        return segment_video_with_sam2(
            temperature_frames=temps,
            reference_index=reference_index,
            prompt_box=prompt_box,
            checkpoint_path=self.checkpoint_path,
            model_size=self.model_size,
            device=self.device,
            offload_video_to_cpu=self.cfg.offload_video_to_cpu,
            config_path=self.config_path,
            cache_dir=self.cache_dir,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            jpeg_quality=self.cfg.jpeg_quality,
        )

    # -- public entry ------------------------------------------------------
    def run_segment(
        self,
        *,
        segment_id: str,
        stems: Sequence[str],
        temperature_frames: Optional[Sequence[np.ndarray]] = None,
        temperature_paths: Optional[Sequence[str]] = None,
        reference_stem: str,
        prompt_cfg: Dict[str, Any],
        output_dir: str | Path,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the full single-segment pipeline.

        Parameters
        ----------
        segment_id : segment identifier (e.g. ``"segment_01"``).
        stems : ordered frame stems (one per frame).
        temperature_frames / temperature_paths : true-temperature inputs.
        reference_stem : stem of the prompted reference frame.
        prompt_cfg : output of :func:`load_prompt_config`.
        output_dir : output root for this segment.
        vmin, vmax : overlay temperature-scale bounds (override config).

        Returns
        -------
        Run summary dict. On QC failure (when ``cfg.qc_fail_raises``) this
        writes ``segment_failed_qc.json`` and raises ``RuntimeError``.
        """
        vmin = vmin if vmin is not None else self.cfg.vmin
        vmax = vmax if vmax is not None else self.cfg.vmax
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(stems)
        if n == 0:
            raise ValueError("stems 不能为空。")
        try:
            ref_global_idx = list(stems).index(reference_stem)
        except ValueError as exc:
            raise FileNotFoundError(f"参考帧不在段内: {reference_stem}") from exc

        if temperature_frames is not None:
            temps = [np.asarray(t, dtype=float) for t in temperature_frames]
        elif temperature_paths is not None:
            temps = [load_temperature(p) for p in temperature_paths]
        else:
            raise ValueError("必须提供 temperature_frames 或 temperature_paths。")

        pt = prompt_cfg.get("prompt_type", "box")
        prompt_box = tuple(prompt_cfg["box"]) if "box" in prompt_cfg else None
        prompt_points = prompt_cfg.get("points")
        prompt_labels = prompt_cfg.get("point_labels")
        if prompt_box is None and prompt_points is not None:
            xs = [p[0] for p in prompt_points]
            ys = [p[1] for p in prompt_points]
            prompt_box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        t0 = time.time()
        merged_masks = self._segment_video(
            temps, ref_global_idx, prompt_box, prompt_points, prompt_labels)

        # Reference-frame cleanup (target-anchored baseline).
        ref_raw = merged_masks.get(
            ref_global_idx, np.zeros(temps[0].shape, dtype=bool))
        ref_clean_mask, _ref_info = clean_target_mask(
            ref_raw,
            reference_points=prompt_points,
            reference_labels=prompt_labels,
            reference_clean=None,
            prev_mask=None,
            is_reference=True,
            dilate_px=self.cfg.dilate_px,
            hard_area_max_px=self.cfg.hard_area_max_px,
        )

        cleaned_frames, cleanup_info = _clean_frames_bidirectional(
            merged_masks, n, ref_global_idx, prompt_points, prompt_labels,
            ref_clean_mask, dilate_px=self.cfg.dilate_px,
            hard_area_max_px=self.cfg.hard_area_max_px,
        )
        cleaned_frames[ref_global_idx] = ref_clean_mask
        cleanup_info[ref_global_idx] = _ref_info

        # ---- per-frame: layering + temperature + persistence ----
        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        qc_rows: List[Dict[str, Any]] = []
        all_overlays: List[Tuple[int, str, Image.Image]] = []
        whole_counts: Dict[int, int] = {}
        cleanup_records_all: Dict[str, Dict[str, Any]] = {}
        fail_frames: List[Dict[str, Any]] = []

        for gi in range(n):
            stem = stems[gi]
            whole = cleaned_frames[gi]
            cinfo = cleanup_info.get(gi, {})
            cleanup_records_all[stem] = cinfo
            temperature = temps[gi]

            # Layering (relative height), then enforce mutual exclusion.
            try:
                layers = partition_canopy_by_relative_height(whole)
            except ValueError:
                layers = {
                    "upper": whole.copy(),
                    "middle": whole.copy(),
                    "lower": whole.copy(),
                }
            resolved, _overlap_count = resolve_layer_overlap(layers, layers)

            union = np.logical_or.reduce(list(resolved.values()))
            overlap_px = sum(
                int(np.logical_and(resolved[a], resolved[b]).sum())
                for a, b in (
                    ("upper", "middle"),
                    ("upper", "lower"),
                    ("middle", "lower"),
                )
            )
            xor_whole = int(np.logical_xor(union, whole).sum())

            # Fail-closed QC gate: cleanup hard fail or area out-of-bounds.
            hard_fail = bool(cinfo.get("hard_fail", False))
            area = int(whole.sum())
            area_fail = (
                area > self.cfg.hard_area_max_px > 0
                or area < self.cfg.min_whole_area_px
            )
            frame_fail = hard_fail or area_fail
            qc_status = "FAIL" if frame_fail else "OK"

            pred_source = (
                "manual" if gi == ref_global_idx
                else ("propagated" if area > 0 else "failed_empty")
            )

            # Temperature statistics (whole + per-layer).
            temp_stats = summarize_masked_temperature(temperature, whole)
            row: Dict[str, Any] = {
                "segment_id": segment_id,
                "frame_stem": stem,
                "frame_index_global_in_segment": gi,
                "pred_source": pred_source,
                "whole_pixel_count": area,
                "overall_qc": qc_status,
                **temp_stats,
                "layer_union_xor_whole_pixels": xor_whole,
                "layer_overlap_pixels": overlap_px,
            }
            for layer, lmask in resolved.items():
                lv = summarize_masked_temperature(temperature, lmask)
                row.update({f"{layer}_{key}": val for key, val in lv.items()})
            rows.append(row)

            qc_row = {
                "segment_id": segment_id,
                "frame_stem": stem,
                "frame_index_global_in_segment": gi,
                "pred_source": pred_source,
                "qc_status": qc_status,
                "cleanup_hard_fail": hard_fail,
                "whole_pixel_count": area,
                "component_count": int(cinfo.get("component_count", 0)),
            }
            qc_rows.append(qc_row)
            whole_counts[gi] = area

            if frame_fail:
                fail_frames.append({
                    "frame_stem": stem,
                    "frame_index": gi,
                    "qc_status": qc_status,
                    "cleanup_hard_fail": hard_fail,
                    "overlay_path": str(overlays_dir / f"{stem}_layers.png"),
                })

            # Persist masks.
            save_mask(whole, masks_dir / "whole" / f"{stem}.png")
            for layer, lmask in resolved.items():
                save_mask(lmask, masks_dir / layer / f"{stem}.png")

            # Review overlay.
            overlay = make_layer_overlay(temperature, resolved, vmin, vmax)
            ov_path = overlays_dir / f"{stem}_layers.png"
            cv2.imwrite(str(ov_path), overlay)
            all_overlays.append((gi, stem, Image.fromarray(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))))

        # Write cleanup component-support records.
        (output_dir / "cleanup_component_records.json").write_text(
            json.dumps({
                "segment_id": segment_id,
                "reference_stem": reference_stem,
                "records": cleanup_records_all,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write metrics + qc CSVs.
        if rows:
            with (output_dir / f"{segment_id}_thermal_metrics.csv").open(
                "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        if qc_rows:
            with (output_dir / f"{segment_id}_qc.csv").open(
                "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(qc_rows[0]))
                w.writeheader()
                w.writerows(qc_rows)

        # Evidence frames → review contact sheet.
        review_gis = _select_evidence_frames(
            segment_id, n, ref_global_idx, qc_rows, whole_counts,
            evidence_policy_segment_id=segment_id)
        review_images = [
            (stems[gi], ov) for gi, stem, ov in all_overlays if gi in review_gis]
        review_path = output_dir / "review_sheet.png"
        if review_images:
            cols = min(5, len(review_images))
            nrows = (len(review_images) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 640, nrows * 480), "white")
            for i, (_stem, img) in enumerate(review_images):
                sheet.paste(img, ((i % cols) * 640, (i // cols) * 480))
            sheet.save(review_path)

        # Fail-closed: never publish on QC failure.
        if fail_frames and self.cfg.qc_fail_raises:
            failed = {
                "segment_id": segment_id,
                "status": "FAILED_QC",
                "reference_stem": reference_stem,
                "fail_frame_count": len(fail_frames),
                "anomalies": fail_frames,
            }
            (output_dir / "segment_failed_qc.json").write_text(
                json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.error(
                "[%s] 存在 %d 个 QC 失败帧，不发布（fail-closed）。",
                segment_id, len(fail_frames))
            raise RuntimeError(
                f"[{segment_id}] QC 失败：{len(fail_frames)} 帧未通过，"
                f"已写 segment_failed_qc.json。")

        summary = {
            "segment_id": segment_id,
            "status": "OK",
            "backend": "official_sam2",
            "model_size": self.model_size,
            "checkpoint_path": self.checkpoint_path,
            "prompt_type": pt,
            "reference_stem": reference_stem,
            "device": self.device,
            "frame_count_actual": n,
            "reference_index": ref_global_idx,
            "review_sheet_frame_count": len(review_gis),
            "review_sheet_sha256": (
                _sha256(review_path) if review_images else ""),
            "all_layers_disjoint": all(r["layer_overlap_pixels"] == 0 for r in rows),
            "all_unions_exact": all(r["layer_union_xor_whole_pixels"] == 0 for r in rows),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        (output_dir / "segment_complete.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%s] 完成: %d 帧, 总耗时 %.1fs", segment_id, n, time.time() - t0)
        return summary


# --------------------------------------------------------------------------
# CLI-friendly entry
# CLI 友好入口
# --------------------------------------------------------------------------

def run_segment(
    *,
    segment_id: str,
    stems: Sequence[str],
    temperature_frames: Optional[Sequence[np.ndarray]] = None,
    temperature_paths: Optional[Sequence[str]] = None,
    reference_stem: str,
    prompt_cfg: Dict[str, Any],
    output_dir: str | Path,
    checkpoint_path: Optional[str] = None,
    model_size: str = "small",
    device: str = "cuda",
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience wrapper around :class:`ThermalVideoSegmenter`.

    ``ThermalVideoSegmenter`` 的便捷封装。模型权重路径由参数传入，绝不硬编码。
    """
    seg = ThermalVideoSegmenter(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        device=device,
        config_path=config_path,
        config=config,
        cache_dir=cache_dir,
        **kwargs,
    )
    return seg.run_segment(
        segment_id=segment_id,
        stems=stems,
        temperature_frames=temperature_frames,
        temperature_paths=temperature_paths,
        reference_stem=reference_stem,
        prompt_cfg=prompt_cfg,
        output_dir=output_dir,
        vmin=vmin,
        vmax=vmax,
    )
