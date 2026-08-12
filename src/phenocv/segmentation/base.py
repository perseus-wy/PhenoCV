# -*- coding: utf-8 -*-
"""Segmentation backend abstraction — the design-redundancy extension point.

PhenoCV's temporal canopy segmentation is built around one data contract:
a :class:`~phenocv.segmentation.engine.PlantSequence` (frames + sparse anchor
masks) goes in, a full-sequence mask result dict comes out. The *how* — SAM 2
video propagation today, a classical CV pipeline or a YOLO mask model
tomorrow — is deliberately swappable behind :class:`BaseSegmenter`.

Why a backend hierarchy (design redundancy)
-------------------------------------------
* Future contributors add a new segmentation algorithm by subclassing
  :class:`BaseSegmenter` and registering it in :func:`build_segmenter`. They
  never touch the orchestration, the adapters, or the CSV/ISAT exporters.
* The SAM 2 backend keeps torch/sam2 **lazily** imported; importing this
  module (or ``phenocv.segmentation``) must not pull in CUDA.

Backends
--------
* :class:`SAM2Segmenter` — the production temporal-propagation backend (delegates
  to :func:`~phenocv.segmentation.engine.run_sam2_video_temporal`, behavior
  preserved bit-for-bit).
* :class:`ClassicalSegmenter` / :class:`YOLOSegmenter` — placeholder backends
  that document the intended interface and raise ``NotImplementedError`` until
  implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

# engine stays the single source of truth for the SAM 2 logic; importing it
# does not pull in torch (cv2 only). No circular import: engine never imports
# this module.
from .engine import (
    PlantSequence,
    run_sam2_video_temporal,
    DEFAULT_SAM2_CONFIG,
)


class BaseSegmenter(ABC):
    """Abstract segmentation backend. 抽象分割后端基类。

    Every backend consumes the same inputs and returns the same result shape,
    so the rest of PhenoCV (adapters, exporters, CLI) is backend-agnostic.

    Contract
    --------
    ``run`` consumes a sequence of
    :class:`~phenocv.segmentation.engine.PlantSequence` objects plus an output
    root and backend-specific options, and returns the result dict produced by
    the engine (``summary`` / ``loo_summary`` / CSV paths, etc.).
    """

    #: Backend identifier used by :func:`build_segmenter` and the
    #: ``backend:`` config field. Subclasses override this.
    backend_name: str = "base"

    @abstractmethod
    def run(self,
            sequences: Sequence[PlantSequence],
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
        """Run segmentation over ``sequences`` -> result dict.

        运行分割：输入时序序列与输出根目录，返回结果字典。
        """
        raise NotImplementedError


class SAM2Segmenter(BaseSegmenter):
    """Production temporal-propagation backend (SAM 2 video predictor).

    生产级时序传播后端，基于 SAM 2 视频预测器。

    The SAM 2 logic lives in
    :func:`~phenocv.segmentation.engine.run_sam2_video_temporal`; this class is
    the canonical, backend-aware wrapper around it. torch/sam2 are imported
    lazily inside the engine, so constructing or importing this class never
    touches CUDA.
    """

    backend_name = "sam2"

    def run(self,
            sequences: Sequence[PlantSequence],
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
        return run_sam2_video_temporal(
            sequences, output_root, checkpoint,
            model_cfg=model_cfg, device=device, cache_root=cache_root,
            config=config, do_loo=do_loo, export_isat=export_isat,
            export_qa=export_qa, resume=resume,
            image_size=image_size, progress=progress)


class ClassicalSegmenter(BaseSegmenter):
    """Placeholder classical-CV segmentation backend. 传统视觉分割后端（占位）。

    Intended interface (not yet implemented)
    ---------------------------------------
    A pure-CPU / OpenCV pipeline — e.g. HSV/LAB thresholding + watershed or
    GrabCut seeded by the anchor masks — that satisfies the same
    :meth:`BaseSegmenter.run` contract. Swap it in via ``backend: classical``
    without touching the orchestration or exporters.

    Implementing this is intentionally left as future work; callers that select
    it must receive a clear signal rather than a silent no-op.
    """

    backend_name = "classical"

    def run(self,
            sequences: Sequence[PlantSequence],
            output_root: str,
            checkpoint: str = "",
            model_cfg: str = "",
            device: str = "cpu",
            cache_root: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            do_loo: bool = True,
            export_isat: bool = True,
            export_qa: bool = True,
            resume: bool = True,
            image_size: Tuple[int, int] = (720, 1280),
            progress=None) -> Dict[str, Any]:
        raise NotImplementedError(
            "ClassicalSegmenter is a placeholder backend. Implement a pure-CPU "
            "OpenCV pipeline (thresholding + watershed / GrabCut seeded by "
            "anchor masks) satisfying BaseSegmenter.run, then register it under "
            "backend='classical'.")


class YOLOSegmenter(BaseSegmenter):
    """Placeholder YOLO-instance-segmentation backend. YOLO 实例分割后端（占位）。

    Intended interface (not yet implemented)
    ---------------------------------------
    A per-frame instance-segmentation model (e.g. Ultralytics YOLO-seg or a
    Mask R-CNN) that produces per-plant canopy masks and satisfies the same
    :meth:`BaseSegmenter.run` contract. Swap it in via ``backend: yolo``.

    Implementing this is intentionally left as future work; callers that select
    it must receive a clear signal rather than a silent no-op.
    """

    backend_name = "yolo"

    def run(self,
            sequences: Sequence[PlantSequence],
            output_root: str,
            checkpoint: str = "",
            model_cfg: str = "",
            device: str = "cpu",
            cache_root: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            do_loo: bool = True,
            export_isat: bool = True,
            export_qa: bool = True,
            resume: bool = True,
            image_size: Tuple[int, int] = (720, 1280),
            progress=None) -> Dict[str, Any]:
        raise NotImplementedError(
            "YOLOSegmenter is a placeholder backend. Implement a per-frame "
            "instance-segmentation model (Ultralytics YOLO-seg / Mask R-CNN) "
            "producing per-plant canopy masks and satisfying "
            "BaseSegmenter.run, then register it under backend='yolo'.")


def build_segmenter(backend: str = "sam2", **kwargs: Any) -> BaseSegmenter:
    """Construct a segmentation backend by name. 按名称构建分割后端。

    Parameters
    ----------
    backend : one of ``"sam2"`` (default), ``"classical"``, ``"yolo"``.
    kwargs : forwarded to the backend constructor (e.g. device, defaults).
    """
    name = (backend or "sam2").lower()
    if name == "sam2":
        return SAM2Segmenter(**kwargs)
    if name == "classical":
        return ClassicalSegmenter(**kwargs)
    if name == "yolo":
        return YOLOSegmenter(**kwargs)
    raise ValueError(
        "Unknown segmentation backend %r (available: sam2, classical, yolo)"
        % backend)


def run_segmentation(sequences: Sequence[PlantSequence],
                     output_root: str,
                     checkpoint: str,
                     backend: str = "sam2",
                     config: Optional[Dict[str, Any]] = None,
                     model_cfg: str = DEFAULT_SAM2_CONFIG,
                     device: str = "cuda",
                     **kwargs: Any) -> Dict[str, Any]:
    """Dispatch to the chosen backend and run it. 选择后端并运行分割。

    The backend is taken from the ``backend`` argument, falling back to a
    ``backend`` key inside ``config`` (so a single YAML config selects the
    algorithm). Defaults to ``"sam2"`` — preserving existing behavior.
    """
    cfg = dict(config or {})
    chosen = cfg.get("backend", backend) or "sam2"
    segmenter = build_segmenter(chosen)
    return segmenter.run(
        sequences, output_root, checkpoint,
        model_cfg=model_cfg, device=device, config=config, **kwargs)
