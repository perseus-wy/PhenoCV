# -*- coding: utf-8 -*-
"""``phenocv.thermal`` — a pure-CPU thermal (FLIR) phenotyping module.

This module ports a private thermal-infrared pipeline into PhenoCV as a clean,
data-agnostic, fail-closed toolkit. It adds a third input tier to the shared
``phenocv.core`` registry (``thermal`` + ``ambient`` temperature alongside the
existing ``mask``) and provides four cooperating submodules:

* :mod:`phenocv.thermal.io` — temperature / thermal-feature / mask IO and
  cv2-only overlays (no matplotlib).
* :mod:`phenocv.thermal.traits` — a registry-driven, fail-closed trait engine
  (canopy temperature, upper/middle/lower layer temperature, canopy ΔT).
* :mod:`phenocv.thermal.environment` — align an environment time-series onto
  frame timestamps by linear interpolation (no extrapolation; gap-guarded).
* :mod:`phenocv.thermal.stress` — before/after stress / rewatering response
  analysis with block-bootstrap & HAC uncertainty, plus generic roll-ups.
* :mod:`phenocv.thermal.segmentation` — optional SAM 2 temporal thermal
  segmentation (torch/sam2 imported lazily; never at import time).

Design contract (shared with ``phenocv.phenotypes``)
----------------------------------------------------
* **Pure CPU** — only ``numpy`` + ``cv2`` + ``pandas`` + ``scipy`` +
  ``statsmodels`` (the last two lazy-imported). No ``torch`` / ``sam2`` /
  ``matplotlib`` at import time, so the core is importable and testable without
  a GPU or a plotting backend.
* **Fail-closed** — missing / unobservable / empty inputs return ``NaN`` +
  ``missing_reason`` (or ``<name>_error``), never a fabricated value.
* **Data-agnostic** — core logic never reads a specific dataset layout; paths
  and column maps are supplied by the caller (adapter / CLI).

``phenocv.thermal`` —— 纯 CPU 热红外（FLIR）表型模块。

本模块把一个私有热红外流水线脱敏移植为 PhenoCV 中干净、数据无关、fail-closed 的
工具包，并向共享 registry 增加第三类输入（thermal + ambient 温度，外加既有 mask）。
四个子模块：io / traits / environment / stress（+ 可选的 segmentation）。设计契约
对齐 phenocv.phenotypes：纯 CPU、fail-closed、数据无关。除零一律走安全除法，禁止
编造数值。
"""

from __future__ import annotations

from .config import ThermalConfig, load_thermal_config
from .io import (
    load_temperature,
    load_thermal_meta,
    thermal_feature_image,
    resolve_layer_overlap,
    make_overlay,
    make_layer_overlay,
    polygons_to_mask,
    save_mask,
    load_mask,
    _robust_normalize,
    _ellipse_kernel,
    _dilate,
    _erode,
)
from .traits import (
    INPUT_THERMAL,
    INPUT_AMBIENT,
    summarize_masked_temperature,
    partition_canopy_by_relative_height,
    CanopyTemperatureExtractor,
    LayerTemperatureExtractor,
    CanopyDeltaTExtractor,
    compute_thermal_traits,
)
from .environment import (
    InterpolationResult,
    parse_flir_stem,
    prepare_sensor_frame,
    interpolate_sensor_frame,
    read_sensor_workbook,
    read_environment_workbook,
    EnvironmentJoiner,
    align_environment_to_frames,
)
from .stress import (
    vapour_pressure_deficit,
    moving_block_bootstrap_ci,
    block_bootstrap_difference,
    hac_mean_ci,
    phase_contrast,
    hac_adjusted_regression,
    recovery_kinetics,
    analyze_stress_response,
    summarize_plants,
    summarize_paired_differences,
    detect_light_transitions,
    calculate_layer_correlations,
    pair_shifted_times,
)
from .segmentation import (
    ThermalSegmentConfig,
    clean_target_mask,
    merge_bidirectional,
    load_prompt_config,
    segment_video_with_sam2,
    ThermalVideoSegmenter,
    run_segment,
)

__all__ = [
    # config
    "ThermalConfig",
    "load_thermal_config",
    # io
    "load_temperature",
    "load_thermal_meta",
    "thermal_feature_image",
    "resolve_layer_overlap",
    "make_overlay",
    "make_layer_overlay",
    "polygons_to_mask",
    "save_mask",
    "load_mask",
    "_robust_normalize",
    "_ellipse_kernel",
    "_dilate",
    "_erode",
    # traits
    "INPUT_THERMAL",
    "INPUT_AMBIENT",
    "summarize_masked_temperature",
    "partition_canopy_by_relative_height",
    "CanopyTemperatureExtractor",
    "LayerTemperatureExtractor",
    "CanopyDeltaTExtractor",
    "compute_thermal_traits",
    # environment
    "InterpolationResult",
    "parse_flir_stem",
    "prepare_sensor_frame",
    "interpolate_sensor_frame",
    "read_sensor_workbook",
    "read_environment_workbook",
    "EnvironmentJoiner",
    "align_environment_to_frames",
    # stress
    "vapour_pressure_deficit",
    "moving_block_bootstrap_ci",
    "block_bootstrap_difference",
    "hac_mean_ci",
    "phase_contrast",
    "hac_adjusted_regression",
    "recovery_kinetics",
    "analyze_stress_response",
    "summarize_plants",
    "summarize_paired_differences",
    "detect_light_transitions",
    "calculate_layer_correlations",
    "pair_shifted_times",
    # segmentation (optional; torch/sam2 lazy)
    "ThermalSegmentConfig",
    "clean_target_mask",
    "merge_bidirectional",
    "load_prompt_config",
    "segment_video_with_sam2",
    "ThermalVideoSegmenter",
    "run_segment",
]
