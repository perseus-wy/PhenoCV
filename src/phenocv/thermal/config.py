# -*- coding: utf-8 -*-
"""Thermal (FLIR) module configuration — YAML-loadable, segmentation-style.

Thermal config governs the shared conventions used by every ``phenocv.thermal``
submodule: the fixed display temperature scale, the max gap allowed when
interpolating environment sensors onto frame timestamps, how a whole-canopy
mask is partitioned into vertical layers, and the uncertainty-estimation
defaults used by the stress/rewatering analysis.

The YAML layout mirrors ``configs/default.yaml``::

    thermal:
      temperature_vmin_c: 22.0
      temperature_vmax_c: 29.0
      interpolation_max_gap_sec: 600.0
      ...
    presets:
      potted_soybean:
        ...

``load_thermal_config`` returns a flat :class:`ThermalConfig` (a dataclass),
ready to thread through ``compute_thermal_traits`` / ``analyze_stress_response``.
No desensitisation needed here — these are pure methodology defaults.

热红外模块配置 —— 可经 YAML 加载，风格对齐 segmentation 模块。

本配置统一 ``phenocv.thermal`` 各子模块的约定：固定温度显示温标、将环境传感器
插值到帧时刻时允许的最大时间缺口、整株掩膜如何划分为垂直分层，以及胁迫/复水
分析中不确定性估计的默认超参。所有字段均为通用方法论默认值，不含任何私人数据。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ThermalConfig:
    """Flat, YAML-loadable configuration for the thermal module.

    热红外模块的扁平可加载配置。
    """

    # Fixed display temperature scale (°C) used by overlays / normalization.
    # 叠加图与归一化所用的固定温度显示温标（°C）。
    temperature_vmin_c: float = 22.0
    temperature_vmax_c: float = 29.0

    # Environment-sensor interpolation: reject gaps larger than this (no extrapolation).
    # 环境传感器插值：超过该缺口的帧返回 NaN + qc_flag（禁止外推）。
    interpolation_max_gap_sec: float = 600.0

    # Whole-canopy vertical partition strategy.
    # 整株垂直分层策略（当前仅支持按相对高度三等分）。
    layer_partition: str = "relative_height_thirds"

    # Semantic column name used as the Delta-T reference ambient temperature.
    # 作为 ΔT 参考环境温度使用的语义列名。
    delta_t_reference_column: str = "ambient_c"

    # Uncertainty-estimation defaults for the stress/rewatering analysis.
    # 胁迫/复水分析中不确定性估计的默认超参。
    block_bootstrap_iterations: int = 4000
    block_bootstrap_block_size: int = 12
    hac_max_lags: int = 12
    light_phase_threshold_lux: float = 50.0
    random_seed: int = 20260728

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ThermalConfig":
        """Build a config from a flat dict, ignoring unknown keys.

        从扁平字典构建配置，忽略未知键（向前兼容）。
        """
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


def load_thermal_config(
    path: Optional[str] = None,
    preset: Optional[str] = None,
) -> ThermalConfig:
    """Load a :class:`ThermalConfig` from YAML (``thermal`` block + optional preset).

    Parameters
    ----------
    path : YAML file path. When ``None`` (and no ``preset``), the built-in
        defaults are returned. 配置 YAML 路径；为 ``None`` 时返回内置默认。
    preset : preset name under the ``presets`` block, overlaid on ``thermal``.

    Returns
    -------
    ThermalConfig
    """
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        base = dict(raw.get("thermal", {}))
        presets = raw.get("presets", {}) or {}
        if preset:
            if preset not in presets:
                raise KeyError(
                    "Unknown thermal preset %r (available: %s)"
                    % (preset, list(presets))
                )
            base.update(presets[preset])
        data = base
    elif preset:
        raise FileNotFoundError(
            "Cannot apply preset %r without a config path." % preset
        )
    return ThermalConfig.from_mapping(data)
