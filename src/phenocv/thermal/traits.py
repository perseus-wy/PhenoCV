# -*- coding: utf-8 -*-
"""Thermal trait engine — a fail-closed, registry-driven trait extractor.

This mirrors ``phenocv.phenotypes`` but for thermal (FLIR) inputs. Three input
tags are recognised (aligned with ``phenocv.core.registry``'s INPUT_MASK):

  * ``INPUT_MASK``    — canopy/plant boolean mask
  * ``INPUT_THERMAL`` — 2D absolute-temperature matrix (°C, float)
  * ``INPUT_AMBIENT`` — reference ambient (air) temperature for the frame (scalar °C)

Every extractor declares its ``requires``; ``compute_thermal_traits`` runs only
the extractors whose inputs you actually pass, in tier order, and merges the
columns. Like the phenotype engine it is **fail-closed**: unobservable outputs
are ``NaN`` + a ``missing_reason`` (or ``<name>_error`` from a thrown extractor),
never a fabricated value.

热表型引擎 —— 受 registry 驱动的 fail-closed 表型提取器。

本模块对齐 ``phenocv.phenotypes``，但面向热红外（FLIR）输入。识别三种输入标签
（与 ``phenocv.core.registry`` 的 INPUT_MASK 对齐）：INPUT_MASK / INPUT_THERMAL /
INPUT_AMBIENT。编排器只运行满足 requires 的提取器，并 fail-closed 地返回 NaN +
missing_reason（或对抛错的提取器记录 ``<name>_error``），绝不编造数值。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phenocv.core.registry import (
    INPUT_MASK,
    TraitExtractor,
    register,
    available_for,
)

# --- Thermal input tags (aligned with phenotypes.INPUT_*) / 输入标签 --------
INPUT_THERMAL = "thermal"
INPUT_AMBIENT = "ambient"


# --------------------------------------------------------------------------
# Core statistic: robust masked-temperature summary / 核心温度统计
# --------------------------------------------------------------------------
def summarize_masked_temperature(
    temperature: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Robust temperature statistics inside a mask.

    Returns ``temp_median_c``, ``temp_mean_c``, ``temp_p10_c``, ``temp_p90_c``,
    ``temp_std_c`` and ``pixel_count`` over finite mask pixels. An empty mask (or
    one with no finite temperature) returns all-``NaN`` + ``pixel_count = 0`` —
    fail-closed, never raises.

    返回掩膜内有限温度像素的稳健统计：中位数/均值/p10/p90/标准差与像素数。空掩膜
    （或掩膜内无有限温度）返回全 NaN + pixel_count=0，fail-closed，不抛异常。
    """
    temp = np.asarray(temperature, dtype=float)
    m = np.asarray(mask, dtype=bool)
    selected = temp[m]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "temp_median_c": float("nan"),
            "temp_mean_c": float("nan"),
            "temp_p10_c": float("nan"),
            "temp_p90_c": float("nan"),
            "temp_std_c": float("nan"),
            "pixel_count": 0,
        }
    return {
        "temp_median_c": float(np.median(selected)),
        "temp_mean_c": float(np.mean(selected)),
        "temp_p10_c": float(np.percentile(selected, 10)),
        "temp_p90_c": float(np.percentile(selected, 90)),
        "temp_std_c": float(np.std(selected)),
        "pixel_count": int(selected.size),
    }


def partition_canopy_by_relative_height(
    whole_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Split a whole-canopy mask into upper/middle/lower thirds by relative height.

    The vertical bounding box of the whole mask is divided into three equal
    bands (top → bottom). Raises ``ValueError`` if the mask is empty or too
    short to form three non-empty bands; the orchestrator records that as
    ``layer_temperature_error`` (fail-closed at the extractor boundary).

    按整株掩膜垂直包围盒的相对高度三等分（上/中/下）。掩膜为空或垂直跨度不足以
    形成三个非空层时抛出 ValueError；编排器将其记为 ``layer_temperature_error``
    （在提取器级 fail-closed）。
    """
    whole = np.asarray(whole_mask, dtype=bool)
    if whole.ndim != 2 or not whole.any():
        raise ValueError("Whole-canopy mask must be a non-empty 2D array.")
    y_coordinates = np.flatnonzero(whole.any(axis=1))
    y_min = int(y_coordinates.min())
    y_max = int(y_coordinates.max())
    height = y_max - y_min + 1
    row_grid = np.arange(whole.shape[0])[:, None]
    relative_height = (row_grid - y_min) / height
    upper = whole & (relative_height < 1.0 / 3.0)
    middle = whole & (relative_height >= 1.0 / 3.0) & (relative_height < 2.0 / 3.0)
    lower = whole & (relative_height >= 2.0 / 3.0)
    layers = {"upper": upper, "middle": middle, "lower": lower}
    if any(not layer.any() for layer in layers.values()):
        raise ValueError(
            "Whole-canopy vertical span too small to form three non-empty layers."
        )
    return layers


def _prefix_dict(source: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Prefix every key in ``source`` (e.g. ``temp_median_c`` -> ``canopy_temp_median_c``)."""
    return {"%s%s" % (prefix, k): v for k, v in source.items()}


# --------------------------------------------------------------------------
# Extractors / 提取器
# --------------------------------------------------------------------------
@register
class CanopyTemperatureExtractor(TraitExtractor):
    name = "canopy_temperature"
    description = "Whole-canopy robust temperature statistics (median/mean/p10/p90/std)."
    requires = [INPUT_MASK, INPUT_THERMAL]
    tier = 1

    def extract(self, *, mask=None, temperature=None, **ctx):
        stats = summarize_masked_temperature(temperature, mask)
        return _prefix_dict(stats, "canopy_")


@register
class LayerTemperatureExtractor(TraitExtractor):
    name = "layer_temperature"
    description = (
        "Upper/middle/lower canopy temperature from relative-height partitioning."
    )
    requires = [INPUT_MASK, INPUT_THERMAL]
    tier = 2

    def extract(self, *, mask=None, temperature=None, **ctx):
        layers = partition_canopy_by_relative_height(mask)
        out: Dict[str, float] = {}
        medians = {}
        for layer, layer_mask in layers.items():
            stats = summarize_masked_temperature(temperature, layer_mask)
            out.update(_prefix_dict(stats, "canopy_%s_" % layer))
            medians[layer] = stats["temp_median_c"]
        ordered = [medians[k] for k in ("upper", "middle", "lower")]
        finite = [v for v in ordered if np.isfinite(v)]
        out["canopy_layer_temperature_range_c"] = (
            float(np.max(finite) - np.min(finite)) if finite else float("nan")
        )
        out["canopy_upper_minus_lower_c"] = (
            medians["upper"] - medians["lower"]
            if np.isfinite(medians["upper"]) and np.isfinite(medians["lower"])
            else float("nan")
        )
        return out


@register
class CanopyDeltaTExtractor(TraitExtractor):
    name = "canopy_delta_t"
    description = "Canopy minus ambient temperature (ΔT). NaN + missing_reason when ambient is missing/non-finite."
    requires = [INPUT_MASK, INPUT_THERMAL, INPUT_AMBIENT]
    tier = 1

    def extract(self, *, mask=None, temperature=None, ambient=None, **ctx):
        if ambient is None or not np.isfinite(float(ambient)):
            return {
                "canopy_delta_t_c": float("nan"),
                "canopy_delta_t_mean_c": float("nan"),
                "missing_reason": "ambient_temperature_missing",
            }
        stats = summarize_masked_temperature(temperature, mask)
        median = stats["temp_median_c"]
        mean = stats["temp_mean_c"]
        a = float(ambient)
        return {
            "canopy_delta_t_c": median - a,
            "canopy_delta_t_mean_c": mean - a,
        }


# --------------------------------------------------------------------------
# Orchestrator / 编排器
# --------------------------------------------------------------------------
def compute_thermal_traits(
    *,
    mask=None,
    temperature=None,
    ambient=None,
    **ctx,
) -> Dict[str, Any]:
    """Compute every applicable thermal trait for one (plant, frame).

    Parameters
    ----------
    mask : np.ndarray
        Canopy/plant boolean (or 0/1) mask in image coordinates.
    temperature : np.ndarray
        2D absolute-temperature matrix (°C, float). 二维绝对温度矩阵（°C）。
    ambient : float, optional
        Reference ambient (air) temperature for this frame. When ``None`` or
        non-finite, the ΔT extractor emits ``NaN`` + ``missing_reason`` instead
        of fabricating a value. 本帧参考环境温度；缺失或非有限时 ΔT 输出 NaN。
    **ctx : extra config
        Passed through to every extractor (e.g. a :class:`ThermalConfig`).

    Returns
    -------
    dict
        Merged trait columns + ``_inputs`` (sorted available input tags) and
        ``_extractors_run`` (names). Fail-closed: a thrown extractor is
        recorded under ``<name>_error`` rather than aborting the row.
    """
    available = set()
    if mask is not None:
        available.add(INPUT_MASK)
    if temperature is not None:
        available.add(INPUT_THERMAL)
    if ambient is not None:
        available.add(INPUT_AMBIENT)

    extractors: List[TraitExtractor] = available_for(available)
    row: Dict[str, Any] = {}
    for ext in extractors:
        try:
            out = ext.extract(
                mask=mask, temperature=temperature, ambient=ambient, **ctx
            )
            row.update(out)
        except Exception as exc:  # fail-closed per extractor
            row["%s_error" % ext.name] = str(exc)
    row["_inputs"] = sorted(available)
    row["_extractors_run"] = [e.name for e in extractors]
    return row
