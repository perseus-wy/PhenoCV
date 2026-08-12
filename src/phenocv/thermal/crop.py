# -*- coding: utf-8 -*-
"""Plant-only thermal cropping — isolate and crop the plant from a FLIR frame.

Given an aligned (temperature, plant-mask) pair from the same FLIR frame, this
module produces two artifacts:

* a **plant-only temperature matrix** — the full-frame H×W temperature with the
  background set to ``NaN`` (saved as ``.npy`` / ``.csv``); and
* a **tight bounding-box crop** of that plant (pseudocolor overlay + cropped
  temperature matrix + cropped mask), so downstream models only ever see the
  plant, never the pot / bench / neighbours.

The mask is supplied by the caller — the recommended production path is SAM 2 on
the *aligned* RGB frame (see :mod:`phenocv.thermal.segmentation`), but any
registered/imported mask of the same H×W works.

Pure CPU (``numpy`` + ``cv2``). Fail-closed: an empty mask or a mask with no
finite temperature yields a result with ``ok=False`` and a ``reason`` string
rather than raising or fabricating temperature.

植株专属热红外裁剪 —— 从 FLIR 帧中隔离并裁出植株。

输入同一 FLIR 帧对齐的（温度矩阵, 植株掩膜）对，输出两类产物：
* 背景置 NaN 的整帧 H×W 植株温度矩阵（存 .npy/.csv）；
* 按紧致 bbox 切出的植株裁剪图（伪彩叠加 + 裁剪温度矩阵 + 裁剪掩膜），
  下游模型只看到植株，看不到盆/台面/邻株。

掩膜由调用方提供 —— 推荐生产路径是同帧对齐 RGB 经 SAM 2 分割
（见 :mod:`phenocv.thermal.segmentation`），但任何同尺寸已注册/导入掩膜均可。
纯 CPU（numpy + cv2），fail-closed：空掩膜或无有限温度时返回 ok=False + reason，
不抛异常、不编造数值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Result container / 结果容器
# --------------------------------------------------------------------------
@dataclass
class PlantCropResult:
    """Outcome of :func:`crop_plant_from_thermal`.

    ``bbox`` uses **exclusive** Python-slice convention ``(x_min, y_min, x_max,
    y_max)`` so callers can directly do ``arr[y_min:y_max, x_min:x_max]``.

    When the crop fails (empty mask / no finite plant temperature) ``ok`` is
    ``False``, ``bbox`` is ``None``, the cropped arrays are empty, and
    ``reason`` explains the failure — never an exception, never a fabricated
    temperature.

    ``bbox`` 采用**左闭右开**的 Python 切片约定 ``(x_min, y_min, x_max, y_max)``，
    可直接 ``arr[y_min:y_max, x_min:x_max]``。裁剪失败时 ``ok=False``、
    ``bbox=None``、裁剪数组为空，``reason`` 说明原因 —— 既非异常也非编造数值。
    """

    ok: bool
    reason: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    scale: Tuple[float, float] = (float("nan"), float("nan"))
    n_plant_pixels: int = 0
    stats: Dict[str, float] = field(default_factory=dict)
    plant_temperature: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), np.float32)
    )
    cropped_temperature: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), np.float32)
    )
    cropped_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), bool)
    )
    cropped_overlay: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), np.uint8)
    )


# --------------------------------------------------------------------------
# Geometry helpers / 几何工具
# --------------------------------------------------------------------------
def _tight_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return exclusive (x0, y0, x1, y1) bbox of a boolean mask, or None if empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    return (x0, y0, x1, y1)


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    *,
    pad_px: int,
    min_size: int,
    height: int,
    width: int,
) -> Tuple[int, int, int, int]:
    """Apply padding + minimum-size growth, then clamp to the frame."""
    x0, y0, x1, y1 = bbox
    if pad_px > 0:
        x0 -= pad_px
        y0 -= pad_px
        x1 += pad_px
        y1 += pad_px
    bw = x1 - x0
    bh = y1 - y0
    if min_size > 0:
        if bw < min_size:
            grow = (min_size - bw) // 2
            x0 -= grow
            x1 += (min_size - bw) - grow
        if bh < min_size:
            grow = (min_size - bh) // 2
            y0 -= grow
            y1 += (min_size - bh) - grow
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, width, height)  # degenerate → whole frame
    return (x0, y0, x1, y1)


def _robust_stats(values: np.ndarray) -> Dict[str, float]:
    """Median/mean/p10/p90/std/count over finite values (all-NaN-safe)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "temp_median_c": float("nan"),
            "temp_mean_c": float("nan"),
            "temp_p10_c": float("nan"),
            "temp_p90_c": float("nan"),
            "temp_std_c": float("nan"),
            "pixel_count": 0,
        }
    return {
        "temp_median_c": float(np.median(finite)),
        "temp_mean_c": float(np.mean(finite)),
        "temp_p10_c": float(np.percentile(finite, 10)),
        "temp_p90_c": float(np.percentile(finite, 90)),
        "temp_std_c": float(np.std(finite)),
        "pixel_count": int(finite.size),
    }


# --------------------------------------------------------------------------
# Core / 核心裁剪
# --------------------------------------------------------------------------
def crop_plant_from_thermal(
    temperature: np.ndarray,
    mask: np.ndarray,
    *,
    pad_px: int = 0,
    min_size: int = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    contour_color: Tuple[int, int, int] = (255, 255, 0),
    fill_alpha: float = 0.30,
    bg_dim: float = 0.20,
) -> PlantCropResult:
    """Isolate and tightly crop the plant region from a thermal frame.

    Parameters
    ----------
    temperature : np.ndarray
        2D absolute-temperature matrix (°C, float). 二维绝对温度矩阵（°C）。
    mask : np.ndarray
        Plant boolean (or 0/1) mask aligned to ``temperature`` (same H×W).
        与温度矩阵对齐的植株二值掩膜。
    pad_px : int
        Pixels of padding added around the tight bbox (clamped to frame).
        紧致 bbox 外扩像素（截断到帧内）。
    min_size : int
        Minimum bbox side length; the bbox is grown (centred) to this size when
        the plant is smaller, then clamped to the frame. 0 disables.
        最小 bbox 边长；植株更小时居中放大到此尺寸再截断到帧内，0 表示不放大。
    vmin, vmax : float, optional
        Fixed colormap scale. When ``None``, derived from the finite plant
        temperatures (robust 1–99 % clip). 固定温标；缺省由植株有限温度稳健推导。
    colormap : int
        ``cv2.COLORMAP_*`` constant for the overlay. 叠加图颜色映射。
    contour_color : (B, G, R)
        Plant contour colour in BGR. 植株边界颜色（BGR）。
    fill_alpha : float
        Plant fill blend (0 = keep colormap, 1 = full tint). 植株填充混合比。
    bg_dim : float
        Background (non-plant, inside bbox) brightness factor in [0,1].
        bbox 内非植株背景的亮度系数（[0,1]，越小越暗）。

    Returns
    -------
    PlantCropResult
        ``ok``/``reason``, ``bbox`` (exclusive), ``plant_temperature`` (full
        H×W, background NaN), ``cropped_temperature`` / ``cropped_mask`` /
        ``cropped_overlay`` (bbox region), ``scale``, ``n_plant_pixels``,
        ``stats``. Fail-closed: empty mask / no finite plant temperature →
        ``ok=False`` with ``reason``.
    """
    temp = np.asarray(temperature, dtype=np.float32)
    if temp.ndim != 2:
        raise ValueError("temperature must be a 2D array.")
    height, width = temp.shape
    m = np.asarray(mask, dtype=bool)
    if m.shape != temp.shape:
        raise ValueError(
            "mask shape %s != temperature shape %s" % (m.shape, temp.shape)
        )

    # Plant-only temperature: full frame, background → NaN.
    plant_t = np.full(temp.shape, np.nan, dtype=np.float32)
    plant_t[m] = temp[m]
    plant_finite = plant_t[np.isfinite(plant_t)]
    n_plant = int(plant_finite.size)
    if n_plant == 0:
        return PlantCropResult(
            ok=False,
            reason="no_finite_plant_temperature",
            plant_temperature=plant_t,
            stats=_robust_stats(plant_finite),
        )

    bbox = _tight_bbox(m)
    if bbox is None:
        return PlantCropResult(
            ok=False,
            reason="empty_mask",
            plant_temperature=plant_t,
            stats=_robust_stats(plant_finite),
        )
    x0, y0, x1, y1 = _expand_bbox(
        bbox, pad_px=pad_px, min_size=min_size, height=height, width=width
    )

    cropped_t = plant_t[y0:y1, x0:x1]
    cropped_m = m[y0:y1, x0:x1]

    # Colormap scale.
    if vmin is None or vmax is None:
        lo, hi = np.percentile(plant_finite, [1, 99])
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
    vmin_f, vmax_f = float(vmin), float(vmax)
    scale = (vmin_f, vmax_f)

    clipped = np.clip((cropped_t - vmin_f) / max(vmax_f - vmin_f, 1e-9), 0.0, 1.0)
    clipped = np.nan_to_num(clipped, nan=0.0)  # background pixels → 0 (dimmed below)
    gray = np.round(clipped * 255).astype(np.uint8)
    base = cv2.applyColorMap(gray, colormap).astype(np.float32)

    fill = cropped_m
    overlay = base.copy()
    overlay[~fill] *= bg_dim  # subdue surroundings so the plant pops
    tint = np.asarray(contour_color, dtype=np.float32)
    overlay[fill] = (1.0 - fill_alpha) * overlay[fill] + fill_alpha * tint
    boundary = cv2.morphologyEx(
        cropped_m.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    overlay[boundary > 0] = tint
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return PlantCropResult(
        ok=True,
        reason=None,
        bbox=(x0, y0, x1, y1),
        scale=scale,
        n_plant_pixels=n_plant,
        stats=_robust_stats(plant_finite),
        plant_temperature=plant_t,
        cropped_temperature=cropped_t,
        cropped_mask=fill,
        cropped_overlay=overlay,
    )


# --------------------------------------------------------------------------
# Colormap name resolution / 颜色映射名解析
# --------------------------------------------------------------------------
_COLORMAP_NAMES = {
    "INFERNO": cv2.COLORMAP_INFERNO,
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "HOT": cv2.COLORMAP_HOT,
    "BONE": cv2.COLORMAP_BONE,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "HSV": cv2.COLORMAP_HSV,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "PARULA": cv2.COLORMAP_PARULA,
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "WINTER": cv2.COLORMAP_WINTER,
    "COOL": cv2.COLORMAP_COOL,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "PINK": cv2.COLORMAP_PINK,
    "TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
}


def resolve_colormap(name: str) -> int:
    """Map a colormap name (upper-case) to a ``cv2.COLORMAP_*`` constant."""
    key = str(name).upper()
    if key not in _COLORMAP_NAMES:
        raise ValueError(
            "unknown colormap %r (choices: %s)"
            % (name, ", ".join(sorted(_COLORMAP_NAMES)))
        )
    return _COLORMAP_NAMES[key]


# --------------------------------------------------------------------------
# Persistence / 落盘
# --------------------------------------------------------------------------
def save_plant_crop(
    result: PlantCropResult,
    out_dir,
    stem: str,
    *,
    save_npy: bool = True,
    save_csv: bool = True,
    save_png: bool = True,
    save_mask_png: bool = True,
) -> Dict[str, str]:
    """Write a :class:`PlantCropResult` to disk.

    Files (all prefixed by ``stem``):
      * ``<stem>_plant_temperature.npy`` — full H×W, background NaN.
      * ``<stem>_plant_temperature.csv`` — same, with ``NaN`` cells.
      * ``<stem>_crop.npy`` — bbox-cropped temperature, background NaN.
      * ``<stem>_crop.png`` — bbox pseudocolor overlay (BGR).
      * ``<stem>_crop_mask.png`` — bbox plant mask.
      * ``<stem>_crop_report.json`` — bbox / scale / stats / reason.

    Returns the mapping of artifact-kind → path written. Even on failure a
    report is written recording ``ok=False`` + ``reason`` (fail-closed), so a
    batch run never silently drops a frame.

    将结果落盘。即使失败也会写出 report（记录 ok=False + reason，fail-closed），
    批处理不会静默丢弃帧。
    """
    import json

    import pandas as pd
    from .io import save_mask

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    report = {
        "stem": stem,
        "ok": result.ok,
        "reason": result.reason,
        "bbox": list(result.bbox) if result.bbox else None,
        "scale_c": list(result.scale),
        "n_plant_pixels": result.n_plant_pixels,
        "stats": result.stats,
    }

    if result.ok:
        if save_npy:
            p = out / ("%s_plant_temperature.npy" % stem)
            np.save(p, result.plant_temperature)
            written["plant_temperature_npy"] = str(p)
            p = out / ("%s_crop.npy" % stem)
            np.save(p, result.cropped_temperature)
            written["crop_npy"] = str(p)
        if save_csv:
            p = out / ("%s_plant_temperature.csv" % stem)
            pd.DataFrame(result.plant_temperature).to_csv(
                p, index=False, header=False
            )
            written["plant_temperature_csv"] = str(p)
            p = out / ("%s_crop.csv" % stem)
            pd.DataFrame(result.cropped_temperature).to_csv(
                p, index=False, header=False
            )
            written["crop_csv"] = str(p)
        if save_png:
            p = out / ("%s_crop.png" % stem)
            cv2.imwrite(str(p), result.cropped_overlay)
            written["crop_png"] = str(p)
        if save_mask_png:
            p = out / ("%s_crop_mask.png" % stem)
            save_mask(result.cropped_mask, p)
            written["crop_mask_png"] = str(p)

    p = out / ("%s_crop_report.json" % stem)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    written["report"] = str(p)
    return written
