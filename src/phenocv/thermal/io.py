# -*- coding: utf-8 -*-
"""Thermal IO: temperature / thermal-feature / mask readers and renderers.

Everything here is pure ``cv2`` + ``numpy`` + ``PIL`` (no ``matplotlib``). The
renderers (``make_overlay`` / ``make_layer_overlay``) use ``cv2`` colormaps and
alpha blending so they are safe to call inside a headless worker process.

Ported (desensitised, bilingual) from the private FLIR pipeline:
  * ``thermal_feature_image`` — absolute temperature / local ΔT / gradient → 3ch
    feature image (reusable as a segmentation prompt).
  * ``resolve_layer_overlap`` — assign overlapping pixels between canopy layers
    by distance to each layer's identity seed (mutually exclusive, within support).
  * ``make_overlay`` / ``make_layer_overlay`` — fixed-scale temperature rendering
    with mask contours.
  * mask readers/writers and polygon rasterisation.

热红外 IO：温度 / 热特征 / 掩膜读写与渲染。

本模块仅依赖 ``cv2`` + ``numpy`` + ``PIL``（不引入 ``matplotlib``）。渲染函数使用
cv2 颜色映射与半透明叠加，可在无显示环境的工作进程中调用。函数均脱敏并保留中英
双语说明。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------
# Internal helpers / 内部工具
# --------------------------------------------------------------------------
def _robust_normalize(values: np.ndarray) -> np.ndarray:
    """Clip to the 1st–99th percentile range, scale to [0, 1].

    将数值裁剪到 1–99 百分位区间并缩放到 [0, 1]（稳健归一化）。
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    return normalized.astype(np.float32)


def _ellipse_kernel(radius: int) -> np.ndarray:
    """Elliptical structuring element of the given radius.

    生成指定半径的椭圆形形态学核。
    """
    radius = max(0, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask by ``radius`` pixels (no-op when radius <= 0).

    按给定半径膨胀二值掩膜（radius <= 0 时原样返回）。
    """
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), _ellipse_kernel(radius)) > 0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a binary mask by ``radius`` pixels (no-op when radius <= 0).

    按给定半径腐蚀二值掩膜（radius <= 0 时原样返回）。
    """
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return cv2.erode(np.asarray(mask, dtype=np.uint8), _ellipse_kernel(radius)) > 0


# --------------------------------------------------------------------------
# Temperature / meta readers / 温度与元数据读取
# --------------------------------------------------------------------------
def load_temperature(path) -> np.ndarray:
    """Load a temperature matrix from a ``.npy`` file as float32.

    从 ``.npy`` 文件加载温度矩阵（float32）。
    """
    p = Path(path)
    arr = np.load(str(p))
    return arr.astype(np.float32)


def load_thermal_meta(path) -> dict:
    """Load the sibling ``*_meta.json`` metadata for a temperature array.

    Accepts either the meta ``.json`` directly, or the temperature ``.npy``
    path (``stem_temp.npy`` -> ``stem_meta.json``). Returns the parsed dict,
    or ``{}`` when no metadata is found (never raises on a missing file).

    读取与温度数组同名的 ``*_meta.json`` 元数据。可接受直接的 meta .json，
    或温度 .npy 路径（``stem_temp.npy`` → ``stem_meta.json``）。文件缺失时
    返回空字典（不抛异常）。
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        meta_path = p
    elif p.name.endswith("_temp.npy"):
        meta_path = p.with_name(p.name[: -len("_temp.npy")] + "_meta.json")
    else:
        meta_path = p.with_suffix(".json")
    if not meta_path.exists():
        return {}
    import json

    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Thermal feature image / 热特征图像
# --------------------------------------------------------------------------
def thermal_feature_image(temperature: np.ndarray) -> np.ndarray:
    """Combine absolute temperature / local ΔT / gradient into a 3ch feature.

    Channel 0: robust-normalised absolute temperature.
    Channel 1: local contrast (pixel − Gaussian-blurred neighbourhood).
    Channel 2: gradient magnitude (edge cue).
    Designed as a GrabCut prompt for thermal mask refinement.

    将绝对温度 / 局部温差 / 梯度组合为 3 通道特征图（供 GrabCut 分割提示）。
    通道0：稳健归一化绝对温度；通道1：局部对比；通道2：梯度幅值。
    """
    normalized = _robust_normalize(temperature)
    absolute = np.round(normalized * 255).astype(np.uint8)
    local_mean = cv2.GaussianBlur(normalized, (0, 0), sigmaX=7.0)
    local_contrast = np.clip((normalized - local_mean) * 3.0 + 0.5, 0.0, 1.0)
    local = np.round(local_contrast * 255).astype(np.uint8)
    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    scale = float(np.percentile(gradient, 99)) if gradient.size else 1.0
    edges = np.round(np.clip(gradient / max(scale, 1e-9), 0.0, 1.0) * 255).astype(
        np.uint8
    )
    return np.dstack([absolute, local, edges])


# --------------------------------------------------------------------------
# Layer overlap resolution / 分层重叠消解
# --------------------------------------------------------------------------
def resolve_layer_overlap(
    layer_masks: Dict[str, np.ndarray],
    identity_seeds: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Assign overlapping pixels between canopy layers by distance to seeds.

    Within the union support of all layers, every pixel that two or more
    layers claim is given to the layer whose identity seed is nearest (distance
    transform). The result is mutually exclusive and never leaves the support.

    在全部层位的并集支持域内，把被两个及以上层位共同占用的像素分配给距其身份
    种子最近（距离变换）的层。结果层间互斥且不越出支持域。

    Returns
    -------
    resolved : dict[str, np.ndarray]
        Per-layer masks after overlap resolution. 消解后的各层掩膜。
    overlap_count : int
        Number of pixels that were contested before resolution. 消解前争议像素数。
    """
    if not layer_masks:
        return {}, 0
    names = list(layer_masks)
    stack = np.stack([np.asarray(layer_masks[name], bool) for name in names])
    support = stack.any(axis=0)
    overlap = stack.sum(axis=0) > 1
    overlap_count = int(overlap.sum())
    if overlap_count == 0:
        return {name: stack[i].copy() for i, name in enumerate(names)}, 0

    distances = np.stack(
        [
            cv2.distanceTransform(
                (~np.asarray(identity_seeds[name], bool)).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            for name in names
        ]
    )
    owners = np.argmin(distances, axis=0)
    resolved: Dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        mask = stack[index].copy()
        mask[overlap] = owners[overlap] == index
        # Identity seeds only decide contested pixels; they cannot re-merge
        # fixed support pixels from outside the current whole-canopy support.
        mask |= np.asarray(identity_seeds[name], bool) & support
        mask &= support
        resolved[name] = mask
    return resolved, overlap_count


# --------------------------------------------------------------------------
# Mask IO / 掩膜读写
# --------------------------------------------------------------------------
def polygons_to_mask(
    shape: Tuple[int, int],
    polygons: Iterable[Iterable[Iterable[float]]],
) -> np.ndarray:
    """Rasterise a list of polygons into a boolean mask (lossless, PIL-based).

    将多边形列表栅格化为二值掩膜（基于 PIL）。
    """
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [(int(round(x)), int(round(y))) for x, y in polygon]
        if len(points) < 3:
            raise ValueError("Each polygon needs at least 3 vertices.")
        draw.polygon(points, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def save_mask(mask: np.ndarray, path) -> None:
    """Save a boolean mask as a lossless 8-bit PNG (0/255).

    以无损 8 位 PNG 保存二值掩膜（0/255）。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(target)


def load_mask(path) -> np.ndarray:
    """Load a boolean mask from a PNG (``>0`` = foreground).

    从 PNG 读取二值掩膜（``>0`` 为前景）。
    """
    return np.asarray(Image.open(path).convert("L")) > 0


# --------------------------------------------------------------------------
# Overlays (cv2 only, no matplotlib) / 叠加渲染（仅 cv2）
# --------------------------------------------------------------------------
def make_overlay(
    temperature: np.ndarray,
    mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Render temperature on a fixed scale with a cyan contour + mask fill.

    固定温标渲染温度，并用青色边界和半透明填充显示掩膜（cv2 实现）。
    """
    clipped = np.clip((temperature - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    gray = np.round(clipped * 255).astype(np.uint8)
    base = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    boundary = cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    overlay = base.astype(np.float32)
    fill = np.asarray(mask, dtype=bool)
    overlay[fill] = 0.72 * overlay[fill] + 0.28 * np.array([220, 170, 0])
    overlay[boundary > 0] = np.array([255, 255, 0])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_layer_overlay(
    temperature: np.ndarray,
    layer_masks: Dict[str, np.ndarray],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Render upper/middle/lower leaf masks with fixed per-layer colours.

    用固定颜色显示上、中、下层叶片掩膜（cv2 实现）。OpenCV 使用 BGR；上层青、
    中层绿、下层洋红。
    """
    clipped = np.clip((temperature - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    gray = np.round(clipped * 255).astype(np.uint8)
    overlay = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO).astype(np.float32)
    colors = {
        "upper": np.array([255, 255, 0]),
        "middle": np.array([80, 220, 80]),
        "lower": np.array([255, 80, 220]),
    }
    for layer, mask in layer_masks.items():
        color = colors.get(layer, np.array([255, 255, 255]))
        fill = np.asarray(mask, dtype=bool)
        boundary = cv2.morphologyEx(
            np.asarray(mask, dtype=np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        overlay[fill] = 0.76 * overlay[fill] + 0.24 * color
        overlay[boundary > 0] = color
    return np.clip(overlay, 0, 255).astype(np.uint8)
