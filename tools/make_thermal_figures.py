# -*- coding: utf-8 -*-
"""Generate synthetic thermal (FLIR) figures for the PhenoCV docs.

All five figures are rendered with **cv2 + numpy only** (no matplotlib, which
segfaults on the build host). Every image is deterministic
(``np.random.default_rng(seed)``) and written as a BGR PNG into ``docs/assets/``
so the README / docs can embed them directly:

  1. fig_thermal_scene.png     — warm canopy disc over a cooler background,
                                 rendered with a cv2 colormap (INFERNO).
  2. fig_thermal_overlay.png   — the canopy disc mask drawn as a green contour
                                 over the thermal scene.
  3. fig_thermal_layers.png    — the disc partitioned into upper/middle/lower
                                 thirds (by relative height of the bbox) and
                                 colour-coded over the thermal scene.
  4. fig_thermal_envalign.png  — a synthetic ambient-temperature time series +
                                 frame-timestamp markers, drawn on a blank
                                 canvas (axes + polyline + markers + cv2.putText).
  5. fig_thermal_stress.png    — a synthetic before/after stress response:
                                 paired bars (pre vs post event) with a delta
                                 annotation, drawn with cv2.

All five figures are illustrative/synthetic only — they contain no real paths,
serials, or private data.

仅用 cv2 + numpy 生成热红外（FLIR）示意图，供文档嵌入使用（不依赖 matplotlib）。
全部图像确定性生成（固定随机种子），不含任何真实路径、序列号或私人数据。

Usage / 用法
------------
    python tools/make_thermal_figures.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np

WIDTH, HEIGHT = 960, 640
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")

# Synthetic scene geometry (image coordinates).  /  合成场景几何（像素坐标）。
SEED = 20260728
CX, CY = 480, 330
DISC_RX, DISC_RY = 230, 180
T_CANOPY = (32.0, 34.0)   # warm canopy temperature range (°C)  /  暖冠层温度区间
T_BG = (22.0, 24.0)       # cooler background temperature range (°C)  /  冷背景温度区间


def make_scene(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Build a float32 [H,W] temperature matrix: warm disc over cool background.

    生成 float32 温度矩阵：暖色圆盘叠加在冷背景之上。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Radial distance from the disc centre.  /  到圆盘中心的径向距离。
    r = np.sqrt(((xx - CX) / DISC_RX) ** 2 + ((yy - CY) / DISC_RY) ** 2)
    disc = r <= 1.0

    # Background: gentle gradient + small noise.  /  背景：缓变梯度 + 轻微噪声。
    bg = T_BG[0] + (T_BG[1] - T_BG[0]) * (yy / h)
    bg += rng.normal(0.0, 0.25, size=(h, w)).astype(np.float32)

    # Canopy: warm core cooling toward the edge (more realistic thermal profile).
    # 冠层：中心最暖、边缘略凉（更接近真实热剖面）。
    edge = np.clip(r, 0.0, 1.0)
    canopy = (T_CANOPY[0] + (T_CANOPY[1] - T_CANOPY[0]) * (1.0 - edge)) + rng.normal(
        0.0, 0.18, size=(h, w)
    ).astype(np.float32)

    temp = bg.copy()
    temp[disc] = canopy[disc]
    return temp.astype(np.float32)


def render_colormap(temp: np.ndarray, vmin: float, vmax: float, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Normalise a float32 temperature matrix to [0,255] and apply a cv2 colormap.

    将 float32 温度归一化到 0–255 并施加 cv2 颜色映射（返回 BGR 画布）。
    """
    clipped = np.clip((temp - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    gray = np.round(clipped * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


def disc_mask(h: int, w: int) -> np.ndarray:
    """Boolean disc mask in image coordinates.  /  像素坐标系下的圆盘掩膜。"""
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx - CX) / DISC_RX) ** 2 + ((yy - CY) / DISC_RY) ** 2
    return r <= 1.0


def figure_scene(temp: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """fig 1: colormapped thermal scene + a thin border + a manual legend.

    图 1：颜色映射后的热场景 + 细边框 + 手写图例。
    """
    canvas = render_colormap(temp, vmin, vmax)
    # Thin border.  /  细边框。
    cv2.rectangle(canvas, (2, 2), (WIDTH - 3, HEIGHT - 3), (255, 255, 255), 1)
    # Manual legend (hot / cold swatches + text).  /  手写图例（暖/冷色块 + 文字）。
    cv2.putText(canvas, "Canopy ~32-34 C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Background ~22-24 C", (20, HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def figure_overlay(temp: np.ndarray, vmin: float, vmax: float, mask: np.ndarray) -> np.ndarray:
    """fig 2: thermal scene with the canopy disc drawn as a green contour.

    图 2：热场景上叠加绿色冠层轮廓。
    """
    canvas = render_colormap(temp, vmin, vmax)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Green outline (BGR = (0, 255, 0)).  /  绿色外轮廓。
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Canopy mask (green contour)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def figure_layers(temp: np.ndarray, vmin: float, vmax: float, mask: np.ndarray) -> np.ndarray:
    """fig 3: partition the canopy disc into upper/middle/lower thirds (by the
    vertical bbox of the disc) and colour-code the three regions over the scene.

    Conceptually identical to ``phenocv.thermal.partition_canopy_by_relative_height``:
    take the vertical bounding box of the whole-canopy mask and split it into
    three equal bands (top -> bottom).

    图 3：按圆盘垂直 bbox 把冠层三等分（上/中/下），在热场景上以三种颜色叠加显示。
    逻辑与 ``partition_canopy_by_relative_height`` 一致：取整株掩膜垂直包围盒三等分。
    """
    canvas = render_colormap(temp, vmin, vmax).astype(np.float32)
    rows = np.flatnonzero(mask.any(axis=1))
    y_min, y_max = int(rows.min()), int(rows.max())
    height = y_max - y_min + 1
    fractions = [1.0 / 3.0, 2.0 / 3.0]
    bounds = [y_min, y_min + int(height * fractions[0]), y_min + int(height * fractions[1]), y_max + 1]

    # BGR colours (OpenCV): upper=cyan, middle=green, lower=magenta.
    # BGR 颜色：上=青、中=绿、下=洋红（对齐 make_layer_overlay 的配色）。
    colors = [(255, 255, 0), (80, 220, 80), (255, 80, 220)]
    labels = ["upper", "middle", "lower"]
    # Blend the three bands in float32.  /  在 float32 下做三层颜色混合。
    for i in range(3):
        band = mask & (np.arange(temp.shape[0])[None, :].T >= bounds[i]) & (np.arange(temp.shape[0])[None, :].T < bounds[i + 1])
        fill = band & mask
        canvas[fill] = 0.70 * canvas[fill] + 0.30 * np.array(colors[i], dtype=np.float32)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    # Draw band contours + labels on the 8-bit canvas.  /  在 8 位画布上画轮廓与文字。
    for i in range(3):
        band = mask & (np.arange(temp.shape[0])[None, :].T >= bounds[i]) & (np.arange(temp.shape[0])[None, :].T < bounds[i + 1])
        fill = band & mask
        cnts, _ = cv2.findContours(fill.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, colors[i], 2, cv2.LINE_AA)
        cy = (bounds[i] + bounds[i + 1]) // 2
        cv2.putText(canvas, labels[i], (CX - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2, cv2.LINE_AA)
    return canvas


def figure_envalign(rng: np.random.Generator) -> np.ndarray:
    """fig 4: synthetic ambient-temperature time series + frame markers.

    Manual axes, ticks and labels (cv2.putText). Drawn on a blank canvas.

    图 4：合成的环境温度时序 + 帧时刻标记。手动绘制坐标轴/刻度/文字（cv2.putText），
    绘制于空白画布。
    """
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    # Plot area margins.  /  绘图区边距。
    left, right, top, bottom = 90, WIDTH - 40, 50, HEIGHT - 70
    t_min, t_max = 0.0, 24.0        # hours of a day
    y_min, y_max = 20.0, 26.0       # ambient temperature (°C)

    # Axes.  /  坐标轴。
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)

    # Y grid + ticks.  /  Y 轴网格 + 刻度。
    for v in np.arange(y_min, y_max + 0.01, 1.0):
        y = int(bottom - (v - y_min) / (y_max - y_min) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(canvas, "%.0f" % v, (left - 30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # X ticks every 4 hours.  /  X 轴每 4 小时一个刻度。
    for hh in np.arange(t_min, t_max + 0.01, 4.0):
        x = int(left + (hh - t_min) / (t_max - t_min) * (right - left))
        cv2.line(canvas, (x, top), (x, bottom), (220, 220, 220), 1)
        cv2.putText(canvas, "%.0fh" % hh, (x - 12, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Synthetic ambient curve: diurnal sinusoid + noise.  /  合成环境温度：日变化正弦 + 噪声。
    n = 96
    hours = np.linspace(t_min, t_max, n)
    ambient = 23.0 + 2.2 * np.sin((hours - 6.0) / 24.0 * 2 * np.pi) + rng.normal(0.0, 0.12, n)
    pts = []
    for hh, val in zip(hours, ambient):
        x = int(left + (hh - t_min) / (t_max - t_min) * (right - left))
        y = int(bottom - (val - y_min) / (y_max - y_min) * (bottom - top))
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    # Blue polyline (BGR = (255, 0, 0)).  /  蓝色折线。
    cv2.polylines(canvas, [pts], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # Frame-timestamp markers (e.g. FLIR frame capture instants).  /  帧时刻标记。
    markers = [3.5, 9.0, 14.0, 19.5, 22.0]
    for hh in markers:
        x = int(left + (hh - t_min) / (t_max - t_min) * (right - left))
        y = int(bottom - (23.0 + 2.2 * np.sin((hh - 6.0) / 24.0 * 2 * np.pi) - y_min) / (y_max - y_min) * (bottom - top))
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "f%.0f" % (markers.index(hh)), (x - 6, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Ambient temperature (synthetic, C) vs hour-of-day", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "blue = ambient curve; red dots = aligned FLIR frames", (left, bottom + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def figure_stress(rng: np.random.Generator) -> np.ndarray:
    """fig 5: synthetic before/after stress response — paired bars (pre vs post
    event) + a delta annotation. Everything drawn with cv2 on a blank canvas.

    图 5：合成胁迫前后对照——配对柱状（事件前 vs 事件后）+ 差值标注。全部用 cv2
    绘制于空白画布。
    """
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 120, WIDTH - 80, 60, HEIGHT - 90
    y_min, y_max = 0.0, 6.0

    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)

    # Y grid + labels.  /  Y 轴网格 + 标签。
    for v in np.arange(0, y_max + 0.01, 1.0):
        y = int(bottom - (v - y_min) / (y_max - y_min) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(canvas, "%.0f" % v, (left - 28, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Synthetic means (canopy delta-T, K) before/after an irrigation event.
    # 合成均值（冠层 ΔT，单位 K）：灌溉事件前/后。
    pre = 4.9 + rng.normal(0, 0.15)
    post = 1.6 + rng.normal(0, 0.15)
    delta = pre - post

    groups = [("pre-event", pre, (0, 0, 200)), ("post-event", post, (0, 160, 0))]
    bw = 110
    gap = 90
    x0 = left + 150
    for i, (label, val, color) in enumerate(groups):
        x = x0 + i * (bw + gap)
        y_top = int(bottom - (val - y_min) / (y_max - y_min) * (bottom - top))
        cv2.rectangle(canvas, (x, y_top), (x + bw, bottom), color, -1)
        cv2.putText(canvas, "%.2f" % val, (x + bw // 2 - 22, y_top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 6, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Delta annotation arrow + text.  /  差值标注。
    mid_x = x0 + bw + gap // 2
    cv2.arrowedLine(canvas, (x0 + bw, int(bottom - (pre - y_min) / (y_max - y_min) * (bottom - top))),
                    (x0 + bw + gap, int(bottom - (post - y_min) / (y_max - y_min) * (bottom - top))),
                    (0, 0, 0), 2)
    cv2.putText(canvas, "delta = -%.2f K" % delta, (mid_x - 30, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Canopy ΔT (synthetic, K): before vs after irrigation", (left, top - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "lower ΔT after rewatering = recovered transpiration", (left, bottom + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    temp = make_scene(rng, HEIGHT, WIDTH)
    vmin, vmax = float(temp.min()), float(temp.max())
    mask = disc_mask(HEIGHT, WIDTH)

    figures = {
        "fig_thermal_scene.png": figure_scene(temp, vmin, vmax),
        "fig_thermal_overlay.png": figure_overlay(temp, vmin, vmax, mask),
        "fig_thermal_layers.png": figure_layers(temp, vmin, vmax, mask),
        "fig_thermal_envalign.png": figure_envalign(rng),
        "fig_thermal_stress.png": figure_stress(rng),
    }

    for name, img in figures.items():
        path = os.path.join(ASSET_DIR, name)
        cv2.imwrite(path, img)
        size = os.path.getsize(path)
        print("wrote %s (%d bytes, %s)" % (path, size, img.shape))


if __name__ == "__main__":
    main()
