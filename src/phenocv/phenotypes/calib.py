# -*- coding: utf-8 -*-
"""Camera intrinsics + calibration loader for metric (mm) 3D traits.

Ported from ``pheno_extract.config.CameraIntrinsics`` / ``load_rgb_intrinsics``.
Kept minimal (no Windows path-mapping machinery) so it is portable and
CPU-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def load_rgb_intrinsics(calibration_path: str, width: int, height: int) -> CameraIntrinsics:
    """Load RGB intrinsics for a given resolution from a JSON calibration file.

    Expected layout (RealSense-style)::

        {"results_by_resolution": {"1280x720": {"attempts": [{"resolved_streams":
            {"color": {"resolved": {"intrinsics": {"fx":..,"fy":..,"ppx":..,"ppy":..}}}}}...]}}}

    ``ppx``/``ppy`` map to ``cx``/``cy``.
    """
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    key = "%dx%d" % (int(width), int(height))
    try:
        intr = payload["results_by_resolution"][key]["attempts"][0]["resolved_streams"]["color"]["resolved"]["intrinsics"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("calibration file missing RGB %s intrinsics" % key) from exc
    return CameraIntrinsics(
        width, height,
        float(intr["fx"]), float(intr["fy"]),
        float(intr["ppx"]), float(intr["ppy"]),
    )
