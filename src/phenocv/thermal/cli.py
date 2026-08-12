# -*- coding: utf-8 -*-
"""``phenocv thermal`` CLI — plant-only thermal cropping (pure CPU).

Subcommand ``crop`` isolates and tightly crops the plant from aligned
(temperature, mask) FLIR frames. A single ``.npy`` pair or a directory of
``*_temp.npy`` + matched masks is supported.

``phenocv thermal`` 命令行 —— 纯 CPU 的植株专属热红外裁剪。子命令 ``crop`` 从
对齐的（温度, 掩膜）FLIR 帧中隔离并紧致裁剪植株，支持单帧 ``.npy`` 对或
``*_temp.npy`` 目录 + 同名掩膜批量处理。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .crop import (
    crop_plant_from_thermal,
    resolve_colormap,
    save_plant_crop,
)
from .io import load_temperature, load_mask


def _stem_of(temp_path: Path) -> str:
    """Derive the frame stem from a temperature file name."""
    name = temp_path.name
    if name.endswith("_temp.npy"):
        return name[: -len("_temp.npy")]
    if name.endswith(".npy"):
        return name[: -len(".npy")]
    return temp_path.stem


def _find_mask(temp_path: Path, mask_dir: Path) -> Optional[Path]:
    """Match a mask for a temperature file by stem.

    Resolution order:
      * ``<mask_dir>/<stem>.png`` — when ``--mask`` already points at a flat
        whole-mask directory;
      * ``<mask_dir>/**/whole/<stem>.png`` — when ``--mask`` points at a
        segmentation run root (e.g. ``label_3_target_anchored_formal_…``), so
        the per-segment ``masks/whole`` masks are discovered recursively.
      * ``<mask_dir>/<stem>_mask.png`` / ``<stem>_whole.png`` — fallbacks.

    Only whole-canopy masks are accepted; layer masks (upper/middle/lower)
    are intentionally ignored because they are plant subsets, not the plant.
    """
    stem = _stem_of(temp_path)
    direct = mask_dir / ("%s.png" % stem)
    if direct.exists():
        return direct
    hits = sorted(mask_dir.rglob("whole/%s.png" % stem))
    if hits:
        return hits[0]
    for cand in (
        mask_dir / ("%s_mask.png" % stem),
        mask_dir / ("%s_whole.png" % stem),
    ):
        if cand.exists():
            return cand
    return None


def cmd_crop(args) -> int:
    temp_paths: List[Path]
    if args.temperature.is_dir():
        temp_paths = sorted(args.temperature.glob("*_temp.npy")) or sorted(
            args.temperature.glob("*.npy")
        )
    else:
        temp_paths = [args.temperature]

    if not temp_paths:
        print("No temperature file(s) found.", file=sys.stderr)
        return 2

    mask_dir: Optional[Path] = args.mask if (args.mask and args.mask.is_dir()) else None
    mask_file: Optional[Path] = args.mask if (args.mask and args.mask.is_file()) else None

    try:
        colormap = resolve_colormap(args.colormap)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    total = len(temp_paths)
    ok = 0
    for i, tp in enumerate(temp_paths, 1):
        mp = mask_file
        if mp is None and mask_dir is not None:
            mp = _find_mask(tp, mask_dir)
        if mp is None or not Path(mp).exists():
            print(
                "[%d/%d] %s: no matching mask, skipped" % (i, total, tp.name),
                file=sys.stderr,
            )
            continue

        temp = load_temperature(tp)
        mask = load_mask(mp)
        if mask.shape != temp.shape:
            # Align mask to temperature (nearest) — fail-closed on mismatch.
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (temp.shape[1], temp.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )

        res = crop_plant_from_thermal(
            temp,
            mask,
            pad_px=args.pad,
            min_size=args.min_size,
            vmin=args.vmin,
            vmax=args.vmax,
            colormap=colormap,
        )
        stem = _stem_of(tp)
        written = save_plant_crop(
            res,
            args.output,
            stem,
            save_npy=not args.no_npy,
            save_csv=not args.no_csv,
            save_png=not args.no_png,
            save_mask_png=not args.no_mask_png,
        )
        if res.ok:
            ok += 1
            sc = tuple(round(s, 2) for s in res.scale)
            med = res.stats.get("temp_median_c", float("nan"))
            print(
                "[%d/%d] %s -> bbox=%s scale=%s plant_px=%d median=%.2fC"
                % (i, total, stem, res.bbox, sc, res.n_plant_pixels, med)
            )
        else:
            print(
                "[%d/%d] %s -> SKIP (%s)" % (i, total, stem, res.reason),
                file=sys.stderr,
            )
    print("Done: %d/%d frames cropped -> %s" % (ok, total, args.output))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the ``thermal`` subcommand tree to the top-level parser."""
    th = sub.add_parser(
        "thermal", help="thermal (FLIR) plant-only cropping & analysis"
    )
    th_sub = th.add_subparsers(dest="thermal_cmd", required=True)

    c = th_sub.add_parser(
        "crop", help="isolate & tightly crop the plant from thermal frames"
    )
    c.add_argument(
        "--temperature",
        required=True,
        type=Path,
        help="temperature .npy (or a directory of *_temp.npy)",
    )
    c.add_argument(
        "--mask",
        required=True,
        type=Path,
        help="plant mask .png (or a directory; matched by stem)",
    )
    c.add_argument("--output", required=True, type=Path, help="output directory")
    c.add_argument("--pad", type=int, default=0, help="bbox padding (px)")
    c.add_argument("--min-size", type=int, default=0, help="min bbox side (px)")
    c.add_argument(
        "--colormap",
        default="INFERNO",
        help="cv2 colormap name (INFERNO/JET/TURBO/VIRIDIS/PLASMA/MAGMA/HOT/...)",
    )
    c.add_argument("--vmin", type=float, default=None, help="fixed colormap min (°C)")
    c.add_argument("--vmax", type=float, default=None, help="fixed colormap max (°C)")
    c.add_argument("--no-npy", action="store_true", help="skip .npy output")
    c.add_argument("--no-csv", action="store_true", help="skip .csv output")
    c.add_argument("--no-png", action="store_true", help="skip .png overlay")
    c.add_argument(
        "--no-mask-png", action="store_true", help="skip cropped mask png"
    )
    c.set_defaults(func=cmd_crop)
