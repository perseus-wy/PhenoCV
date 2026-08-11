# -*- coding: utf-8 -*-
"""Command-line interface for PhenoCV temporal canopy segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import __version__
from .config import load_config
from .adapters import CsvManifestAdapter, PlantPhenotypingAdapter
from .engine import run_sam2_video_temporal
from . import phenotypes as P
from .phenotypes.calib import CameraIntrinsics, load_rgb_intrinsics


def _build_sequences(args) -> List:
    if args.adapter == "plant":
        return PlantPhenotypingAdapter(
            index_csv=args.index,
            anchor_root=args.anchor_root,
            rgb_root=args.rgb_root,
            min_anchors=args.min_anchors,
        ).build_sequences()
    return CsvManifestAdapter(args.manifest, min_anchors=args.min_anchors).build_sequences()


def cmd_segment(args) -> int:
    cfg = load_config(args.config, args.preset)
    sequences = _build_sequences(args)
    if not sequences:
        print("No sequences built — check the manifest/adapter arguments.",
              file=sys.stderr)
        return 2

    result = run_sam2_video_temporal(
        sequences, args.output, args.checkpoint,
        model_cfg=args.model_cfg, device=args.device,
        config=cfg, do_loo=not args.no_loo, export_isat=not args.no_isat,
        export_qa=not args.no_qa, resume=not args.no_resume,
        image_size=tuple(args.image_size),
    )

    loo = result.get("loo_summary_interior", {})
    print("Done: %d sequences, %d frames, %.1f min"
          % (result["n_sequences"], result["n_frames"], result["elapsed_min"]))
    if loo.get("n"):
        print("LOO (interior) IoU median=%.4f mean=%.4f  BF1 median=%.4f  (n=%d)"
              % (loo.get("iou_median", 0.0), loo.get("iou_mean", 0.0),
                 loo.get("bf1_median", 0.0), loo.get("n", 0)))
    return 0


def _read_gray_mask(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("cannot read mask: %s" % path)
    return img > 0


def _read_rgb(path: str) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("cannot read rgb: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_depth_mm(path: str) -> np.ndarray:
    """Load a depth image (16-bit PNG in mm, or .npy) as float32 mm."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(str(p)).astype(np.float32)
    import cv2
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("cannot read depth: %s" % path)
    return img.astype(np.float32)


def cmd_phenotype(args) -> int:
    """Compute phenotype traits for a single (mask[, rgb][, depth+calib]) frame.

    Runs every registered extractor whose inputs are satisfied and writes one
    flat trait row to JSON (and optionally CSV).
    """
    mask = _read_gray_mask(args.mask)
    rgb = _read_rgb(args.rgb) if args.rgb else None
    depth = _read_depth_mm(args.depth) if args.depth else None
    calibration = None
    if depth is not None:
        if args.calibration:
            calibration = load_rgb_intrinsics(
                args.calibration, int(depth.shape[1]), int(depth.shape[0]))
        elif args.intrinsics:
            fx, fy, cx, cy = args.intrinsics
            calibration = CameraIntrinsics(
                int(depth.shape[1]), int(depth.shape[0]), fx, fy, cx, cy)
        else:
            print("depth given but no --calibration/--intrinsics; skipping 3D.",
                  file=sys.stderr)
            depth = None

    row = P.compute_traits(mask=mask, rgb=rgb, depth=depth, calibration=calibration)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    print("Wrote %d trait columns -> %s (extractors: %s)"
          % (len(row), out, ", ".join(row.get("_extractors_run", []))))
    if args.csv:
        cols = [k for k in row if not k.startswith("_")]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerow([row[c] for c in cols])
        print("Wrote CSV -> %s" % args.csv)
    return 0


def cmd_list_traits(args) -> int:
    """List all registered trait extractors (name / tier / inputs)."""
    for name, ext in sorted(P.all_extractors().items(),
                            key=lambda kv: (kv[1].tier, kv[0])):
        print("[tier %d] %-32s requires=%s" % (ext.tier, name, ext.requires))
        if args.verbose and ext.description:
            print("           %s" % ext.description)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="phenocv", description="PhenoCV temporal canopy segmentation toolkit")
    ap.add_argument("--version", action="version", version="phenocv %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None,
                       help="YAML config (propagation params + presets)")
        p.add_argument("--preset", default=None, help="preset name to overlay")
        p.add_argument("--checkpoint", required=True, help="SAM 2 .pt checkpoint")
        p.add_argument("--model-cfg", default="sam2.1_hiera_l.yaml",
                       help="SAM 2 model config name (resolved from the sam2 package)")
        p.add_argument("--output", required=True, help="output root directory")
        p.add_argument("--device", default="cuda", help="cuda or cpu")
        p.add_argument("--no-loo", action="store_true", help="skip LOO validation")
        p.add_argument("--no-isat", action="store_true", help="skip ISAT export")
        p.add_argument("--no-qa", action="store_true", help="skip QA grid export")
        p.add_argument("--no-resume", action="store_true", help="ignore DONE flags")
        p.add_argument("--min-anchors", type=int, default=2)
        p.add_argument("--image-size", nargs=2, type=int, default=(720, 1280),
                       metavar=("H", "W"))

    p = sub.add_parser("segment", help="propagate keyframe masks to full sequences")
    common(p)
    p.add_argument("--adapter", choices=["csv", "plant"], default="csv")
    p.add_argument("--manifest", help="manifest CSV/JSON (csv adapter)")
    p.add_argument("--index", help="frame-index CSV (plant adapter)")
    p.add_argument("--anchor-root", help="anchor mask root (plant adapter)")
    p.add_argument("--rgb-root", default=None, help="local mirror of indexed frames")
    p.set_defaults(func=cmd_segment)

    # --- phenotype: compute traits for one frame -------------------------
    ph = sub.add_parser(
        "phenotype",
        help="compute phenotype traits (2D shape / RGB indices / 3D height) for a frame")
    ph.add_argument("--mask", required=True, help="binary mask image (PNG)")
    ph.add_argument("--rgb", default=None, help="RGB image for vegetation indices (Tier-2)")
    ph.add_argument("--depth", default=None,
                    help="depth image in mm (16-bit PNG or .npy) for 3D height (Tier-3)")
    ph.add_argument("--calibration", default=None,
                    help="RealSense-style calibration JSON (resolves intrinsics for --depth)")
    ph.add_argument("--intrinsics", nargs=4, type=float, default=None,
                    metavar=("FX", "FY", "CX", "CY"),
                    help="explicit pinhole intrinsics (alternative to --calibration)")
    ph.add_argument("--out", required=True, help="output trait JSON path")
    ph.add_argument("--csv", default=None, help="also write a one-row CSV")
    ph.set_defaults(func=cmd_phenotype)

    # --- list-traits -----------------------------------------------------
    lt = sub.add_parser("list-traits", help="list registered trait extractors")
    lt.add_argument("-v", "--verbose", action="store_true", help="show descriptions")
    lt.set_defaults(func=cmd_list_traits)

    args = ap.parse_args(argv)

    if args.cmd == "segment":
        if args.adapter == "csv" and not args.manifest:
            ap.error("--manifest is required for the csv adapter")
        if args.adapter == "plant" and (not args.index or not args.anchor_root):
            ap.error("--index and --anchor-root are required for the plant adapter")

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
