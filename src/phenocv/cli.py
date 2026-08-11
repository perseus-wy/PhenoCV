# -*- coding: utf-8 -*-
"""Command-line interface for PhenoCV temporal canopy segmentation."""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from . import __version__
from .config import load_config
from .adapters import CsvManifestAdapter, PlantPhenotypingAdapter
from .engine import run_sam2_video_temporal


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

    args = ap.parse_args(argv)

    if args.adapter == "csv" and not args.manifest:
        ap.error("--manifest is required for the csv adapter")
    if args.adapter == "plant" and (not args.index or not args.anchor_root):
        ap.error("--index and --anchor-root are required for the plant adapter")

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
