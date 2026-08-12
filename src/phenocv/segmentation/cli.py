# -*- coding: utf-8 -*-
"""Segmentation CLI — backend-aware dispatch.

This is the in-scope dispatch layer for the segmentation module. It mirrors the
``segment`` subcommand of the top-level ``phenocv`` CLI but routes through
:func:`~phenocv.segmentation.base.run_segmentation`, so the ``--backend`` flag
selects the algorithm (``sam2`` default; ``classical`` / ``yolo`` are
placeholder backends).

Note: the historical top-level ``src/phenocv/cli.py`` still calls
``engine.run_sam2_video_temporal`` directly and preserves its existing
behavior; swapping it to use :func:`run_segmentation` is a one-line change that
lives outside this module's edit scope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from .config import load_config
from .adapters import CsvManifestAdapter, PlantPhenotypingAdapter
from .base import build_segmenter, run_segmentation


def _build_sequences(args) -> List:
    if args.adapter == "plant":
        return PlantPhenotypingAdapter(
            index_csv=args.index,
            anchor_root=args.anchor_root,
            rgb_root=args.rgb_root,
            min_anchors=args.min_anchors,
        ).build_sequences()
    return CsvManifestAdapter(args.manifest, min_anchors=args.min_anchors).build_sequences()


def run(args) -> int:
    """Build sequences and dispatch to the selected backend."""
    cfg = load_config(args.config, args.preset)
    sequences = _build_sequences(args)
    if not sequences:
        print("No sequences built — check the manifest/adapter arguments.",
              file=sys.stderr)
        return 2

    result = run_segmentation(
        sequences, args.output, args.checkpoint,
        backend=args.backend, config=cfg,
        model_cfg=args.model_cfg, device=args.device,
        do_loo=not args.no_loo, export_isat=not args.no_isat,
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
        prog="python -m phenocv.segmentation.cli",
        description="PhenoCV temporal canopy segmentation (backend-aware)")
    ap.add_argument("--backend", default="sam2",
                   choices=["sam2", "classical", "yolo"],
                   help="segmentation backend (default: sam2)")
    ap.add_argument("--config", default=None, help="YAML config / preset file")
    ap.add_argument("--preset", default=None, help="preset name to overlay")
    ap.add_argument("--checkpoint", required=True, help="SAM 2 .pt checkpoint")
    ap.add_argument("--model-cfg", default="sam2.1_hiera_l.yaml")
    ap.add_argument("--output", required=True, help="output root directory")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--no-loo", action="store_true", help="skip LOO validation")
    ap.add_argument("--no-isat", action="store_true", help="skip ISAT export")
    ap.add_argument("--no-qa", action="store_true", help="skip QA grid export")
    ap.add_argument("--no-resume", action="store_true", help="ignore DONE flags")
    ap.add_argument("--min-anchors", type=int, default=2)
    ap.add_argument("--image-size", nargs=2, type=int, default=(720, 1280),
                   metavar=("H", "W"))
    ap.add_argument("--adapter", choices=["csv", "plant"], default="csv")
    ap.add_argument("--manifest", help="manifest CSV/JSON (csv adapter)")
    ap.add_argument("--index", help="frame-index CSV (plant adapter)")
    ap.add_argument("--anchor-root", help="anchor mask root (plant adapter)")
    ap.add_argument("--rgb-root", default=None, help="local mirror of frames")
    args = ap.parse_args(argv)

    if args.adapter == "csv" and not args.manifest:
        ap.error("--manifest is required for the csv adapter")
    if args.adapter == "plant" and (not args.index or not args.anchor_root):
        ap.error("--index and --anchor-root are required for the plant adapter")
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
