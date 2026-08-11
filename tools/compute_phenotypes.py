# -*- coding: utf-8 -*-
"""Batch phenotype computation over a mask deliverable.

Entry point so another agent (or a human) can turn a directory of plant
masks (+ optional RGB / depth+calibration / multispectral) into a flat,
**fail-closed** trait table — without knowing PhenoCV internals.

This mirrors the PhenoScreen mask-deliverable "four-piece" contract: a
provenance manifest, a long CSV of traits, and per-frame JSON. The underlying
engine (`phenocv.phenotypes.compute_traits`) only runs the extractors whose
inputs are present, and returns ``NaN`` + ``missing_reason`` (or
``<name>_error``) instead of fabricating values.

Usage
-----
    python tools/compute_phenotypes.py \
        --mask-dir results/masks \
        --rgb-dir  /local/mirror/rgb \
        --depth-dir /local/mirror/depth_mm \
        --calibration configs/intrinsics_second.yaml \
        --ms-root /local/mirror/ms --bands 555 660 720 840 \
        --out results/phenotypes

Outputs (under --out)
---------------------
    traits_long.csv   one row per frame; all trait columns + provenance
    manifest.json     per-frame provenance (sources, inputs, extractors run)
    <stem>.json       full trait row for each frame
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make the script runnable without PYTHONPATH=src.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from phenocv.phenotypes import compute_traits  # noqa: E402
from phenocv.phenotypes.calib import load_rgb_intrinsics  # noqa: E402


# --------------------------------------------------------------------------
# Readers (self-contained, minimal)
# --------------------------------------------------------------------------
def _read_gray_mask(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("cannot read mask: %s" % path)
    return img > 0


def _read_rgb(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("cannot read rgb: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_depth_mm(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path)).astype(np.float32)
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("cannot read depth: %s" % path)
    return img.astype(np.float32)


def _read_ms_band(path: Path) -> np.ndarray:
    """Load one multispectral band as float32 reflectance (~unit interval).

    If the source is 16-bit (>1), normalize by 65535. Calibration against a
    reference panel is the caller's responsibility (see ``empirical_line_gains``).
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    else:
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError("cannot read ms band: %s" % path)
        arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:
        arr = arr / 65535.0
    return arr


def _first_match(directory: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch phenotype computation over a mask deliverable.")
    ap.add_argument("--mask-dir", required=True,
                    help="directory of binary mask images (PNG/JPG/...)")
    ap.add_argument("--rgb-dir", default=None,
                    help="directory of RGB images, matched by stem (Tier-2)")
    ap.add_argument("--depth-dir", default=None,
                    help="directory of depth images in mm (Tier-3)")
    ap.add_argument("--calibration", default=None,
                    help="intrinsics JSON/YAML for --depth (Tier-3)")
    ap.add_argument("--ms-root", default=None,
                    help="root dir with one subdir per band, e.g. <ms-root>/555/<stem>.png")
    ap.add_argument("--bands", nargs="+", type=int, default=[555, 660, 720, 840],
                    help="wavelengths present under --ms-root (default 555 660 720 840)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--glob", default="*", help="stem glob inside --mask-dir")
    args = ap.parse_args(argv)

    mask_dir = Path(args.mask_dir)
    if not mask_dir.is_dir():
        print("mask-dir not found: %s" % mask_dir, file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Discover frames by mask stem.
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    stems = sorted(
        p.stem for p in mask_dir.glob(args.glob)
        if p.suffix.lower() in exts and p.stem)
    if not stems:
        print("no mask files found in %s (glob=%r)"
              % (mask_dir, args.glob), file=sys.stderr)
        return 2

    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    ms_root = Path(args.ms_root) if args.ms_root else None

    rows = []          # full trait dicts (provenance + traits)
    manifest = []      # provenance only
    trait_cols: list[str] = []

    n_tiers = {1: 0, 2: 0, 3: 0, 4: 0}

    for stem in stems:
        mask_path = _first_match(mask_dir, stem)
        try:
            mask = _read_gray_mask(mask_path)
        except Exception as exc:
            manifest.append({"frame": stem, "mask_path": str(mask_path),
                             "error": "mask_read: %s" % exc})
            print("skip %s (mask read failed: %s)" % (stem, exc), file=sys.stderr)
            continue

        rgb_path = _first_match(rgb_dir, stem) if rgb_dir else None
        depth_path = _first_match(depth_dir, stem) if depth_dir else None

        rgb = _read_rgb(rgb_path) if rgb_path else None
        depth = _read_depth_mm(depth_path) if depth_path else None

        multispectral = None
        ms_bands_present = []
        if ms_root:
            bands = {}
            for b in args.bands:
                bp = _first_match(ms_root / str(b), stem)
                if bp:
                    try:
                        bands[b] = _read_ms_band(bp)
                        ms_bands_present.append(b)
                    except Exception as exc:
                        print("  %s: ms band %d read failed: %s"
                              % (stem, b, exc), file=sys.stderr)
            if bands:
                multispectral = bands

        row = compute_traits(
            mask=mask, rgb=rgb, depth=depth,
            calibration=args.calibration if depth is not None else None,
            multispectral=multispectral)

        inputs = row.get("_inputs", [])
        if "mask" in inputs:
            n_tiers[1] += 1
        if "rgb" in inputs:
            n_tiers[2] += 1
        if "depth" in inputs and "calibration" in inputs:
            n_tiers[3] += 1
        if "multispectral" in inputs:
            n_tiers[4] += 1

        row_with_meta = {
            "frame": stem,
            "mask_path": str(mask_path),
            "rgb_path": str(rgb_path) if rgb_path else "",
            "depth_path": str(depth_path) if depth_path else "",
            "ms_bands": ",".join(str(b) for b in ms_bands_present),
            "inputs_present": ";".join(inputs),
            "extractors_run": ";".join(row.get("_extractors_run", [])),
        }
        # trait columns (exclude orchestrator bookkeeping that we already surfaced)
        traits = {k: v for k, v in row.items() if not k.startswith("_")}
        row_with_meta.update(traits)
        rows.append(row_with_meta)
        manifest.append({
            "frame": stem,
            "mask_path": str(mask_path),
            "rgb_path": str(rgb_path) if rgb_path else None,
            "depth_path": str(depth_path) if depth_path else None,
            "ms_bands": ms_bands_present or None,
            "inputs_present": inputs,
            "extractors_run": row.get("_extractors_run", []),
        })

        # per-frame JSON
        (out / ("%s.json" % stem)).write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8")

        for k in traits:
            if k not in trait_cols:
                trait_cols.append(k)

    # Long CSV
    header = ["frame", "mask_path", "rgb_path", "depth_path", "ms_bands",
              "inputs_present", "extractors_run"] + sorted(trait_cols)
    csv_path = out / "traits_long.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("Processed %d frames -> %s" % (len(rows), out))
    print("  traits_long.csv : %d rows x %d trait columns" % (len(rows), len(trait_cols)))
    print("  tiers satisfied : L1=%d L2=%d L3=%d L4=%d"
          % (n_tiers[1], n_tiers[2], n_tiers[3], n_tiers[4]))
    print("  manifest.json   : provenance for every frame")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
