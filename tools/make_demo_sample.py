# -*- coding: utf-8 -*-
"""Generate a tiny synthetic temporal sequence for quickstart and CI.

Produces, under ``--out`` (default: ``samples/demo``):

    frames/0000.png ...        RGB frames (a green disc that grows over time)
    masks/0000.png ...         anchor masks for the labeled frames
    manifest.csv               a :class:`CsvManifestAdapter`-compatible manifest

The RGB frames and masks are synthetic — regenerated on demand and git-ignored.
The thermal sample under ``samples/demo/thermal/`` is **real FLIR data**
committed with the repo (temperature matrices, SAM2 canopy masks, and
environment log).  Use ``--skip-thermal`` to skip the thermal verification
step.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict

import cv2
import numpy as np


def make_sequence(out: str, n_frames: int = 6, seed: int = 0) -> List[Dict[str, str]]:
    """Write a synthetic growing-disc sequence + manifest; return the rows."""
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    cx, cy = w // 2, h // 2
    frames_dir = os.path.join(out, "frames")
    masks_dir = os.path.join(out, "masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for i in range(n_frames):
        r = int(18 + i * 6)                      # canopy grows over time
        jx = int(rng.normal(0, 3))               # small positional jitter
        jy = int(rng.normal(0, 3))
        img = np.full((h, w, 3), 235, np.uint8)  # light background
        yy, xx = np.ogrid[:h, :w]
        disc = (xx - (cx + jx)) ** 2 + (yy - (cy + jy)) ** 2 <= r * r
        img[disc] = (60, 180, 60)                # green "canopy"
        noise = rng.integers(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        stem = "%04d" % i
        cv2.imwrite(os.path.join(frames_dir, stem + ".png"), img)

        is_anchor = 1 if i in (0, 3, 5) else 0    # sparse manual labels
        mask_path = ""
        if is_anchor:
            mask_path = os.path.join(masks_dir, stem + ".png")
            cv2.imwrite(mask_path, disc.astype(np.uint8) * 255)

        rows.append({
            "sequence_key": "demo_01",
            "frame_idx": i,
            "frame_path": os.path.join(frames_dir, stem + ".png"),
            "frame_label": "T%d" % i,
            "is_anchor": is_anchor,
            "mask_path": mask_path,
            "das": i + 1,
        })

    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote demo sample to %s (%d frames, %d anchors)"
          % (out, n_frames, sum(int(r["is_anchor"]) for r in rows)))
    return rows


def verify_thermal_sample(out: str) -> str:
    """Verify the committed real FLIR thermal sample exists.

    Returns the thermal directory path.

    Raises FileNotFoundError if the committed sample is missing (should not
    happen after a normal ``git clone`` unless the user deleted it).
    """
    thermal_dir = os.path.join(out, "thermal")
    masks_dir = os.path.join(thermal_dir, "masks")

    missing: list[str] = []
    for i in range(6):
        stem = "%04d" % i
        npy = os.path.join(thermal_dir, f"temperature_{stem}.npy")
        png = os.path.join(masks_dir, f"{stem}.png")
        for p, label in [(npy, "temperature"), (png, "mask")]:
            if not os.path.isfile(p):
                missing.append(f"thermal/{label}_{stem}")

    env_csv = os.path.join(thermal_dir, "environment.csv")
    if not os.path.isfile(env_csv):
        missing.append("thermal/environment.csv")

    if missing:
        raise FileNotFoundError(
            "Real FLIR thermal sample is missing from the repo. Expected files:\n  "
            + "\n  ".join(missing)
            + "\n\nDid you clone with --depth=1 or delete samples/? "
            "Re-clone the full repo or restore the samples/demo/thermal/ directory."
        )

    # Print a quick summary
    npy0 = os.path.join(thermal_dir, "temperature_0000.npy")
    temp = np.load(npy0)
    print(
        "thermal sample OK — 6 real FLIR frames (%dx%d), 6 SAM2 masks, "
        "environment.csv"
        % (temp.shape[1], temp.shape[0])
    )
    return thermal_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "samples", "demo"))
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-thermal", action="store_true",
                    help="skip thermal sample verification")
    args = ap.parse_args()
    make_sequence(args.out, args.n_frames, args.seed)
    if not args.skip_thermal:
        verify_thermal_sample(args.out)


if __name__ == "__main__":
    main()
