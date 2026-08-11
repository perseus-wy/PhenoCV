# -*- coding: utf-8 -*-
"""Generate a tiny synthetic temporal sequence for quickstart and CI.

Produces, under ``--out`` (default: ``samples/demo``):

    frames/0000.png ...        RGB frames (a green disc that grows over time)
    masks/0000.png ...         anchor masks for the labeled frames
    manifest.csv               a :class:`CsvManifestAdapter`-compatible manifest

Everything is synthesized — no real data is involved. Commit the generator
only; the PNGs are git-ignored, so each user (and CI) regenerates on demand.
This keeps the repo free of any dataset while still giving a zero-config
\"clone -> run\" path.
"""

from __future__ import annotations

import argparse
import csv
import os
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "samples", "demo"))
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_sequence(args.out, args.n_frames, args.seed)


if __name__ == "__main__":
    main()
