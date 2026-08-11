# -*- coding: utf-8 -*-
"""Worked example: potted-soybean temporal adapter.

This mirrors a reference dataset layout: a frame-index CSV plus a directory of
manual anchor masks (one sub-directory per plant, masks named after the source
RGB stems). It is provided as a concrete example of subclassing
:class:`~phenocv.adapters.base.BaseAdapter`. For new datasets the generic
:class:`~phenocv.adapters.csv_manifest.CsvManifestAdapter` is recommended.

Notes
-----
* Order by the filename timestamp stem, not by date — two captures can share a
  calendar day, and the filename stem's lexicographic order is the reliable
  temporal order.
* Anchor masks match source frames by basename stem (no index convention).
* ``rgb_root`` lets you redirect the index's frame paths to a fast local mirror
  (keep the ``<root>/<date>/<stem>.<ext>`` layout) so frame reads don't hit
  slow network storage during propagation.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import cv2
import numpy as np

from .base import BaseAdapter
from ..engine import AnchorFrame, PlantSequence

_PLANT_DIR_RE = re.compile(r"^plant_(\d+)$")


# --------------------------------------------------------------------------
# Pure string / path helpers (no IO, directly unit-testable)
# --------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """Normalize backslashes to forward slashes; preserve any leading UNC."""
    if not path:
        return ""
    p = path.replace("\\", "/")
    if path.startswith("\\\\") and not p.startswith("//"):
        p = "//" + p.lstrip("/")
    return p


def plant_key(plant_no: Any) -> str:
    """``1`` / ``"1.0"`` / ``1.0`` -> ``"plant_01"``."""
    return "plant_%02d" % parse_plant_no(plant_no)


def parse_plant_no(plant_no: Any) -> int:
    """Index plant_no may be a float string like ``"1.0"``; normalize to int."""
    if isinstance(plant_no, str):
        plant_no = plant_no.strip()
        if not plant_no:
            raise ValueError("plant_no is empty")
    return int(float(plant_no))


def _stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(normalize_path(path)))[0]


def _resolve_rgb_path(raw_path: str, date: str, rgb_root: Optional[str]) -> str:
    """Redirect an indexed frame path to a local mirror root when given."""
    if not rgb_root:
        return normalize_path(raw_path)
    return "%s/%s/%s" % (rgb_root.rstrip("/\\"), date,
                         os.path.basename(normalize_path(raw_path)))


# --------------------------------------------------------------------------
# Anchor (manual mask) loading
# --------------------------------------------------------------------------

def load_anchor_masks(anchor_dir: str,
                      threshold: int = 127) -> Dict[str, np.ndarray]:
    """Read all manual anchor masks for one plant.

    ``anchor_dir`` contains a ``masks/`` sub-directory; if absent, the directory
    itself is used. Returns ``{stem: bool mask}`` keyed by source RGB stem.
    """
    mask_dir = os.path.join(anchor_dir, "masks")
    if not os.path.isdir(mask_dir):
        mask_dir = anchor_dir
    out: Dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[os.path.splitext(os.path.basename(path))[0]] = img > threshold
    return out


def _discover_anchor_dirs(anchor_root: str) -> Dict[int, str]:
    """Scan ``anchor_root`` for ``plant_XX`` directories -> ``{plant_no: dir}``."""
    found: Dict[int, str] = {}
    if not os.path.isdir(anchor_root):
        return found
    for name in sorted(os.listdir(anchor_root)):
        m = _PLANT_DIR_RE.match(name)
        if not m:
            continue
        full = os.path.join(anchor_root, name)
        if os.path.isdir(full):
            found[int(m.group(1))] = full
    return found


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------

class PlantPhenotypingAdapter(BaseAdapter):
    """Adapter for the potted-soybean reference layout."""

    def __init__(self,
                 index_csv: str,
                 anchor_root: str,
                 plants: Optional[Iterable[Any]] = None,
                 rgb_root: Optional[str] = None,
                 frame_role: str = "plant",
                 min_anchors: int = 2,
                 strict: bool = False,
                 progress=None) -> None:
        self.index_csv = index_csv
        self.anchor_root = anchor_root
        self.plants = plants
        self.rgb_root = rgb_root
        self.frame_role = frame_role
        self.min_anchors = min_anchors
        self.strict = strict
        self.progress = progress

    def build_sequences(self, **kwargs) -> List[PlantSequence]:
        return build_sequences_from_index(
            self.index_csv, self.anchor_root,
            plants=self.plants, rgb_root=self.rgb_root,
            frame_role=self.frame_role, min_anchors=self.min_anchors,
            strict=self.strict, progress=self.progress)


def build_sequences_from_index(index_csv: str,
                               anchor_root: str,
                               plants: Optional[Iterable[Any]] = None,
                               rgb_root: Optional[str] = None,
                               frame_role: str = "plant",
                               min_anchors: int = 2,
                               strict: bool = False,
                               progress=None) -> tuple:
    """Build engine-ready sequences from a frame-index CSV + anchor masks.

    Parameters
    ----------
    index_csv : path to the frame-index CSV (one row per frame).
    anchor_root : root containing per-plant ``plant_XX`` mask directories.
    plants : only process these plant numbers (None = all with anchors).
    rgb_root : optional local mirror of the indexed frames.
    frame_role : filter on the index ``frame_role`` column (fixed ``plant`` in
                 the reference layout; calibration frames are excluded).
    min_anchors : skip plants with fewer anchors (<=1 can't be propagated).
    strict : raise on an unmatched anchor instead of recording + skipping.

    Returns
    -------
    ``(sequences, skipped)``. ``skipped`` lists each dropped plant with a reason.
    """
    def _log(msg: str) -> None:
        if progress:
            progress(msg)

    with open(index_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if (r.get("frame_role") or "").strip() == frame_role
                and (r.get("plant_no") or "").strip()]
    if not rows:
        raise ValueError("%s: no rows with frame_role=%s and non-empty plant_no"
                         % (index_csv, frame_role))

    grouped: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(parse_plant_no(r["plant_no"]), []).append(r)

    anchor_dirs = _discover_anchor_dirs(anchor_root)
    if not anchor_dirs:
        raise ValueError("%s: no plant_XX anchor directories found" % anchor_root)

    wanted: Optional[Set[int]] = None
    if plants is not None:
        wanted = {parse_plant_no(p) for p in plants}
        missing = sorted(wanted - set(grouped))
        if missing:
            raise ValueError("plant_no(s) not in index: %s" % missing)

    sequences: List[PlantSequence] = []
    skipped: List[Dict[str, Any]] = []

    for plant_no in sorted(grouped):
        if wanted is not None and plant_no not in wanted:
            continue
        key = plant_key(plant_no)

        anchor_dir = anchor_dirs.get(plant_no)
        if anchor_dir is None:
            skipped.append({"plant_no": plant_no, "key": key,
                            "reason": "no_anchor_dir"})
            continue

        # filename timestamp stem order == temporal order
        frames = sorted(grouped[plant_no], key=lambda r: _stem_of(r["rgb_path"]))
        frame_paths = [_resolve_rgb_path(r["rgb_path"], r["date"], rgb_root)
                       for r in frames]
        stems = [_stem_of(r["rgb_path"]) for r in frames]
        stem_to_idx = {s: i for i, s in enumerate(stems)}

        frame_labels = ["%s(DAS%s)" % (r.get("date", ""), r.get("das", ""))
                        for r in frames]
        frame_extras: List[Dict[str, Any]] = [{
            "plant_no": plant_no,
            "date": r.get("date", ""),
            "das": r.get("das", ""),
            "ts": stems[i],
            "acquisition_id": r.get("acquisition_id_x", ""),
        } for i, r in enumerate(frames)]

        masks = load_anchor_masks(anchor_dir)
        anchors: List[AnchorFrame] = []
        unmatched: List[str] = []
        for stem, mask in sorted(masks.items()):
            idx = stem_to_idx.get(stem)
            if idx is None:
                unmatched.append(stem)
                continue
            anchors.append(AnchorFrame(frame_idx=idx, mask=mask, source="manual"))

        if unmatched:
            msg = "%s: %d anchor(s) unmatched to index frames: %s" % (
                key, len(unmatched), unmatched[:5])
            if strict:
                raise ValueError(msg)
            _log("WARN " + msg)

        if len(anchors) < min_anchors:
            skipped.append({"plant_no": plant_no, "key": key,
                            "reason": "anchors=%d<%d" % (len(anchors), min_anchors),
                            "unmatched": len(unmatched)})
            continue

        anchors.sort(key=lambda a: a.frame_idx)
        sequences.append(PlantSequence(
            key=key,
            frame_paths=frame_paths,
            anchors=anchors,
            frame_labels=frame_labels,
            frame_extras=frame_extras,
            meta={
                "plant_no": plant_no,
                "anchor_dir": anchor_dir,
                "n_unmatched_anchors": len(unmatched),
                "dates": sorted({r.get("date", "") for r in frames}),
            },
        ))
        _log("%s frames=%d anchors=%d" % (key, len(frame_paths), len(anchors)))

    return sequences, skipped
