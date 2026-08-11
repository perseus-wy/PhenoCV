# -*- coding: utf-8 -*-
"""Generic CSV/JSON manifest adapter — the default, data-source-agnostic path.

Write one manifest describing your sequences and their sparse anchor masks;
the engine consumes it directly, with no dataset-specific code required.

Manifest columns
----------------
``sequence_key`` : str   sequence id (e.g. ``plant_01``)
``frame_idx``    : int   0-based temporal index (row order used if absent)
``frame_path``   : str   path to the RGB frame
``frame_label``  : str   optional human-readable label (date / DAS)
``is_anchor``    : 0/1   whether this frame carries a manual mask
``mask_path``    : str   anchor mask PNG path (required when ``is_anchor=1``)
``...``          : any other columns are carried through as ``frame_extras``

JSON manifests are also accepted: either a list of row dicts, or an object
with a ``"frames"`` list of row dicts.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np

from .base import BaseAdapter
from ..engine import AnchorFrame, PlantSequence

_TRUE = {"1", "true", "t", "yes", "y"}
_CORE_COLS = ("sequence_key", "frame_idx", "frame_path", "frame_label",
              "is_anchor", "mask_path")


class CsvManifestAdapter(BaseAdapter):
    """Build sequences from a single manifest CSV/JSON file."""

    def __init__(self, manifest_path: str, min_anchors: int = 2) -> None:
        self.manifest_path = manifest_path
        self.min_anchors = min_anchors

    def build_sequences(self, **kwargs) -> List[PlantSequence]:
        rows = self._read_rows()
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            key = r.get("sequence_key", "").strip()
            if not key:
                raise ValueError("manifest row missing 'sequence_key': %r" % r)
            grouped.setdefault(key, []).append(r)

        sequences: List[PlantSequence] = []
        for key, frames in grouped.items():
            if frames[0].get("frame_idx", "").strip():
                frames.sort(key=lambda r: int(float(r["frame_idx"])))

            frame_paths = [r["frame_path"] for r in frames]
            labels = [r.get("frame_label", "") or str(i) for i, r in enumerate(frames)]
            extras = [
                {k: v for k, v in row.items() if k not in _CORE_COLS}
                for row in frames
            ]

            anchors: List[AnchorFrame] = []
            for i, r in enumerate(frames):
                flag = str(r.get("is_anchor", "0")).strip().lower()
                if flag not in _TRUE:
                    continue
                mp = r.get("mask_path", "").strip()
                if not mp:
                    raise ValueError(
                        "%s frame %d is an anchor but has no mask_path" % (key, i))
                img = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError("Cannot read anchor mask: %s" % mp)
                anchors.append(AnchorFrame(frame_idx=i, mask=img > 127, source="manual"))
            anchors.sort(key=lambda a: a.frame_idx)

            if len(anchors) < self.min_anchors:
                continue

            sequences.append(PlantSequence(
                key=key,
                frame_paths=frame_paths,
                anchors=anchors,
                frame_labels=labels,
                frame_extras=extras,
            ))
        return sequences

    def _read_rows(self) -> List[Dict[str, str]]:
        ext = os.path.splitext(self.manifest_path)[1].lower()
        if ext == ".json":
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "frames" in data:
                return data["frames"]
            return data if isinstance(data, list) else []
        with open(self.manifest_path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
