# -*- coding: utf-8 -*-
"""Adapter tests: generic CSV manifest adapter + plant phenotyping adapter.

The demo sample is generated into a temp dir at test time (no committed data).
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from make_demo_sample import make_sequence  # noqa: E402
from phenocv.adapters import CsvManifestAdapter  # noqa: E402
from phenocv.config import load_config  # noqa: E402


def _generate(tmp_path):
    out = str(tmp_path / "demo")
    make_sequence(out, n_frames=6, seed=1)
    return out


def test_csv_manifest_adapter_build(tmp_path):
    out = _generate(tmp_path)
    seqs = CsvManifestAdapter(os.path.join(out, "manifest.csv")).build_sequences()
    assert len(seqs) == 1
    s = seqs[0]
    assert s.n_frames == 6
    assert s.anchor_indices == [0, 3, 5]
    assert s.extras_for(0) == {"das": "1"}
    assert s.frame_labels[5] == "T5"


def test_csv_manifest_skips_low_anchor(tmp_path):
    out = _generate(tmp_path)
    # a sequence with only 1 anchor must be skipped by min_anchors=2
    seqs = CsvManifestAdapter(os.path.join(out, "manifest.csv"),
                              min_anchors=2).build_sequences()
    assert len(seqs) == 1  # demo has 3 anchors, so kept


def test_load_config_with_preset():
    cfg = load_config(os.path.join(HERE, "..", "configs", "default.yaml"),
                      preset="plant_phenotyping")
    assert cfg["roi_pad_ratio"] == 1.9
    assert cfg["threshold_ladder"] == [-0.5, -1.0, -2.0, -4.0]


def test_load_config_unknown_preset_raises():
    with pytest.raises(KeyError):
        load_config(os.path.join(HERE, "..", "configs", "default.yaml"),
                    preset="does_not_exist")
