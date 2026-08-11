# PhenoCV

> Open-source **vision toolkit for plant phenotyping** — starting with temporal canopy segmentation via SAM 2 video propagation.

[![PyPI version](https://img.shields.io/pypi/v/phenocv.svg)](https://pypi.org/project/phenocv/)
[![Python versions](https://img.shields.io/pypi/pyversions/phenocv.svg)](https://pypi.org/project/phenocv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml/badge.svg)](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub-blue.svg)](./docs/)

PhenoCV turns a **few manually labeled keyframes** of a plant sequence into a
**fully segmented time series** using [SAM 2](https://github.com/facebookresearch/sam2)
video propagation. It is built to be a *general* plant-phenotyping vision tool,
not a one-off script: the core engine is **data-source agnostic**, and any
dataset format plugs in through a thin **adapter**.

---

## 🌟 Why PhenoCV

- **Temporal consistency.** One manual label every few days is enough; SAM 2
  propagates it across the whole sequence with smooth, drift-free masks.
- **Small-seedling aware.** ROI cropping lifts a tiny seedling's effective
  resolution by ~10× (SAM 2 otherwise resizes every frame to 1024px and crushes
  early shoots out of existence).
- **Fail-closed, auditable.** Every frame records a `pred_source`
  (`manual` / `propagated` / `propagated_lowthr` / `point_rescue` / `failed_empty`)
  so you always know *how* a mask was produced.
- **Annotation-free QA.** Leave-One-Out (LOO) validation reports IoU / Boundary-F1
  on your real anchors without any extra labeling.
- **CPU-testable.** The whole logic layer (ROI math, threshold ladder, rescue,
  IoU/BF1, ISAT export) runs without CUDA — so CI and contributors never need a GPU.

## ✨ Features

| Area | What you get |
|---|---|
| Propagation | SAM 2 video model, bidirectional (fwd+rev) logit averaging |
| Recall safety | Threshold-ladder fallback + box-constrained point-rescue for empty frames |
| Resolution | Square ROI around the union of anchor masks, padded for growth |
| Quality | LOO IoU / Boundary-F1 report, `pred_source` provenance on every frame |
| Data | Pluggable adapters — generic CSV/JSON manifest (default) + potted-soybean example |
| Export | Full-image mask PNGs, ISAT annotation JSONs, `area.csv`, QA grid |
| Engineering | `pip`-installable, MIT, Python 3.10–3.13, CPU-only test suite |

---

## 💿 Installation

```bash
# Core (CPU-only, no torch needed for the engine + adapters + tests)
pip install phenocv

# From source, with dev tools (pytest)
pip install -e ".[dev]"

# Optional: GPU video propagation (SAM 2)
pip install "phenocv[video]"
```

> **Note:** The `video` extra pulls in `torch` + `sam2`. Running the actual
> segmentation needs a SAM 2 checkpoint (e.g. `sam2.1_hiera_l.pt`) and its
> model config (e.g. `sam2.1_hiera_l.yaml`, shipped with the `sam2` package).
> Everything else — adapters, config, CPU unit tests — works without it.

## 🚀 Quickstart

### 1. Generate the synthetic demo (no real data, CPU-only)

```bash
python tools/make_demo_sample.py --out samples/demo
```

This writes `samples/demo/{frames,masks,manifest.csv}` — a green disc that
grows over 6 frames with 3 sparse anchors.

### 2. Run segmentation

```bash
phenocv segment \
  --adapter csv \
  --manifest samples/demo/manifest.csv \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint /path/to/sam2.1_hiera_l.pt \
  --model-cfg sam2.1_hiera_l.yaml \
  --output results/demo \
  --device cuda
```

Outputs land under `results/demo/` (see [Outputs & QA](#-outputs--qa)).

### 3. Programmatic API

```python
from phenocv.adapters import CsvManifestAdapter
from phenocv.config import load_config
from phenocv.engine import run_sam2_video_temporal

sequences = CsvManifestAdapter("samples/demo/manifest.csv").build_sequences()
cfg = load_config("configs/default.yaml", preset="plant_phenotyping")

result = run_sam2_video_temporal(
    sequences,
    output_root="results/demo",
    checkpoint="/path/to/sam2.1_hiera_l.pt",
    model_cfg="sam2.1_hiera_l.yaml",
    device="cuda",
)
print(result["loo_summary_interior"])  # IoU / BF1 medians
```

### 4. CPU-only smoke test (no GPU, no SAM 2)

```bash
pip install -e ".[dev]"
pytest                      # 16 tests, all CPU
python -c "import phenocv; print(phenocv.__version__)"
```

---

## 🧩 Adapter Contract

The engine consumes `PlantSequence` objects. The default
`CsvManifestAdapter` reads **one manifest** and needs no dataset code:

| Column | Type | Required | Meaning |
|---|---|---|---|
| `sequence_key` | str | ✅ | sequence id, e.g. `plant_01` |
| `frame_idx` | int | ⚪ | 0-based temporal index (row order used if absent) |
| `frame_path` | str | ✅ | path to the RGB frame |
| `frame_label` | str | ⚪ | human-readable label (date / DAS) |
| `is_anchor` | 0/1 | ✅ | does this frame carry a manual mask? |
| `mask_path` | str | ⚪ | anchor mask PNG (required when `is_anchor=1`) |
| *(any other)* | — | ⚪ | carried through verbatim as `frame_extras` |

JSON manifests are also accepted (a list of row dicts, or `{"frames": [...]}`).
See [docs/adapter_guide.md](./docs/adapter_guide.md) to write your own adapter.

## 🔧 Presets

Presets live under the `presets:` block of `configs/default.yaml` and are
applied with `--preset <name>` (or `load_config(path, preset=...)`).

| Preset | Use case | Key knobs |
|---|---|---|
| `plant_phenotyping` | potted soybean temporal (reference) | generous ROI pad (1.9), threshold ladder on, rescue on |
| `rigid_object` | sharp-boundary, stable-scale objects | tight ROI (1.3), no ladder, no rescue, category `object` |
| `high_recall` | weak / easily-lost targets | larger ROI (2.2), deeper ladder (to −8.0), smaller `isat_min_area` |

Every `TemporalPropagationConfig` field is overridable — see
[docs/tuning.md](./docs/tuning.md).

## 🏗️ Architecture

```
phenocv/
├── engine.py          # core: ROI, propagation, threshold ladder, rescue,
│                      #        LOO, ISAT export  (data-source agnostic)
├── cli.py             # `phenocv segment` entry point
├── config.py          # YAML + preset loader
└── adapters/
    ├── base.py            # BaseAdapter (subclass to support new formats)
    ├── csv_manifest.py    # default generic CSV/JSON manifest adapter
    └── plant_phenotyping.py  # worked example: potted soybean
```

The engine never reads your filesystem layout directly — it only sees
`PlantSequence` objects. That separation is what keeps it reusable across
domains.

## 📊 Outputs & QA

`phenocv segment` writes, under `--output`:

```
<output>/
├── run_manifest.json        # full run record + LOO summary
├── loo_quality.csv          # per-anchor LOO: iou, bf1, pred_source
├── frame_manifest.csv       # every frame: pred_source, area, thr, extras
├── sequence_summary.csv     # per-sequence rollup (area first/last/max, counts)
└── <sequence_key>/
    ├── masks/<stem>.png     # full-image boolean masks (0/255)
    ├── jsons/<stem>.json    # ISAT annotation (if --no-isat not set)
    ├── area.csv             # per-frame area + pred_source + extras
    └── qa_grid.png          # overview grid of frames × masks (if --no-qa not set)
```

`pred_source` values and their meaning:

| `pred_source` | Meaning |
|---|---|
| `manual` | human-labeled anchor, copied through |
| `propagated` | SAM 2 propagation at base threshold (0.0) |
| `propagated_lowthr` | recovered via the threshold ladder (empty at base) |
| `point_rescue` | empty even after ladder; rescued with a box-constrained point |
| `failed_empty` | no mask found after all fallbacks |

## 📚 Documentation

- [docs/tuning.md](./docs/tuning.md) — every knob, and *why* the defaults are what they are
- [docs/export_formats.md](./docs/export_formats.md) — mask / ISAT / CSV / QA layout
- [docs/adapter_guide.md](./docs/adapter_guide.md) — write your own data adapter
- [SKILL.md](./SKILL.md) — agent skill (WorkBuddy / Claude Code / Codex)

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). CPU-only setup, run `pytest`, open a PR.

## 📜 Citation

```bibtex
@software{phenocv2026,
  title  = {PhenoCV: Open-source vision toolkit for plant phenotyping},
  author = {perseus-wy},
  year   = {2026},
  url    = {https://github.com/perseus-wy/PhenoCV},
  license = {MIT}
}
```

## 📄 License

[MIT](./LICENSE) © 2026 perseus-wy.
