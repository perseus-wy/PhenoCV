# PhenoCV

> Open-source **computer-vision toolkit for plant phenotyping** — a composable
> set of modules (temporal canopy segmentation, a 4-tier trait engine, and room
> to grow), all sharing one core.

[![PyPI version](https://img.shields.io/pypi/v/phenocv.svg)](https://pypi.org/project/phenocv/)
[![Python versions](https://img.shields.io/pypi/pyversions/pypi.svg)](https://pypi.org/project/phenocv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml/badge.svg)](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub-blue.svg)](./docs/)

PhenoCV is **not** a single algorithm — it is a toolbox of independent,
pluggable modules for plant phenotyping. Today it ships two:

- **`phenocv.segmentation`** — turn a *few* manually labeled keyframes of a
  plant sequence into a **fully segmented time series** using
  [SAM 2](https://github.com/facebookresearch/sam2) video propagation.
  The engine is data-source agnostic; any dataset format plugs in through a
  thin **adapter**.
- **`phenocv.phenotypes`** — a **4-tier, fail-closed trait engine** that turns a
  plant mask (+ optional RGB / depth / multispectral) into a flat table of
  traits: 2D shape → RGB vegetation indices → 3D height/volume → multispectral
  indices. It runs *every* registered extractor whose inputs you actually have.

Both modules build on a shared **`phenocv.core`** (the trait-extractor registry
+ IO helpers), so adding a third module (e.g. `phenocv.counting`) is just
"write a package, register your tools" — no core change.

---

## 🌟 Why PhenoCV

- **Composable, not monolithic.** Adopt the segmentation module, the trait
  engine, or both — each is self-contained and importable on its own.
- **Temporal consistency (segmentation).** One manual label every few days is
  enough; SAM 2 propagates it across the whole sequence with smooth,
  drift-free masks.
- **Small-seedling aware.** ROI cropping lifts a tiny seedling's effective
  resolution by ~10× (SAM 2 otherwise resizes every frame to 1024px and crushes
  early shoots out of existence).
- **Fail-closed, auditable traits.** Every trait row records `pred_source` /
  `_inputs` / `_extractors_run`, and unobservable outputs are `NaN` +
  `missing_reason` — never fabricated.
- **Annotation-free QA.** Leave-One-Out (LOO) validation reports IoU /
  Boundary-F1 on your real anchors without extra labeling.
- **CPU-testable.** The whole logic layer (ROI math, threshold ladder, rescue,
  IoU/BF1, the trait engine) runs without CUDA — so CI and contributors never
  need a GPU.

## ✨ Modules

| Module | What you get |
|---|---|
| `phenocv.segmentation` | SAM 2 video propagation, bidirectional (fwd+rev) logit averaging, threshold-ladder fallback + point-rescue, LOO IoU/BF1 QA, pluggable adapters, ISAT/CSV/QA export |
| `phenocv.phenotypes` | 4-tier trait engine: 2D shape (area/bbox/solidity…), RGB vegetation indices (ExG/ExR/VARI…), 3D height/volume (mm, needs depth+intrinsics), multispectral indices (12 + reflectance stats) |
| `phenocv.core` | Shared trait-extractor **registry** (`@register`) + minimal IO helpers (mask/RGB/depth/multispectral readers) used by every module |

See [Roadmap](#-roadmap) for what is planned next.

---

## 💿 Installation

```bash
# Core (CPU-only, no torch needed for the engine + adapters + trait engine + tests)
pip install phenocv

# From source, with dev tools (pytest)
pip install -e ".[dev]"

# Optional: GPU video propagation (SAM 2)
pip install "phenocv[video]"
```

> **Note:** The `video` extra pulls in `torch` + `sam2`. Running the actual
> segmentation needs a SAM 2 checkpoint (e.g. `sam2.1_hiera_l.pt`) and its
> model config (e.g. `sam2.1_hiera_l.yaml`, shipped with the `sam2` package).
> Everything else — adapters, config, the trait engine, CPU unit tests —
> works without it.

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

### 3. Compute traits from a mask

```bash
phenocv phenotype \
  --mask results/demo/plant_01/<stem>.png \
  --rgb  /local/mirror/rgb/<stem>.png \
  --depth /local/mirror/depth_mm/<stem>.png \
  --calibration configs/intrinsics_second.yaml \
  --out results/demo/traits/plant_01_<stem>.json
```

Or batch a whole deliverable (masks + optional RGB / depth / multispectral):

```bash
python tools/compute_phenotypes.py \
  --mask-dir results/demo/masks \
  --rgb-dir  /local/mirror/rgb \
  --depth-dir /local/mirror/depth_mm \
  --calibration configs/intrinsics_second.yaml \
  --out results/demo/phenotypes
```

### 4. Programmatic API

```python
from phenocv.segmentation.adapters import CsvManifestAdapter
from phenocv.segmentation.config import load_config
from phenocv.segmentation.engine import run_sam2_video_temporal

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

```python
import numpy as np
from phenocv.phenotypes import compute_traits

mask = (cv2.imread("mask.png", 0) > 0)
rgb = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)
row = compute_traits(mask=mask, rgb=rgb)   # only the extractors whose
print(row)                                  # inputs you pass will run
```

### 5. CPU-only smoke test (no GPU, no SAM 2)

```bash
pip install -e ".[dev]"
pytest                      # 30 tests, all CPU
python -c "import phenocv; print(phenocv.list_modules())"
```

---

## 🧩 Adapter Contract (segmentation)

The segmentation engine consumes `PlantSequence` objects. The default
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

## 🔧 Presets (segmentation)

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
├── __init__.py / __main__.py   # toolbox entry; phenocv.list_modules()
├── cli.py                      # `phenocv segment|phenotype|list-traits`
├── core/                       # SHARED BASE for every module
│   ├── registry.py             # TraitExtractor + @register + available_for
│   └── io.py                   # mask/RGB/depth/multispectral readers
├── segmentation/               # MODULE 1 — temporal canopy segmentation
│   ├── engine.py               # ROI, propagation, threshold ladder, rescue,
│   │                           #   LOO, ISAT export (data-source agnostic)
│   ├── config.py               # YAML + preset loader
│   └── adapters/               # BaseAdapter + CsvManifest + plant example
└── phenotypes/                 # MODULE 2 — 4-tier trait engine
    ├── base.py                 # re-export of phenocv.core.registry
    ├── compute_traits.py       # tier-orchestrated, fail-closed
    ├── shape2d.py / rgb_indices.py / geometry3d.py / multispectral.py
    └── calib.py                # camera intrinsics
```

The segmentation engine never reads your filesystem layout directly — it only
sees `PlantSequence` objects. The trait engine never hard-codes a trait — it
runs whatever is registered. That separation is what keeps PhenoCV reusable
and composable.

## 📊 Outputs & QA (segmentation)

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

## 🗺️ Roadmap

PhenoCV is a growing toolbox. Planned / welcome modules (each a sibling package
under `phenocv/`, each registering its tools with `phenocv.core.registry`):

- **`phenocv.counting`** — flower / fruit / tiller counting from masks.
- **`phenocv.disease`** — lesion / symptom segmentation and scoring.
- **`phenocv.growth`** — stage classification and growth-curve fitting from the
  trait engine's long tables.

## 📚 Documentation

- [docs/tuning.md](./docs/tuning.md) — every segmentation knob, and *why* the defaults are what they are
- [docs/export_formats.md](./docs/export_formats.md) — mask / ISAT / CSV / QA layout
- [docs/adapter_guide.md](./docs/adapter_guide.md) — write your own data adapter
- [SKILL.md](./SKILL.md) — agent skill (WorkBuddy / Claude Code / Codex)
- [skills/phenocv-phenotype-port/](./skills/phenocv-phenotype-port/) — WorkBuddy skill for porting a phenotype pipeline into PhenoCV

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). CPU-only setup, run `pytest`, open a PR.

## 📜 Citation

```bibtex
@software{phenocv2026,
  title  = {PhenoCV: Open-source computer-vision toolkit for plant phenotyping},
  author = {perseus-wy},
  year   = {2026},
  url    = {https://github.com/perseus-wy/PhenoCV},
  license = {MIT}
}
```

## 📄 License

[MIT](./LICENSE) © 2026 perseus-wy.
