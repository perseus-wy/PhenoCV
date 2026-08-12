> 📖 Chinese (中文) documentation: [README.zh-CN.md](./README.zh-CN.md)

# PhenoCV

> A composable, multi-modal **plant-phenotyping** toolkit — modules share one
> `phenocv.core`; [SAM 2](https://github.com/facebookresearch/sam2) is one
> *optional* segmentation backend, not the project's identity. Three modules
> ship today: `segmentation`, `phenotypes`, and `thermal`.


[![PyPI version](https://img.shields.io/pypi/v/phenocv.svg)](https://pypi.org/project/phenocv/)
[![Python versions](https://img.shields.io/pypi/pyversions/phenocv.svg)](https://pypi.org/project/phenocv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml/badge.svg)](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub-blue.svg)](./docs/)


PhenoCV is **not** a single algorithm — it is a toolbox of independent,
pluggable modules for plant phenotyping. Today it ships three, presented as
equals:

- **`phenocv.segmentation`** — turn a *few* manually labeled keyframes of a
  plant sequence into a **fully segmented time series**. The default backend
  uses [SAM 2](https://github.com/facebookresearch/sam2) video propagation, but
  the engine is data-source agnostic; any dataset format plugs in through a
  thin **adapter**. SAM 2 is one backend among several (Classical and YOLO
  backends are designed as drop-in alternatives — see [Architecture](#-architecture)).
- **`phenocv.phenotypes`** — a **4-tier, fail-closed trait engine** that turns a
  plant mask (+ optional RGB / depth / multispectral) into a flat table of
  traits: 2D shape → RGB vegetation indices → 3D height/volume → multispectral
  indices. It runs *every* registered extractor whose inputs you actually have.
- **`phenocv.thermal`** — a **pure-CPU thermal (FLIR) phenotyping** module:
  per-pixel temperature traits, upper/middle/lower canopy-layer partitioning by
  relative height, environment-sensor time alignment, and before/after
  stress / rewatering analysis with block-bootstrap & HAC uncertainty. Only
  `numpy` + `cv2` + `pandas` + `scipy` + `statsmodels` (last two lazy); no GPU
  needed for the core. An optional SAM 2 temporal-segmentation layer is lazy.

All three modules build on a shared **`phenocv.core`** (the trait-extractor
registry + a modality/adapter registry + IO helpers), so adding another module
(e.g. `phenocv.counting`) is just "write a package, register your tools" — no
core change.







---

## 🌟 Why PhenoCV


- **Composable, not monolithic.** Adopt the segmentation module, the trait
  engine, the thermal module, or any combination — each is self-contained and
  importable on its own.
- **Fail-closed, auditable traits.** Every trait row records `pred_source` /
  `_inputs` / `_extractors_run`, and unobservable outputs are `NaN` +
  `missing_reason` — never fabricated.
- **Annotation-free QA.** Leave-One-Out (LOO) validation reports IoU /
  Boundary-F1 on your real anchors without extra labeling (segmentation) and
  provenance-tagged outputs everywhere.
- **Small-seedling aware.** ROI cropping lifts a tiny seedling's effective
  resolution by ~10× (a propagation backend otherwise resizes every frame to
  1024px and crushes early shoots out of existence).
- **CPU-testable.** The whole logic layer (ROI math, threshold ladder, rescue,
  IoU/BF1, the trait engine, the thermal stack) runs without CUDA — so CI and
  contributors never need a GPU.
- **Backend-agnostic segmentation.** SAM 2 is the default but not mandatory; the
  `BaseSegmenter` contract lets Classical and YOLO backends drop in.



![ROI cropping lifts a small seedling's effective resolution ~10× (synthetic demo)](docs/assets/fig2_roi_crop_benefit.png)

---

## ✨ Modules


| Module | What you get |
|---|---|
| `phenocv.segmentation` | Temporal canopy segmentation with pluggable backends (SAM 2 default; Classical / YOLO designed as drop-ins). Bidirectional (fwd+rev) logit averaging, threshold-ladder fallback + point-rescue, LOO IoU/BF1 QA, pluggable adapters, ISAT/CSV/QA export |
| `phenocv.phenotypes` | 4-tier trait engine: 2D shape (area/bbox/solidity…), RGB vegetation indices (ExG/ExR/VARI…), 3D height/volume (mm, needs depth+intrinsics), multispectral indices (12 + reflectance stats) |
| `phenocv.thermal` | Pure-CPU FLIR phenotyping: temperature traits (`temp_*` keys), canopy-layer (upper/middle/lower) temperatures by relative height, canopy ΔT vs ambient, environment-sensor alignment (no extrapolation, gap-guarded), before/after stress analysis (block-bootstrap CI + HAC + light/dark control); optional SAM 2 segmentation layer |
| `phenocv.core` | Shared trait-extractor **registry** (`@register`) + a modality/adapter **registry** + minimal IO helpers (mask/RGB/depth/multispectral/thermal readers) used by every module |



![The 4-tier phenotype engine: from a single mask (+ optional RGB / depth / multispectral) to a flat trait table](docs/assets/fig3_four_tiers.png)


---

## 💿 Installation


```bash
# Core (CPU-only, no torch needed for the engine + adapters + trait engine + tests)
pip install phenocv

# From source, with dev tools (pytest)
pip install -e ".[dev]"

# Optional: GPU video propagation (SAM 2 backend)
pip install "phenocv[video]"
```

> **Note:** The `video` extra pulls in `torch` + `sam2` for the SAM 2 *segmentation
> backend*. Running the actual segmentation needs a SAM 2 checkpoint (e.g.
> `sam2.1_hiera_l.pt`) and its model config (e.g. `sam2.1_hiera_l.yaml`, shipped
> with the `sam2` package). Everything else — adapters, config, the trait
> engine, the thermal stack, CPU unit tests — works without it. The other
> segmentation backends (Classical / YOLO) are being added behind the same
> `BaseSegmenter` contract and do not require SAM 2.






---

## 🚀 Quickstart


PhenoCV's three modules are independent — run whichever you need.

### 1. Generate the synthetic demo (no real data, CPU-only)

```bash
python tools/make_demo_sample.py --out samples/demo
```

This writes `samples/demo/{frames,masks,manifest.csv}` — a green disc that
grows over 6 frames with 3 sparse anchors.

### 2. Segmentation (one of several module entry points)

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

![Temporal propagation: a few sparse anchors expand to a fully segmented sequence (synthetic demo)](docs/assets/fig1_temporal_propagation.png)

Outputs land under `results/demo/` (see [Outputs & QA](#-outputs--qa)).

### 3. Phenotype traits from a mask

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
# segmentation
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
# phenotypes
import cv2
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

## 🧩 Adapter Contract


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
The same adapter idea is generalized by the `phenocv.core` **modality/adapter
registry**, so any module can declare how it ingests a new data source.





---

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





---

## 🏗️ Architecture


```
phenocv/
├── __init__.py / __main__.py   # toolbox entry; phenocv.list_modules()
├── cli.py                      # `phenocv segment|phenotype|list-traits`
├── core/                       # SHARED BASE for every module
│   ├── registry.py             # TraitExtractor registry + @register + available_for,
│   │                           #   and a modality/adapter registry (new data source = new adapter)
│   └── io.py                   # mask/RGB/depth/multispectral/thermal readers
├── segmentation/               # MODULE — temporal canopy segmentation
│   ├── base.py                 # BaseSegmenter: pluggable backends (SAM2 / Classical / YOLO)
│   ├── engine.py               # ROI, propagation, threshold ladder, rescue,
│   │                           #   LOO, ISAT export (data-source agnostic)
│   ├── config.py               # YAML + preset loader
│   └── adapters/               # BaseAdapter + CsvManifest + plant example
├── phenotypes/                 # MODULE — 4-tier trait engine
│   ├── base.py                 # re-export of phenocv.core.registry
│   ├── compute_traits.py       # tier-orchestrated, fail-closed
│   ├── shape2d.py / rgb_indices.py / geometry3d.py / multispectral.py
│   └── calib.py                # camera intrinsics
└── thermal/                    # MODULE — pure-CPU FLIR phenotyping (full sibling)
    ├── config.py io.py traits.py environment.py stress.py   # core (no torch at import)
    └── segmentation.py         # optional SAM 2 layer; torch/sam2 lazy-imported
```

**Extension points.** PhenoCV is built to be extended without forking:

- **`phenocv.core.registry`** — the `TraitExtractor` registry (`@register` +
  `available_for`). Any module registers its tools here; they become visible to
  `phenocv.list_modules()` / the CLI automatically.
- **Modality / adapter registry** (in `core/registry.py`) — declare how a
  module ingests a new data source; the CSV/JSON adapter is one implementation.
- **`segmentation/base.py BaseSegmenter`** — a backend-agnostic contract. SAM 2
  is the shipped backend; **Classical** and **YOLO** backends are designed as
  drop-in alternatives behind the same interface, so swapping backends never
  touches the engine or the adapters.

The segmentation engine never reads your filesystem layout directly — it only
sees `PlantSequence` objects. The trait engine never hard-codes a trait — it
runs whatever is registered. That separation is what keeps PhenoCV reusable
and composable.






---

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

![Annotation-free QA: provenance (`pred_source`) distribution and Leave-One-Out IoU across anchors](docs/assets/fig5_qa_provenance.png)







---

## 🌡️ Thermal (FLIR) phenotyping


`phenocv.thermal` is a **pure-CPU** thermal (infrared) phenotyping module: it
turns a true-temperature matrix (`°C`) + a canopy mask into a flat table of
temperature traits and lets you align environment sensors onto frame timestamps
and analyse before/after stress / rewatering responses. Design contract is the
same as the trait engine — **fail-closed** (missing / unobservable / empty →
`NaN` + `missing_reason`, never fabricated) and **data-agnostic** (caller
supplies paths & column maps). The core (`io` / `traits` / `environment` /
`stress`) imports and runs with **no GPU and no torch**; only the optional
`segmentation` sub-layer lazy-imports `torch`/`sam2`.

> **Python-API only:** thermal is not yet wired into the `phenocv` CLI — call it
> from Python.

![A FLIR scene: the true-temperature matrix over a plant canopy.](docs/assets/fig_thermal_scene.png)

![Temperature overlay on the canopy mask (cv2-only render, fixed scale).](docs/assets/fig_thermal_overlay.png)

![Canopy-layer partitioning: upper / middle / lower by relative height.](docs/assets/fig_thermal_layers.png)

![Environment-sensor alignment onto frame timestamps (no extrapolation, gap-guarded).](docs/assets/fig_thermal_envalign.png)

![Before/after stress response around a rewatering event (block-bootstrap CI + HAC).](docs/assets/fig_thermal_stress.png)

```python
import numpy as np
import phenocv.thermal as thermal

# io: read a true-temperature matrix + build a 3ch feature image for SAM2 prompts
temperature = np.load("stem_temp.npy").astype("float32")     # true °C matrix
feat = thermal.thermal_feature_image(temperature)             # abs / local-ΔT / gradient
mask = thermal.polygons_to_mask((H, W), polygons)

# traits: one call runs only the extractors whose inputs you pass
row = thermal.compute_thermal_traits(mask=mask, temperature=temperature, ambient=23.0)
# -> canopy_temp_median_c, canopy_upper_median_c, canopy_delta_t_c, ...
# missing ambient -> canopy_delta_t_c = NaN + missing_reason (never fabricated)

# environment: align sensors onto frame timestamps (no extrapolation; gap-guarded)
env = thermal.read_environment_workbook(
    "environment.xlsx",
    column_map={"DateTime": "timestamp", "AirTemp": "ambient_c", "CO2": "co2_ppm"})
aligned = thermal.align_environment_to_frames(
    frame_timestamps, env, ["ambient_c", "co2_ppm"],
    max_gap_sec=600.0, timezone="UTC")        # out-of-range -> NaN + qc_flag

# stress: before/after contrast around an event, with block-bootstrap CI +
# covariate-adjusted HAC regression and a dark-phase internal negative control
result = thermal.analyze_stress_response(
    timeseries_df, event_time, metric="canopy_temp_c",
    phase_column="phase", lit_value="light", random_seed=42)
```

SAM 2 temporal thermal segmentation (`ThermalVideoSegmenter` /
`segment_video_with_sam2`) needs `pip install "phenocv[video]"` plus a SAM 2
checkpoint; `thermal_feature_image` is fed to SAM 2 as the 3-channel input (not
a pseudo-color frame), and cleanup is target-anchored so an engulfed pot is
never published (fail-closed).















---

## 🗺️ Roadmap


PhenoCV is a growing toolbox organized around **plant-phenotyping research
hotspots**. Each item below is a planned / welcome sibling module under
`phenocv/`, registering its tools with `phenocv.core.registry`. Existing roadmap
ideas are mapped into these groups.

### Multi-modal fusion
Combine RGB + thermal + multispectral + hyperspectral + LiDAR/depth into a
single phenotypic view. The three shipped modules already share one IO/registry
core to make this tractable.

### Stress & disease (biotic / abiotic)
Drought, heat, and other abiotic stress scoring; lesion / symptom segmentation
and disease scoring (`phenocv.disease`, formerly a standalone roadmap item) with
the same fail-closed, annotation-light contract as thermal.

### Growth & development
Stage classification and growth-curve fitting from the trait engine's long
tables (`phenocv.growth`, formerly a standalone roadmap item).

### Organ counting
Flower / fruit / tiller / panicle counting from masks (`phenocv.counting`,
formerly a standalone roadmap item) — reuse the segmentation + trait registries.

### Root / rhizosphere
Below-ground imaging and root-system traits; a natural extension of the
registry-driven trait engine to a new modality.

### Postharvest quality
Fruit / grain quality traits (color, defects, texture) reusing the phenotypes
tier machinery on a new data source via an adapter.

### Yield & G2P (genotype-to-phenotype)
Breeding-oriented roll-ups: linking high-throughput phenotypes to genotype,
trial design, and selection indices.

### Foundation models
Prompt-based segmentation / trait models (SAM-family and successors) behind the
`BaseSegmenter` contract — SAM 2 is the first such backend; more are drop-in.

### Edge / lightweight deployment
CPU/ONNX/TFLite paths so modules run on edge cameras and field laptops without
a GPU (the thermal module already leads here).

### Uncertainty estimation / interpretability
Per-trait confidence (bootstrap / HAC, already in thermal), saliency, and
provenance tooling reusable across modules.

![Population curves: mask-only canopy area / vertical extent across all 44 plants
(real, desensitized crops) — one uniform call scales from a single seedling frame to a whole population](docs/assets/fig4_population_curves.png)














---

## 📚 Documentation


- [docs/tuning.md](./docs/tuning.md) — every segmentation knob, and *why* the defaults are what they are
- [docs/export_formats.md](./docs/export_formats.md) — mask / ISAT / CSV / QA layout
- [docs/adapter_guide.md](./docs/adapter_guide.md) — write your own data adapter
- [SKILL.md](./SKILL.md) — agent skill (WorkBuddy / Claude Code / Codex)
- [skills/phenocv-phenotype-port/](./skills/phenocv-phenotype-port/) — WorkBuddy skill for porting a phenotype pipeline into PhenoCV



---

## 🤝 Contributing


See [CONTRIBUTING.md](./CONTRIBUTING.md) and [docs/extending.md](./docs/extending.md).
CPU-only setup, run `pytest`, open a PR.



---

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



---

## 📄 License


[MIT](./LICENSE) © 2026 perseus-wy.
