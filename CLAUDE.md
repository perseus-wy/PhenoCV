# CLAUDE.md — PhenoCV (for Claude Code)

PhenoCV is a **composable open-source computer-vision toolkit for plant
phenotyping** — a set of independent, pluggable modules that share one
`phenocv.core`. Three modules ship today:
(1) **`phenocv.segmentation`** — temporal canopy segmentation: a few manually
labeled keyframes of a plant/object sequence are propagated to a fully
segmented time series with SAM 2 video propagation;
(2) **`phenocv.phenotypes`** — a pluggable, fail-closed trait engine that
derives 2D shape / RGB vegetation / 3D canopy height / multispectral traits
from a mask (+ optional richer inputs);
(3) **`phenocv.thermal`** — a pure-CPU thermal (FLIR) phenotyping module
(io / traits / environment / stress + an optional lazy SAM 2 segmentation
layer): temperature traits, canopy-layer partitioning by relative height,
environment-sensor time alignment, and before/after stress / rewatering
analysis with bootstrap & HAC uncertainty. **The thermal core is importable and
testable without a GPU** (`numpy` + `cv2` + `pandas` + `scipy` + `statsmodels`,
lazy; `torch`/`sam2` only inside the optional segmentation layer).

> **中文定位：** PhenoCV 是一个可组合的开源植物表型计算机视觉工具箱，由共享 `phenocv.core`
> 的独立模块构成。已发布 `phenocv.segmentation`（SAM 2 时序分割）、`phenocv.phenotypes`
> （四级 fail-closed 表型引擎）与 `phenocv.thermal`（纯 CPU 热红外表型模块）。完整中英双语说明见
> [`SKILL.md`](./SKILL.md)。

**Agent skill:** see [`SKILL.md`](./SKILL.md) for the full trigger conditions,
concepts, CLI/API usage, adapter contract, presets, phenotyping tiers, and QA
conventions. This file is a thin project-instructions entry point — do not
duplicate the skill body here.

## Quick commands

```bash
pip install -e ".[dev]"                       # CPU-only core + tests
pytest                                        # 30 CPU tests (segmentation + phenotypes)
python tools/make_demo_sample.py --out samples/demo   # synthetic sample (no real data)

# Real segmentation (needs torch + sam2 + a SAM 2 checkpoint):
phenocv segment --adapter csv --manifest samples/demo/manifest.csv \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint /path/to/sam2.1_hiera_l.pt --model-cfg sam2.1_hiera_l.yaml \
  --output results/demo --device cuda

# Phenotype computation (CPU-only, numpy+cv2, no torch):
phenocv list-traits -v                        # registry: extractors + input contracts
phenocv phenotype --mask mask.png [--rgb rgb.png] \
  [--depth depth_mm.png --calibration intrinsics.yaml] \
  --out traits.json [--csv traits.csv]
```

## Contribution / test conventions

- **Multi-module layout.** Each capability lives under `src/phenocv/<module>/`
  as a sibling package; the shared base is `src/phenocv/core/` (registry +
  IO). The segmentation engine is `src/phenocv/segmentation/engine.py`.
- Engine logic (`src/phenocv/segmentation/engine.py`) must stay **importable
  without CUDA**; `torch`/`sam2` are imported lazily inside `Sam2VideoPropagator`.
- Keep the **core engine data-source agnostic** — never read a dataset layout
  directly in `engine.py`; add an adapter under `src/phenocv/segmentation/adapters/`.
- Every frame must carry a `pred_source` provenance tag (manual / propagated /
  propagated_lowthr / point_rescue / failed_empty).
- **Phenotypes stay CPU-only** (numpy + cv2, no torch) and **fail-closed** —
  missing/unobservable → `NaN` + `missing_reason` (or `<name>_error`), never a
  fabricated value. Add a new trait by writing one `TraitExtractor` subclass and
  decorating it with `@register`; the orchestrator picks it up — do not modify it.
- To add a **new module** (e.g. `phenocv.counting`): create the package, and
  register its tools with `phenocv.core.registry.register` so they become
  visible to `phenocv.list_modules()` / the CLI automatically.
- Run `pytest` (CPU-only) before opening a PR. See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Repo map

```
src/phenocv/{__init__,__main__,cli}.py        # toolbox entry; phenocv.list_modules()
src/phenocv/core/                             # SHARED BASE for every module
  registry.py                                 #   TraitExtractor + @register + available_for
  io.py                                       #   mask/RGB/depth/multispectral readers
src/phenocv/segmentation/                     # MODULE 1 — temporal canopy segmentation
  engine.py config.py                         #   ROI/propagation/ladder/rescue/LOO/ISAT
  adapters/{base,csv_manifest,plant_phenotyping}.py
src/phenocv/phenotypes/                       # MODULE 2 — 4-tier trait engine (CPU-only, no torch)
  base.py                                     #   re-exports phenocv.core.registry
  shape2d.py rgb_indices.py                   #   L1 mask-only / L2 mask+rgb
  geometry3d.py calib.py                      #   L3 mask+depth+calibration (plant height, mm)
  multispectral.py                            #   L4 mask+multispectral (MS400 4-band indices)
  compute_traits.py                           #   orchestrator: available_for → merge rows, per-extractor try/except
src/phenocv/thermal/                          # MODULE 3 — pure-CPU thermal (FLIR) phenotyping
  config.py io.py traits.py environment.py stress.py   #   core (no torch at import)
  segmentation.py                             #   optional SAM 2 layer; torch/sam2 lazy-imported
configs/default.yaml          # propagation params + presets
tests/                        # CPU-only unit + orchestration tests (incl. test_phenotypes.py)
tools/make_demo_sample.py     # synthetic sample generator (no real data)
tools/compute_phenotypes.py   # batch trait computation over a mask deliverable
docs/{tuning,export_formats,adapter_guide}.md
skills/phenocv-phenotype-port/  # WorkBuddy skill: port a phenotype pipeline into PhenoCV
```

## Phenotype engine — how it fits together

`phenocv.phenotypes.compute_traits(mask=..., rgb=..., depth=...,
calibration=..., multispectral=...)` builds the set of available inputs, calls
`available_for(inputs)` to select every registered extractor whose `requires`
is a subset (sorted by tier), runs each in a try/except, and merges the rows.
Tiers: **1** `shape2d` (mask) · **2** `rgb_vegetation_indices` (mask+rgb) ·
**3** `canopy_3d_geometry` (mask+depth+calibration, plant height in mm) ·
**4** `multispectral_vegetation_indices` (mask+multispectral). Extending =
one `@register`-decorated `TraitExtractor`; never edit the orchestrator. Full
API/CLI/tier reference lives in `SKILL.md` → "Phenotyping / 表型计算".
