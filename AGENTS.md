# AGENTS.md — PhenoCV (for OpenAI Codex)

PhenoCV is a **composable open-source computer-vision toolkit for plant
phenotyping** — a set of independent, pluggable modules that share one
`phenocv.core`. Two modules ship today:
(1) **`phenocv.segmentation`** — temporal canopy segmentation: a few manually
labeled keyframes of a plant/object sequence are propagated to a fully
segmented time series with SAM 2 video propagation;
(2) **`phenocv.phenotypes`** — a pluggable, fail-closed trait engine that
derives 2D shape / RGB vegetation / 3D canopy height / multispectral traits
from a mask (+ optional richer inputs).

**Agent skill:** see [`SKILL.md`](./SKILL.md) for the full trigger conditions,
concepts, CLI/API usage, adapter contract, presets, phenotyping tiers, and QA
conventions. This file is a thin entry point — do not duplicate the skill body here.

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
  without CUDA**; `torch`/`sam2` are imported lazily inside
  `Sam2VideoPropagator`.
- Keep the **core engine data-source agnostic** — never read a dataset layout
  directly in `engine.py`; add an adapter under `src/phenocv/segmentation/adapters/`.
- Every frame must carry a `pred_source` provenance tag (manual / propagated /
  propagated_lowthr / point_rescue / failed_empty).
- **Phenotypes stay CPU-only** (numpy + cv2, no torch) and **fail-closed** —
  missing/unobservable → `NaN` + `missing_reason` (or `<name>_error`), never a
  fabricated value. Add a new trait by writing one `TraitExtractor` subclass and
  decorating it with `@register`; do not modify the orchestrator.
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
configs/default.yaml          # propagation params + presets
tests/                        # CPU-only unit + orchestration tests (incl. test_phenotypes.py)
tools/make_demo_sample.py     # synthetic sample generator (no real data)
tools/compute_phenotypes.py   # batch trait computation over a mask deliverable
docs/{tuning,export_formats,adapter_guide}.md
skills/phenocv-phenotype-port/  # WorkBuddy skill: port a phenotype pipeline into PhenoCV
```
