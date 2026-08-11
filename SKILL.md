---
name: phenocv
description: >-
  PhenoCV — composable open-source computer-vision toolkit for plant
  phenotyping. Modules: (1) `phenocv.segmentation` — temporal canopy
  segmentation via SAM 2 video propagation from sparse manual keyframes; (2)
  `phenocv.phenotypes` — a pluggable, fail-closed **phenotype computation**
  engine that derives 2D shape, RGB-vegetation, 3D-canopy-height, and
  multispectral traits from a mask (+ optional RGB / depth+calibration /
  multispectral inputs); both build on `phenocv.core` (shared registry + IO).
  Use when a user wants to (a) segment a plant/object time-series from a few
  labeled frames, (b) build a phenotyping mask dataset, (c) write a data
  adapter for a new dataset format, (d) tune ROI / threshold / rescue
  parameters, (e) compute phenotypic traits (canopy area, vegetation indices,
  plant height, multispectral indices) from existing masks, or (f) extend
  PhenoCV with a new module/tool.
  PhenoCV —— 可组合的开源植物表型计算机视觉工具箱。模块：① `phenocv.segmentation`
  —— 基于 SAM 2 视频传播的时序冠层分割；② `phenocv.phenotypes` —— 可插拔、
  fail-closed 的**表型计算**引擎；二者都构建在 `phenocv.core`（共享注册表+IO）
  之上。当用户需要（a）用少量标注帧分割植物/物体时序，（b）构建表型掩膜数据集，
  （c）为新数据集格式编写适配器，（d）调整 ROI / 阈值 / 救援参数，（e）基于已有
  掩膜计算表型（冠层面积、植被指数、株高、多光谱指数），或（f）为 PhenoCV
  扩展新模块/工具时使用。
---

# PhenoCV Skill

> Composable open-source computer-vision toolkit for plant phenotyping. Two
> modules ship today — `phenocv.segmentation` (SAM 2 temporal canopy
> segmentation) and `phenocv.phenotypes` (a 4-tier, fail-closed trait
> engine) — both on a shared `phenocv.core`. 可组合的开源植物表型计算机
> 视觉工具箱：当前含 `phenocv.segmentation`（SAM 2 时序分割）与
> `phenocv.phenotypes`（四级 fail-closed 表型引擎），共享 `phenocv.core`。

## When to use / 何时触发

- User has a **sequence of plant/object images over time** and wants a mask for
  every frame. 用户有一组**随时间变化的植物/物体图像序列**，想要每帧的掩膜。
- User wants **annotation-free accuracy** (LOO IoU / Boundary-F1) on their own
  anchors. 用户想要对其锚帧做**无需额外标注的精度评估**（LOO IoU / Boundary-F1）。
- User is building a **phenotyping dataset** (canopy area over time, growth
  curves). 用户正在构建**表型数据集**（时序冠层面积、生长曲线）。
- User needs to **plug in a new dataset format** (write an adapter). 用户需要
  **接入新的数据集格式**（编写适配器）。
- User has **masks (+ optionally RGB / depth+calibration / multispectral)** and
  wants **phenotypic traits** — canopy area & shape, RGB vegetation indices,
  plant height, multispectral indices. 用户已有**掩膜（+ 可选 RGB / 深度+标定 /
  多光谱）**，想要**表型指标** —— 冠层面积与形状、RGB 植被指数、株高、多光谱指数。
- User wants to **add a new trait/algorithm** to the phenotype engine. 用户想
  给表型引擎**新增一种表型/算法**。

## Core concepts / 核心概念

- **Anchor frame** — a human-labeled keyframe mask. Sparse anchors (a few per
  sequence) are enough. **锚帧** —— 人工标注的关键帧掩膜，只需少量。
- **ROI cropping** — SAM 2 resizes frames to 1024px, crushing small seedlings;
  we crop a padded square ROI around the anchor union bbox first. **ROI 裁剪** ——
  先把锚帧并集 bbox 周围的方形 ROI 裁出，保住幼苗分辨率。
- **Bidirectional propagation** — forward + reverse passes, logits averaged.
  **双向传播** —— 正向+反向，logits 取平均，抑制端点漂移。
- **Threshold ladder + point-rescue** — when the base threshold yields empty,
  step the threshold down, then (last resort) a box-constrained point prompt.
  **阈值阶梯 + 点救援** —— 基础阈值判空时逐步降低阈值；最后兜底用带框约束的点提示。
- **`pred_source` provenance** — every frame tagged `manual` / `propagated` /
  `propagated_lowthr` / `point_rescue` / `failed_empty`. **溯源** —— 每帧标注
  其产生方式，便于审计与人工复核。

## Install / 安装

```bash
pip install -e ".[dev]"          # CPU-only core + tests
pip install "phenocv[video]"     # + torch + sam2 for actual GPU propagation
```

Running real segmentation needs a SAM 2 checkpoint (e.g. `sam2.1_hiera_l.pt`)
and its model config (`sam2.1_hiera_l.yaml`).

## CLI / 命令行

```bash
# Generic CSV/JSON manifest (default adapter)
phenocv segment --adapter csv --manifest manifest.csv \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint /path/to/sam2.1_hiera_l.pt --model-cfg sam2.1_hiera_l.yaml \
  --output results/run --device cuda

# Potted-soybean example adapter
phenocv segment --adapter plant \
  --index data/frame_index.csv --anchor-root data/manual_masks \
  --rgb-root /local/mirror --config configs/default.yaml \
  --checkpoint /path/to/sam2.1_hiera_l.pt --output results/run

# Flags: --no-loo / --no-isat / --no-qa / --no-resume / --min-anchors N
#        / --image-size H W / --preset <name>
```

## Python API / 编程接口

```python
from phenocv.segmentation.adapters import CsvManifestAdapter
from phenocv.segmentation.config import load_config
from phenocv.segmentation.engine import run_sam2_video_temporal

seqs = CsvManifestAdapter("manifest.csv").build_sequences()
cfg = load_config("configs/default.yaml", preset="plant_phenotyping")
result = run_sam2_video_temporal(
    seqs, output_root="results/run",
    checkpoint="/path/to/sam2.1_hiera_l.pt",
    model_cfg="sam2.1_hiera_l.yaml", device="cuda")
print(result["loo_summary_interior"])   # {'iou_median':..., 'bf1_median':...}
```

## Adapter contract / 适配器契约

Engine consumes `PlantSequence(key, frame_paths, anchors, frame_labels,
frame_extras)`. Default `CsvManifestAdapter` reads one manifest:
`sequence_key, frame_idx, frame_path, frame_label, is_anchor, mask_path`
(+ any extra columns → `frame_extras`). Subclass `BaseAdapter.build_sequences`
for new formats. See `docs/adapter_guide.md`.

## Presets / 预设

`plant_phenotyping` (reference, generous ROI + ladder + rescue),
`rigid_object` (tight ROI, no fallback), `high_recall` (deeper ladder, weak
targets). All `TemporalPropagationConfig` fields are overridable — see
`docs/tuning.md`.

## QA / 质量保障

- **LOO validation** reports IoU / Boundary-F1 on held-out anchors — no extra
  labeling. 留一法在留出锚帧上报告 IoU / Boundary-F1，无需额外标注。
- **`pred_source` histogram** in `sequence_summary.csv` shows how many frames
  needed rescue. `sequence_summary.csv` 中的 `pred_source` 直方图显示救援占比。
- **QA grid** (`qa_grid.png`) for a one-glance visual check. QA 拼图一眼排查漂移/泄漏/空帧。

## Outputs / 输出

```
<output>/
  run_manifest.json  loo_quality.csv  frame_manifest.csv  sequence_summary.csv
  <sequence_key>/
    masks/<stem>.png   jsons/<stem>.json   area.csv   qa_grid.png
```
See `docs/export_formats.md`.

## Phenotyping / 表型计算

The `phenocv.phenotypes` package turns a **mask** (and optional richer inputs)
into a flat, fail-closed trait table. It is **input-tier driven**: each
extractor declares what inputs it needs; only extractors whose requirements are
satisfied for the current frame run. This lets a single uniform call serve a
seedling RGB frame, a full D435 RGB-D frame, and an MS400 multispectral frame.

### Input layers (Tiers) / 输入分层

| Tier | Inputs | Extractor | What it computes |
|------|--------|-----------|------------------|
| 1 | `mask` | `shape2d` | area, bbox, centroid, convex-hull area, perimeter, solidity, circularity, aspect ratio (mask-only, cv2+numpy). |
| 2 | `mask`, `rgb` | `rgb_vegetation_indices` | 16 normalized-RGB indices (ExG/ExR/ExGR/GLI/GRVI/NGRDI/VEG/CIVE/…) + 10 experience features, each ×{mean,median,std,p10,p90}. |
| 3 | `mask`, `depth`, `calibration` | `canopy_3d_geometry` | plant height = distance above fitted soil plane (mean / p95), projected area, visible leaf-surface area, envelope volume (mm). Needs intrinsics + fixed-ground plane. |
| 4 | `mask`, `multispectral` | `multispectral_vegetation_indices` | MS400 4-band (555/660/720/840nm) → 12 indices (NDVI/NDRE/GNDVI/SAVI/OSAVI/RVI/DVI/CIgreen/CIrededge/MTCI/MCARI/TCARI) + per-band reflectance stats; optional pot-rim false-positive filter. |

- **Fail-closed**: when an input is missing or a value is unobservable, the
  extractor emits `NaN` plus a `missing_reason` (or an `<name>_error` column),
  never a fabricated number. 缺失或不可观测时，输出 `NaN` + `missing_reason`
  （或 `<name>_error` 列），绝不编造数值。
- Sign-preserving safe division: VARI/WI legitimately have negative denominators
  — use `_safe_divide` / `signed_safe_divide`, never raw `/`.

### Python API / 编程接口

```python
import numpy as np
from phenocv.phenotypes import compute_traits, compute_index_images
from phenocv.phenotypes import compute_plant_height, empirical_line_gains, apply_gains
from phenocv.phenotypes import remove_pot_rim_false_positive
from phenocv.phenotypes import CameraIntrinsics, load_rgb_intrinsics

# One uniform call. Only satisfied tiers run; unmet tiers emit NaN + missing_reason.
row = compute_traits(
    mask=mask_bool,                          # 2D bool/uint8, 0/255
    rgb=rgb_uint8,                           # optional [H,W,3] uint8
    depth=depth_mm,                          # optional [H,W] float mm
    calibration=intrinsics_or_preset_path,   # optional CameraIntrinsics | path
    multispectral={555: g, 660: r, 720: re, 840: nir},  # optional {nm: [H,W] float}
)
# row: dict keyed by trait name, plus "_inputs" and "_extractors_run".

# Per-layer helpers (work without the orchestrator):
index_imgs = compute_index_images(bands)          # 12 MS index images
gains = empirical_line_gains(panel_medians)       # empirical-line calibration
calib = apply_gains(signal_bands, gains)
h = compute_plant_height(mask, depth, intrinsics, soil_plane=None)  # auto-fits soil plane
```

### CLI / 命令行

```bash
# List every registered extractor and its input contract:
phenocv list-traits -v

# Compute all traits available for one mask (RGB + depth optional):
phenocv phenotype \
  --mask  results/run/seq1/masks/frame_0003.png \
  --rgb   /local/mirror/seq1/rgb/frame_0003.png \
  --depth /local/mirror/seq1/depth/frame_0003_mm.png \
  --calibration configs/intrinsics_second.yaml \
  --out results/run/seq1/traits/frame_0003.json \
  --csv results/run/seq1/traits/frame_0003.csv
```

- `phenotype` writes **JSON** (one row) and, with `--csv`, a CSV; omits any
  input not supplied (that tier is simply not run). 未提供的输入对应的层级直接不运行。
- `list-traits -v` is the authoritative registry dump for agents — read it to
  learn which extractors exist and what each requires. 这是给智能体的权威注册表清单。

### Extending the engine (add a trait) / 扩展引擎

Write one `TraitExtractor` subclass and decorate it with `@register`. The
orchestrator picks it up automatically — **no engine change needed**.

```python
from phenocv.phenotypes import TraitExtractor, register, INPUT_MASK, INPUT_RGB

@register
class MyTraitExtractor(TraitExtractor):
    name = "my_trait"                  # unique key in the registry
    requires = [INPUT_MASK, INPUT_RGB] # subset of available inputs
    tier = 2                           # 1..4 (informational)

    def extract(self, *, mask=None, rgb=None, **ctx):
        if mask is None or rgb is None:
            return {}                  # fail-closed: not applicable
        try:
            return {"my_value": float(compute(rgb, mask))}
        except Exception as e:          # never abort the whole row
            return {"my_trait_error": str(e)}
```

- Available inputs are detected from the kwargs passed to `compute_traits`;
  `available_for(available_inputs)` selects every extractor whose `requires`
  is a subset, sorted by `tier`. `compute_traits` wraps each `extract()` in
  try/except so one throwing extractor records `<name>_error` instead of
  aborting the row. 可用输入由 `compute_traits` 的实参推断；单个提取器抛错只
  记录 `<name>_error`，不影响其他层级。
- `requires` constants: `INPUT_MASK`, `INPUT_RGB`, `INPUT_DEPTH`,
  `INPUT_CALIB`, `INPUT_MULTISPECTRAL`.

### Calibration notes / 标定要点

- **L3 needs intrinsics + a fixed ground reference.** Pass a `CameraIntrinsics`
  or a preset YAML; the soil plane is fitted by IRLS + deterministic RANSAC over
  depth points outside the mask. 冠层高度依赖内参 + 固定地面参考；土壤平面由
  IRLS + 确定性 RANSAC 拟合（用掩膜外深度点）。
- **L4 reflectance is calibrated via empirical-line gains** from a reference
  panel (`PANEL_REFLECTANCE` for serial `CA320233044`). The pot-rim
  false-positive filter removes unsupported arcs using RGB + NDVI dual evidence.
  L4 反射率走经验线增益校准；盆沿假阳性用 RGB + NDVI 双证据过滤。

## References / 参考

- README.md / README.zh-CN.md
- docs/tuning.md, docs/export_formats.md, docs/adapter_guide.md
- Segmentation module: `src/phenocv/segmentation/` — `engine.py` (CPU logic
  layer is importable & testable without CUDA; `torch`/`sam2` imported lazily
  only inside `Sam2VideoPropagator`), `config.py`, `adapters/`.
- Phenotype module: `src/phenocv/phenotypes/` — `base.py` (re-exports the
  shared `phenocv.core.registry` + `TraitExtractor`), `shape2d.py` (L1),
  `rgb_indices.py` (L2), `geometry3d.py` + `calib.py` (L3), `multispectral.py`
  (L4), `compute_traits.py` (orchestrator). Pure-CPU (numpy + cv2), no torch.
  Tests: `tests/test_phenotypes.py`.
- Shared core: `src/phenocv/core/` — `registry.py` (`TraitExtractor`,
  `@register`, `available_for`) and `io.py` (mask/RGB/depth/multispectral
  readers). Add a sibling module under `src/phenocv/<name>/` and register its
  tools with `phenocv.core.registry.register` to extend the toolkit.
