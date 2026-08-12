---
name: phenocv
description: >-
  PhenoCV — a composable, open-source computer-vision toolkit for plant
  phenotyping. Modules share one `phenocv.core` (registry + IO); segmentation is
  one of several modules, and SAM 2 is one of several pluggable segmentation
  backends (Classical / YOLO are designed as drop-ins). Shipped modules:
  (1) `phenocv.segmentation` — temporal canopy segmentation with pluggable
  backends (SAM 2 by default); (2) `phenocv.phenotypes` — a pluggable,
  fail-closed **phenotype computation** engine (2D shape, RGB-vegetation,
  3D-canopy-height, multispectral traits) from a mask (+ optional RGB /
  depth+calibration / multispectral inputs); (3) `phenocv.thermal` — a pure-CPU
  **thermal (FLIR) phenotyping** module (per-pixel temperature traits,
  canopy-layer partitioning, environment-sensor alignment, before/after
  stress/rewatering analysis with block-bootstrap & HAC uncertainty). Build new
  modules by writing a package and registering tools with `phenocv.core.registry`.
  Use when a user wants to (a) segment a plant/object time-series from a few
  labeled frames, (b) build a phenotyping mask dataset, (c) write a data
  adapter for a new dataset format, (d) tune ROI / threshold / rescue
  parameters, (e) compute phenotypic traits (canopy area, vegetation indices,
  plant height, multispectral indices) from existing masks, (f) compute thermal
  (FLIR) traits from a temperature matrix + mask, align environment sensors onto
  frame timestamps, or analyse before/after stress/rewatering responses, or
  (g) extend PhenoCV with a new module/tool.
  PhenoCV —— 可组合的开源植物表型计算机视觉工具箱。各模块共享 `phenocv.core`（注册表+IO）；
  分割只是多个模块之一，SAM 2 只是多个可插拔分割后端之一（Classical / YOLO 设计为即插即换）。
  已发布模块：① `phenocv.segmentation` —— 时序冠层分割，可插拔后端（默认 SAM 2）；②
  `phenocv.phenotypes` —— 可插拔、fail-closed 的**表型计算**引擎（2D 形状、RGB 植被指数、
  3D 株高、多光谱指数）；③ `phenocv.thermal` —— 纯 CPU 的**热红外（FLIR）表型**模块：
  逐像素温度表型、按相对高度的分层、环境传感器时序对齐、前后对照的胁迫/复水分析
  （移动块 bootstrap + HAC 不确定性）。新增模块只需写包并把工具注册到 `phenocv.core.registry`。
  当用户需要（a）用少量标注帧分割植物/物体时序，（b）构建表型掩膜数据集，（c）为新数据集格式
  编写适配器，（d）调整 ROI / 阈值 / 救援参数，（e）基于已有掩膜计算表型（冠层面积、植被指数、
  株高、多光谱指数），（f）基于温度矩阵+掩膜计算热红外表型、把环境传感器对齐到帧时刻、或分析
  前后对照的胁迫/复水响应，或（g）为 PhenoCV 扩展新模块/工具时使用。
---

# PhenoCV Skill

> Composable open-source computer-vision toolkit for plant phenotyping. Three
> modules ship today — `phenocv.segmentation` (temporal canopy segmentation; SAM 2
> is the default *one* of several pluggable backends), `phenocv.phenotypes` (a
> 4-tier, fail-closed trait engine), and `phenocv.thermal` (a pure-CPU thermal /
> FLIR phenotyping module) — all on a shared `phenocv.core`.
> 可组合的开源植物表型计算机视觉工具箱：当前含 `phenocv.segmentation`（时序冠层分割；
> SAM 2 只是多个可插拔后端中的默认项）、`phenocv.phenotypes`（四级 fail-closed 表型引擎）
> 与 `phenocv.thermal`（纯 CPU 热红外表型模块），共享 `phenocv.core`。

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
- User has a **thermal (FLIR) temperature matrix + canopy mask** and wants
  per-pixel temperature traits, upper/middle/lower canopy-layer temperatures, or
  canopy ΔT vs ambient. 用户有**热红外温度矩阵 + 冠层掩膜**，想要逐像素温度表型、
  上/中/下层冠层温度或冠层相对环境的 ΔT。
- User wants to **align environment sensors** (air temp / CO₂ / VPD / soil
  moisture) onto FLIR frame timestamps, or **analyse before/after stress or
  rewatering** (block-bootstrap CI + HAC + light/dark control). 用户想**把环境
  传感器**对齐到热红外帧时刻，或**分析前后对照的胁迫/复水响应**（移动块 bootstrap +
  HAC + 光暗对照）。

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

`pip install -e ".[dev]"` installs the **full CPU thermal stack** (pandas +
scipy + statsmodels + openpyxl). The thermal core (io / traits / environment /
stress) is importable and testable with **no GPU and no torch**. The optional
`phenocv.thermal.segmentation` (SAM 2 temporal segmentation) needs
`pip install "phenocv[video]"` plus a SAM 2 checkpoint; `torch`/`sam2` are
imported lazily only inside that sub-layer, so importing `phenocv.thermal`
itself never pulls in CUDA.
`pip install -e ".[dev]"` 即包含**完整的 CPU 热红外栈**（pandas + scipy +
statsmodels + openpyxl）。热红外核心（io / traits / environment / stress）无需
GPU、无需 torch 即可导入与测试。可选的 `phenocv.thermal.segmentation`（SAM 2 时序
分割）需要 `pip install "phenocv[video]"` 加 SAM 2 权重；`torch`/`sam2` 仅在该子层
内懒加载，因此导入 `phenocv.thermal` 本身不会引入 CUDA。

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

## Thermal (FLIR) phenotyping / 热红外（FLIR）表型

`phenocv.thermal` is a **pure-CPU thermal (FLIR) phenotyping module** ported
from a private pipeline and shared into PhenoCV as a clean, data-agnostic,
fail-closed toolkit. It adds a third input tier to the shared `phenocv.core`
registry (`thermal` + `ambient` temperature alongside the existing `mask`) and
ships four cooperating submodules plus one optional GPU sub-layer:

- **`io`** — temperature / meta / mask readers, the 3-channel `thermal_feature_image`
  (absolute / local-ΔT / gradient), cv2-only overlays, and layer-overlap
  resolution. 温度/元数据/掩膜读取、3 通道热特征图、cv2 叠加、分层重叠消解。
- **`traits`** — a registry-driven, fail-closed trait engine (canopy temperature,
  upper/middle/lower layer temperature by relative height, canopy ΔT). 受注册表
  驱动的 fail-closed 表型引擎（整株温度、上/中/下层温度、冠层 ΔT）。
- **`environment`** — align an environment time-series onto frame timestamps by
  linear interpolation (no extrapolation; gap-guarded). 环境时序按帧时刻线性
  插值对齐（禁止外推、缺口防护）。
- **`stress`** — before/after stress / rewatering analysis with block-bootstrap &
  HAC uncertainty, plus generic population roll-ups. 前后对照胁迫/复水分析（移动块
  bootstrap + HAC 不确定性）与通用人群级汇总。
- **`segmentation`** *(optional, GPU)* — SAM 2 temporal thermal segmentation;
  `torch`/`sam2` imported lazily, never at import time. 可选的 SAM 2 时序分割，
  torch/sam2 仅懒加载。

> **Python-API only:** `phenocv.thermal` is not yet wired into the `phenocv` CLI
> (no `thermal` subcommand exists in `src/phenocv/cli.py`). Drive it from Python
> as shown below. **纯 Python 接口：** `phenocv.thermal` 尚未接入 `phenocv` CLI
> （`cli.py` 中暂无 `thermal` 子命令），请按下列示例用 Python 调用。

### Design contract / 设计契约

- **Pure CPU** — only `numpy` + `cv2` + `pandas` + `scipy` + `statsmodels` (the
  last two lazy-imported). No `torch` / `sam2` / `matplotlib` at import time, so
  the core is importable and testable without a GPU or a plotting backend.
  **纯 CPU** —— 仅 numpy + cv2 + pandas + scipy + statsmodels（后两者懒加载）。
  导入时不引入 torch / sam2 / matplotlib，因此核心无需 GPU 或绘图后端即可导入与测试。
- **Fail-closed** — missing / unobservable / empty inputs return `NaN` +
  `missing_reason` (or `<name>_error`), never a fabricated value.
  **失败即留痕** —— 缺失/不可观测/空输入返回 `NaN` + `missing_reason`（或
  `<name>_error`），绝不编造数值。
- **Data-agnostic** — core logic never reads a specific dataset layout; paths and
  column maps are supplied by the caller. **数据无关** —— 核心逻辑不读取特定数据集
  布局，路径与列映射由调用方提供。
- All bootstrap / HAC randomness is seeded via `random_seed` → reproducible.
  所有 bootstrap / HAC 随机性由 `random_seed` 控制，结果可复现。
- Trait test key names are `temp_*` (e.g. `temp_median_c`). 表型测试列名为
  `temp_*`（如 `temp_median_c`）。

### Python API / 编程接口

**io — read, render, partition**

```python
import numpy as np
import phenocv.thermal as thermal

temperature = np.load("stem_temp.npy").astype("float32")   # true °C matrix
meta = thermal.load_thermal_meta("stem_temp.npy")           # sibling *_meta.json ({} if absent)

# 3-channel feature image (absolute / local-ΔT / gradient) for SAM2 prompts:
feat = thermal.thermal_feature_image(temperature)

# render on a fixed scale (cv2 only, headless-safe, no matplotlib):
overlay = thermal.make_overlay(temperature, mask, vmin=22.0, vmax=29.0)
layer_overlay = thermal.make_layer_overlay(
    temperature, layer_masks, vmin=22.0, vmax=29.0)        # upper/middle/lower colours

# resolve overlapping canopy layers by distance to each layer's identity seed:
resolved, overlap_count = thermal.resolve_layer_overlap(layer_masks, identity_seeds)

# polygon -> boolean mask (lossless, PIL-based), and round-trip IO:
mask = thermal.polygons_to_mask((H, W), polygons)
thermal.save_mask(mask, "mask.png"); mask = thermal.load_mask("mask.png")
```

**traits — fail-closed temperature traits**

```python
from phenocv.thermal import (
    summarize_masked_temperature, partition_canopy_by_relative_height,
    compute_thermal_traits, CanopyTemperatureExtractor,
    LayerTemperatureExtractor, CanopyDeltaTExtractor,
)

# one uniform call runs only the extractors whose inputs you pass:
row = compute_thermal_traits(mask=mask, temperature=temperature, ambient=23.0)
# -> canopy_temp_median_c, canopy_upper_median_c, canopy_delta_t_c, ...
# missing ambient -> canopy_delta_t_c = NaN + missing_reason (never fabricated)

# core helpers:
stats = summarize_masked_temperature(temperature, mask)            # temp_median_c, temp_p10_c, pixel_count, ...
layers = partition_canopy_by_relative_height(mask)                # {"upper","middle","lower"} by relative height

# extractor classes (each declares requires + tier, registered via @register):
CanopyTemperatureExtractor   # mask + thermal            -> whole-canopy temperature
LayerTemperatureExtractor    # mask + thermal            -> upper/middle/lower temperatures + range
CanopyDeltaTExtractor       # mask + thermal + ambient  -> canopy minus ambient ΔT
```

**environment — align sensors onto frame timestamps**

```python
from phenocv.thermal import (
    read_sensor_workbook, read_environment_workbook,
    align_environment_to_frames, EnvironmentJoiner,
)

# map raw columns -> semantic names; the reader never coerces `timestamp` to numeric:
env = read_environment_workbook(
    "environment.xlsx",
    column_map={"DateTime": "timestamp", "AirTemp": "ambient_c", "CO2": "co2_ppm"},
)

# linear-interpolate sensor series onto frame timestamps.
# No extrapolation: out-of-range frames -> NaN + qc_flag="outside_sensor_range".
# Gap guard: brackets wider than max_gap_sec -> NaN + qc_flag="sensor_gap_exceeds_limit".
aligned = align_environment_to_frames(
    frame_timestamps, env,
    value_columns=["ambient_c", "co2_ppm"],
    max_gap_sec=600.0, timezone="UTC",
)   # DataFrame: value_columns + qc_flag + bracket_gap_sec + nearest_offset_sec

# stateful alternative over many target times:
joiner = EnvironmentJoiner(env, ["ambient_c"], timezone="UTC", max_gap_sec=600.0)
result = joiner.align(frame_timestamps[0])   # InterpolationResult(values, qc_flag, ...)
```

**stress — before/after response with uncertainty**

```python
from phenocv.thermal import (
    analyze_stress_response, recovery_kinetics, summarize_plants,
    summarize_paired_differences, detect_light_transitions,
    calculate_layer_correlations, pair_shifted_times,
    vapour_pressure_deficit, moving_block_bootstrap_ci, hac_mean_ci,
)

# before/after contrast around an event (e.g. rewatering): block-bootstrap CI +
# covariate-adjusted HAC regression, with a dark-phase internal negative control
# (transpiration-driven cooling should vanish without light):
result = analyze_stress_response(
    timeseries_df, event_time, metric="canopy_temp_c",
    phase_column="phase", lit_value="light",
    covariate_columns=["vpd_kpa", "co2_ppm"],
    random_seed=42,
)
result["phase_contrast"]   # per-phase effect + block-bootstrap CI + Welch p
result["hac_adjusted"]     # metric ~ post (+ covariates), HAC-robust
result["kinetics"]         # optional post-event recovery bins

# generic, data-agnostic roll-ups (no private hard-coding):
plant_df, phase_df = summarize_plants(
    {"plant_1": df_1, "plant_2": df_2}, ["canopy_temp_c"], phase_column="phase")
paired = pair_shifted_times(df, df, shift_hours=24.0, metric_columns=["canopy_temp_c"])
summarize_paired_differences(paired, ["canopy_temp_c"])              # HAC + block-bootstrap roll-ups
detect_light_transitions(frames, "timestamp", "phase", ["canopy_temp_c"])
calculate_layer_correlations(frames, ["ambient_c"], ["canopy_temp_c"])  # Spearman + BH q
vpd = vapour_pressure_deficit(temp_c_series, rh_pct_series)          # Tetens VPD (kPa)

# reusable uncertainty estimators:
lo, hi = moving_block_bootstrap_ci(values, statistic="mean", random_seed=42)   # circular block bootstrap 95% CI
mean, ci_lo, ci_hi = hac_mean_ci(values, max_lags=12)                         # Newey–West HAC 95% CI
```

**segmentation (optional) — SAM 2 temporal thermal segmentation**

```python
from phenocv.thermal import (
    ThermalVideoSegmenter, segment_video_with_sam2,
    ThermalSegmentConfig, clean_target_mask, merge_bidirectional,
    load_prompt_config, thermal_feature_image,
)

# Pure-CPU logic layer (importable & testable without CUDA):
cfg = ThermalSegmentConfig()                       # 640x480 validation, cleanup + QC defaults
raw = np.zeros((H, W), bool); raw[20:80, 20:80] = True
cleaned, info = clean_target_mask(
    raw, reference_points=[(50.0, 50.0)], reference_labels=[1], is_reference=True)
# info["hard_fail"] flags a background-engulf / plant-merged-with-soil event
merged = merge_bidirectional(fwd_map, bwd_map, n, (H, W))   # forward priority, bidirectional merge

# GPU layer (torch/sam2 imported lazily inside this call):
summary = ThermalVideoSegmenter(
    checkpoint_path="/path/to/sam2.1_hiera_l.pt",
    model_size="small", device="cuda",
).run_segment(
    segment_id="segment_01", stems=stems,
    temperature_paths=temp_paths, reference_stem=ref_stem,
    prompt_cfg=load_prompt_config("prompt.yaml"),
    output_dir="results/thermal/segment_01",
)
# Every frame carries `pred_source` (manual / propagated / failed_empty), and
# `thermal_feature_image` (absolute/local-ΔT/gradient) is fed to SAM2 as the
# 3-channel input — never a pseudo-color frame. Cleanup is target-anchored:
# components without identity-support are dropped, so an engulfed pot is never
# published (fail-closed: a QC failure raises and writes segment_failed_qc.json).
```

## Extension points / 扩展点

PhenoCV is built to be extended without forking:

- **`phenocv.core.registry`** — the `TraitExtractor` registry (`@register` +
  `available_for`). Any module registers its tools here; they become visible to
  `phenocv.list_modules()` / the CLI automatically.
- **Modality / adapter registry** (in `core/registry.py`) — declare how a module
  ingests a new data source; the CSV/JSON adapter is one implementation.
- **`segmentation/base.py BaseSegmenter`** — a backend-agnostic contract. SAM 2
  is the shipped backend; **Classical** and **YOLO** backends are designed as
  drop-in alternatives behind the same interface, so swapping backends never
  touches the engine or the adapters.

PhenoCV 被设计为可无损扩展：`phenocv.core.registry` 是 `TraitExtractor` 注册表
（`@register` + `available_for`），任何模块在此注册即可被自动发现；模态/适配器注册表
声明如何接入新数据源；`segmentation/base.py BaseSegmenter` 是后端无关的契约，SAM 2 是已发布
后端，Classical 与 YOLO 后端可即插即换，切换后端无需改动引擎或适配器。

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
- Thermal module: `src/phenocv/thermal/` — `config.py` (`ThermalConfig` /
  `load_thermal_config`), `io.py` (temperature/mask IO + `thermal_feature_image`
  + cv2 overlays + `resolve_layer_overlap`), `traits.py`
  (`summarize_masked_temperature`, `partition_canopy_by_relative_height`,
  `CanopyTemperatureExtractor` / `LayerTemperatureExtractor` /
  `CanopyDeltaTExtractor`, `compute_thermal_traits` — pure-CPU, no torch),
  `environment.py` (`read_*_workbook`, `align_environment_to_frames`,
  `EnvironmentJoiner`, no-extrapolation + gap guard), `stress.py`
  (`analyze_stress_response`, `recovery_kinetics`, `summarize_plants`,
  `calculate_layer_correlations`, `pair_shifted_times`, `vapour_pressure_deficit`,
  block-bootstrap & HAC uncertainty — scipy/statsmodels lazy-imported), and
  `segmentation.py` (optional SAM 2 layer; `torch`/`sam2` lazy-imported only
  inside `segment_video_with_sam2` / `ThermalVideoSegmenter`). Thermal core is
  importable & testable without CUDA. Tests: `tests/test_thermal*.py`.
- Shared core: `src/phenocv/core/` — `registry.py` (`TraitExtractor`,
  `@register`, `available_for`) and `io.py` (mask/RGB/depth/multispectral
  readers). Add a sibling module under `src/phenocv/<name>/` and register its
  tools with `phenocv.core.registry.register` to extend the toolkit.
