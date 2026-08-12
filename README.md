# PhenoCV

> A composable, multi-modal **plant-phenotyping** toolkit — modules share one
> `phenocv.core`; [SAM 2](https://github.com/facebookresearch/sam2) is one
> *optional* segmentation backend, not the project's identity. Three modules
> ship today: `segmentation`, `phenotypes`, and `thermal`.

> 可组合、多模态的**植物表型**工具箱 —— 各模块共享同一个 `phenocv.core`；
> [SAM 2](https://github.com/facebookresearch/sam2) 只是一种*可选的*分割后端，并非
> 项目本体。当前已发布三个模块：`segmentation`、`phenotypes`、`thermal`。

[![PyPI version](https://img.shields.io/pypi/v/phenocv.svg)](https://pypi.org/project/phenocv/)
[![Python versions](https://img.shields.io/pypi/pyversions/phenocv.svg)](https://pypi.org/project/phenocv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml/badge.svg)](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub-blue.svg)](./docs/)

**English**

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

**中文**

PhenoCV 不是单个算法，而是一套**可组合、可插拔**的植物表型视觉模块。当前已提供三个，地位平等：

- **`phenocv.segmentation`**：用**少量人工标注的关键帧**，得到**完整时序的分割掩膜**。
  默认后端采用 [SAM 2](https://github.com/facebookresearch/sam2) 视频传播，但引擎**与数据源无关**，
  任何数据集格式都可通过轻量**适配器（adapter）**接入。SAM 2 只是多种后端之一
  （Classical 与 YOLO 后端被设计为可即插即换的替代项 —— 见[架构](#-架构)）。
- **`phenocv.phenotypes`**：**四级、失败即留痕的表型引擎**，从一张植株掩膜
  （外加可选 RGB / 深度 / 多光谱）算出一张扁平的表型表：2D 形状 → RGB 植被指数 →
  3D 高度/体积 → 多光谱指数。它只运行你**实际提供输入**的那些提取器。
- **`phenocv.thermal`**：**纯 CPU 的热红外（FLIR）表型模块**：逐像素温度表型、
  按相对高度划分的上/中/下层冠层温度、冠层相对环境的 ΔT、环境传感器时序对齐，
  以及前后对照的胁迫/复水分析（移动块 bootstrap + HAC 不确定性）。核心仅依赖
  `numpy` + `cv2` + `pandas` + `scipy` + `statsmodels`（后两者懒加载），无需 GPU；
  可选的 SAM 2 时序分割层懒加载。

三个模块都构建在共享的 **`phenocv.core`**（表型提取器注册表 + 模态/适配器注册表 + IO 工具）
之上；因此新增模块（例如 `phenocv.counting`）只需「写一个包、注册你的工具」——核心无需改动。

> **🖼️ About the figures.** Every image in this README is **desensitized**: the
> first two are rendered from a fully synthetic demo (no real field data); the
> thermal figures are method illustrations with calibration boards and labels
> removed. They illustrate the *method*, not any specific experiment or genotype.

> **🖼️ 关于图中的图像。** 本 README 中的每一张图都经过**脱敏**：前两张来自
> 完全合成的样本（无真实田间数据）；热红外图为方法示意，已移除标定板与标签。
> 它们只用于说明*方法*，不代表任何具体实验或基因型。

---

## 🌟 Why PhenoCV

**English**

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

**中文**

- **可组合，非单体**：分割模块、表型引擎、热红外模块，可单独或任意组合使用，各自独立、可单独导入。
- **失败即留痕、可审计**：每个表型行都记录 `pred_source` / `_inputs` / `_extractors_run`，
  不可观测的输出为 `NaN` + `missing_reason`——绝不伪造。
- **无需额外标注的 QA**：留一法（LOO）在你真实的锚帧上报告 IoU / Boundary-F1，无需任何额外标注；
  且所有输出都带溯源标签。
- **幼苗友好**：ROI 裁剪把早期幼苗的有效分辨率提升约 10×（否则传播后端会把每帧缩放到 1024px，把幼苗压没）。
- **CPU 可测**：整个逻辑层（ROI 运算、阈值阶梯、救援、IoU/BF1、表型引擎、热红外栈）无需 CUDA ——
  CI 与贡献者都不需要 GPU。
- **分割后端无关**：SAM 2 是默认而非必须；`BaseSegmenter` 契约允许 Classical 与 YOLO 后端即插即换。

  ![ROI cropping lifts a small seedling's effective resolution ~10× (synthetic demo)](docs/assets/fig2_roi_crop_benefit.png)

---

## ✨ Modules

**English**

| Module | What you get |
|---|---|
| `phenocv.segmentation` | Temporal canopy segmentation with pluggable backends (SAM 2 default; Classical / YOLO designed as drop-ins). Bidirectional (fwd+rev) logit averaging, threshold-ladder fallback + point-rescue, LOO IoU/BF1 QA, pluggable adapters, ISAT/CSV/QA export |
| `phenocv.phenotypes` | 4-tier trait engine: 2D shape (area/bbox/solidity…), RGB vegetation indices (ExG/ExR/VARI…), 3D height/volume (mm, needs depth+intrinsics), multispectral indices (12 + reflectance stats) |
| `phenocv.thermal` | Pure-CPU FLIR phenotyping: temperature traits (`temp_*` keys), canopy-layer (upper/middle/lower) temperatures by relative height, canopy ΔT vs ambient, environment-sensor alignment (no extrapolation, gap-guarded), before/after stress analysis (block-bootstrap CI + HAC + light/dark control); optional SAM 2 segmentation layer |
| `phenocv.core` | Shared trait-extractor **registry** (`@register`) + a modality/adapter **registry** + minimal IO helpers (mask/RGB/depth/multispectral/thermal readers) used by every module |

**中文**

| 模块 | 能力 |
|---|---|
| `phenocv.segmentation` | 时序冠层分割，可插拔后端（默认 SAM 2；Classical / YOLO 设计为即插即换）。双向（正向+反向）logits 平均，阈值阶梯回退 + 点救援，LOO IoU/BF1 QA，可插拔适配器，ISAT/CSV/QA 导出 |
| `phenocv.phenotypes` | 四级表型引擎：2D 形状（面积/ bbox/ 实心度…）、RGB 植被指数（ExG/ExR/VARI…）、3D 高度/体积（mm，需深度+内参）、多光谱指数（12 个 + 反射率统计） |
| `phenocv.thermal` | 纯 CPU 热红外表型：温度表型（`temp_*` 列名）、按相对高度划分的上/中/下层冠层温度、冠层相对环境 ΔT、环境传感器对齐（禁止外推、缺口防护）、前后对照胁迫分析（移动块 bootstrap CI + HAC + 光暗对照）；可选的 SAM 2 分割层 |
| `phenocv.core` | 所有模块共享的表型提取器**注册表**（`@register`）+ 模态/适配器**注册表** + 极简 IO 工具（掩膜 / RGB / 深度 / 多光谱 / 热红外读取器） |

![The 4-tier phenotype engine: from a single mask (+ optional RGB / depth / multispectral) to a flat trait table](docs/assets/fig3_four_tiers.png)

See [Roadmap](#-roadmap) for what is planned next.

---

## 💿 Installation

**English**

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

**中文**

```bash
# 核心（仅 CPU，引擎/适配器/测试均不需要 torch）
pip install phenocv

# 源码安装 + 开发工具（pytest）
pip install -e ".[dev]"

# 可选：GPU 视频传播（SAM 2 后端）
pip install "phenocv[video]"
```

> **说明：** `video` 额外依赖会拉入 `torch` + `sam2`，用于 SAM 2 *分割后端*。实际跑分割需要
> SAM 2 权重（如 `sam2.1_hiera_l.pt`）及其模型配置（如 `sam2.1_hiera_l.yaml`，随 `sam2` 包提供）。
> 其余部分 —— 适配器、配置、表型引擎、热红外栈、CPU 单元测试 —— 都不需要它。其它分割后端
> （Classical / YOLO）正以同一 `BaseSegmenter` 契约加入，无需 SAM 2。

---

## 🚀 Quickstart

**English**

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

**中文**

PhenoCV 的三个模块相互独立 —— 按需使用其中任何一个。

### 1. 生成合成示例（无真实数据，纯 CPU）

```bash
python tools/make_demo_sample.py --out samples/demo
```

生成 `samples/demo/{frames,masks,manifest.csv}` —— 一个随时间增大的绿色圆盘，共 6 帧、3 个稀疏锚帧。

### 2. 分割（多个模块入口之一）

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

结果落在 `results/demo/`（见 [输出与 QA](#-输出与-qa)）。

### 3. 从掩膜计算表型

```bash
phenocv phenotype \
  --mask results/demo/plant_01/<stem>.png \
  --rgb  /local/mirror/rgb/<stem>.png \
  --depth /local/mirror/depth_mm/<stem>.png \
  --calibration configs/intrinsics_second.yaml \
  --out results/demo/traits/plant_01_<stem>.json
```

### 4. 编程 API

```python
# 分割
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
print(result["loo_summary_interior"])  # IoU / BF1 中位数
```

```python
# 表型
import cv2
import numpy as np
from phenocv.phenotypes import compute_traits

mask = (cv2.imread("mask.png", 0) > 0)
rgb = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)
row = compute_traits(mask=mask, rgb=rgb)   # 只运行你传入输入所满足的提取器
print(row)
```

### 5. 纯 CPU 冒烟测试（无 GPU、无 SAM 2）

```bash
pip install -e ".[dev]"
pytest                      # 30 个测试，全部 CPU
python -c "import phenocv; print(phenocv.list_modules())"
```

---

## 🧩 Adapter Contract

**English**

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

**中文**

引擎消费 `PlantSequence` 对象。默认的 `CsvManifestAdapter` 读取**单个 manifest**，
无需任何数据集代码：

| 列名 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `sequence_key` | str | ✅ | 序列 id，如 `plant_01` |
| `frame_idx` | int | ⚪ | 0 基时序索引（缺省按行序） |
| `frame_path` | str | ✅ | RGB 帧路径 |
| `frame_label` | str | ⚪ | 可读标签（日期 / DAS） |
| `is_anchor` | 0/1 | ✅ | 该帧是否带人工掩膜 |
| `mask_path` | str | ⚪ | 锚帧掩膜 PNG（当 `is_anchor=1` 时必填） |
| *（任意其他列）* | — | ⚪ | 原样透传为 `frame_extras` |

也支持 JSON manifest（行字典列表，或 `{"frames": [...]}`）。
自定义适配器见 [docs/adapter_guide.md](./docs/adapter_guide.md)。同一套适配器思路由
`phenocv.core` 的**模态/适配器注册表**统一抽象，任何模块都能声明如何接入新数据源。

---

## 🔧 Presets

**English**

Presets live under the `presets:` block of `configs/default.yaml` and are
applied with `--preset <name>` (or `load_config(path, preset=...)`).

| Preset | Use case | Key knobs |
|---|---|---|
| `plant_phenotyping` | potted soybean temporal (reference) | generous ROI pad (1.9), threshold ladder on, rescue on |
| `rigid_object` | sharp-boundary, stable-scale objects | tight ROI (1.3), no ladder, no rescue, category `object` |
| `high_recall` | weak / easily-lost targets | larger ROI (2.2), deeper ladder (to −8.0), smaller `isat_min_area` |

Every `TemporalPropagationConfig` field is overridable — see
[docs/tuning.md](./docs/tuning.md).

**中文**

预设位于 `configs/default.yaml` 的 `presets:` 块，用 `--preset <名称>`（或
`load_config(path, preset=...)`）应用。

| 预设 | 适用场景 | 关键参数 |
|---|---|---|
| `plant_phenotyping` | 盆栽大豆时序（参考配置） | ROI 余量 1.9，开启阈值阶梯，开启救援 |
| `rigid_object` | 边界清晰、尺度稳定的物体 | ROI 收紧 1.3，无阶梯，无救援，类别 `object` |
| `high_recall` | 弱目标 / 易丢失目标 | ROI 放大 2.2，更深阶梯（至 −8.0），更小 `isat_min_area` |

每个 `TemporalPropagationConfig` 字段都可覆盖 —— 见 [docs/tuning.md](./docs/tuning.md)。

---

## 🏗️ Architecture

**English**

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

**中文**

```
phenocv/
├── __init__.py / __main__.py   # 工具箱入口；phenocv.list_modules()
├── cli.py                      # `phenocv segment|phenotype|list-traits`
├── core/                       # 所有模块共享的【基础】
│   ├── registry.py             # TraitExtractor 注册表 + @register + available_for，
│   │                           #   以及模态/适配器注册表（新数据源 = 新适配器）
│   └── io.py                   # 掩膜/RGB/深度/多光谱/热红外 读取器
├── segmentation/               # 模块 —— 时序冠层分割
│   ├── base.py                 # BaseSegmenter：可插拔后端（SAM2 / Classical / YOLO）
│   ├── engine.py               # ROI、传播、阈值阶梯、救援、
│   │                           #   LOO、ISAT 导出（与数据源无关）
│   ├── config.py               # YAML + 预设加载
│   └── adapters/               # BaseAdapter + CsvManifest + 盆栽示例
├── phenotypes/                 # 模块 —— 四级表型引擎
│   ├── base.py                 # 对 phenocv.core.registry 的再导出
│   ├── compute_traits.py       # 按层级编排、失败即留痕
│   ├── shape2d.py / rgb_indices.py / geometry3d.py / multispectral.py
│   └── calib.py                # 相机内参
└── thermal/                    # 模块 —— 纯 CPU 热红外表型（完整同级）
    ├── config.py io.py traits.py environment.py stress.py   # 核心（导入时无需 torch）
    └── segmentation.py         # 可选的 SAM 2 层，torch/sam2 懒加载
```

**扩展点。** PhenoCV 被设计为可无损扩展：

- **`phenocv.core.registry`** —— `TraitExtractor` 注册表（`@register` + `available_for`）。
  任何模块在此注册工具，即可被 `phenocv.list_modules()` / CLI 自动发现。
- **模态/适配器注册表**（位于 `core/registry.py`）—— 声明模块如何接入新数据源；
  CSV/JSON 适配器是其中一种实现。
- **`segmentation/base.py BaseSegmenter`** —— 后端无关的契约。SAM 2 是已发布后端；
  **Classical** 与 **YOLO** 后端被设计为同一接口下的即插即换替代项，切换后端无需改动引擎或适配器。

分割引擎从不直接读取你的文件系统布局 —— 它只看到 `PlantSequence` 对象。
表型引擎从不硬编码某个表型 —— 它只运行已注册的工具。正是这种分离让 PhenoCV
可复用、可组合。

---

## 📊 Outputs & QA

**English**

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

**中文**

`phenocv segment` 在 `--output` 下写入：

```
<output>/
├── run_manifest.json        # 完整运行记录 + LOO 汇总
├── loo_quality.csv          # 每个锚帧的 LOO：iou、bf1、pred_source
├── frame_manifest.csv       # 每帧：pred_source、area、thr、extras
├── sequence_summary.csv     # 每序列汇总（area 首/末/最大、计数）
└── <sequence_key>/
    ├── masks/<stem>.png     # 全图二值掩膜（0/255）
    ├── jsons/<stem>.json    # ISAT 标注（未设 --no-isat 时）
    ├── area.csv             # 每帧面积 + pred_source + extras
    └── qa_grid.png          # 帧×掩膜概览拼图（未设 --no-qa 时）
```

`pred_source` 取值与含义：

| `pred_source` | 含义 |
|---|---|
| `manual` | 人工标注锚帧，原样透传 |
| `propagated` | SAM 2 在基础阈值（0.0）下传播 |
| `propagated_lowthr` | 基础阈值为空，经阈值阶梯找回 |
| `point_rescue` | 阶梯后仍为空，用带框约束的点救援 |
| `failed_empty` | 所有兜底后仍无掩膜 |

![无需额外标注的 QA：溯源（`pred_source`）分布与留一法（LOO）IoU](docs/assets/fig5_qa_provenance.png)

---

## 🌡️ Thermal (FLIR) phenotyping

**English**

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

**中文**

`phenocv.thermal` 是一个**纯 CPU** 的热红外（红外）表型模块：它把真实的
温度矩阵（`°C`）+ 冠层掩膜，算成一张扁平的温度表型表，并可把环境传感器对齐到
帧时刻、分析前后对照的胁迫/复水响应。其设计契约与表型引擎一致 —— **失败即留痕**
（缺失/不可观测/空 → `NaN` + `missing_reason`，绝不编造）且**数据无关**（路径与列
映射由调用方提供）。核心（`io` / `traits` / `environment` / `stress`）**无需 GPU、无需
torch** 即可导入运行；只有可选的 `segmentation` 子层才懒加载 `torch`/`sam2`。

> **纯 Python 接口：** 热红外模块尚未接入 `phenocv` CLI —— 请用 Python 调用。

![热红外场景：植株冠层上的真实温度矩阵。](docs/assets/fig_thermal_scene.png)

![冠层掩膜上的温度叠加（仅 cv2 渲染，固定色阶）。](docs/assets/fig_thermal_overlay.png)

![冠层分层：按相对高度划分的上/中/下层。](docs/assets/fig_thermal_layers.png)

![环境传感器对齐到帧时刻（禁止外推、缺口防护）。](docs/assets/fig_thermal_envalign.png)

![复水事件前后的胁迫响应（移动块 bootstrap CI + HAC）。](docs/assets/fig_thermal_stress.png)

```python
import numpy as np
import phenocv.thermal as thermal

# io：读取真实温度矩阵 + 生成供 SAM2 提示的 3 通道特征图
temperature = np.load("stem_temp.npy").astype("float32")     # 真实 °C 矩阵
feat = thermal.thermal_feature_image(temperature)            # 绝对/局部ΔT/梯度
mask = thermal.polygons_to_mask((H, W), polygons)

# traits：一次调用只运行满足输入的提取器
row = thermal.compute_thermal_traits(mask=mask, temperature=temperature, ambient=23.0)
# -> canopy_temp_median_c、canopy_upper_median_c、canopy_delta_t_c 等
# 缺失 ambient → canopy_delta_t_c = NaN + missing_reason（绝不编造）

# environment：把传感器对齐到帧时刻（禁止外推、缺口防护）
env = thermal.read_environment_workbook(
    "environment.xlsx",
    column_map={"DateTime": "timestamp", "AirTemp": "ambient_c", "CO2": "co2_ppm"})
aligned = thermal.align_environment_to_frames(
    frame_timestamps, env, ["ambient_c", "co2_ppm"],
    max_gap_sec=600.0, timezone="UTC")        # 越界 → NaN + qc_flag

# stress：围绕事件的前/后对照，带移动块 bootstrap CI + 协变量校正 HAC
# 回归与暗期内部阴性对照
result = thermal.analyze_stress_response(
    timeseries_df, event_time, metric="canopy_temp_c",
    phase_column="phase", lit_value="light", random_seed=42)
```

SAM 2 时序热红外分割（`ThermalVideoSegmenter` / `segment_video_with_sam2`）需要
`pip install "phenocv[video]"` 加 SAM 2 权重；`thermal_feature_image` 作为 3 通道输入
喂给 SAM 2（而非伪彩色帧），清理以目标锚定为锚，确保吞并的盆体绝不被发布（fail-closed）。

---

## 🗺️ Roadmap

**English**

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

**中文**

PhenoCV 是一个围绕**植物表型研究热点**持续生长的模块箱。下列每一项都是计划 / 欢迎贡献的
`phenocv/` 同级模块，并把自己的工具注册到 `phenocv.core.registry`。已有的路线图构想已归入相应分组。

### 多模态融合
把 RGB + 热红外 + 多光谱 + 高光谱 + LiDAR/深度 融合为统一的表型视图。已发布的三个模块共享同一
IO/注册表核心，使这一目标可行。

### 胁迫与病害（生物/非生物）
干旱、高温等非生物胁迫评分；病斑/病征分割与病害评分（`phenocv.disease`，原独立路线图项），
复用与热红外一致的 fail-closed、少标注契约。

### 生长与发育
基于表型引擎长表的生育期分级与生长曲线拟合（`phenocv.growth`，原独立路线图项）。

### 器官计数
基于掩膜的花 / 果 / 分蘖 / 穗计数（`phenocv.counting`，原独立路线图项）—— 复用分割与表型注册表。

### 根系 / 根际
地下成像与根系性状；是把注册表驱动的表型引擎扩展到新模态的自然延伸。

### 采后品质
果/谷物品质性状（颜色、缺陷、纹理），通过适配器以新数据源复用表型分级机制。

### 产量与 G2P（基因型到表型）
面向育种的汇总：把高通量表型关联基因型、试验设计与选择指数。

### 基础模型
提示式分割 / 表型模型（SAM 家族及其后续），位于 `BaseSegmenter` 契约之下 —— SAM 2 是首个此类后端，
更多后端即插即换。

### 边缘 / 轻量部署
CPU/ONNX/TFLite 路径，使模块可在边缘相机与田间笔记本上无 GPU 运行（热红外模块已率先实现）。

### 不确定性估计 / 可解释性
逐表型置信度（bootstrap / HAC，热红外已实现）、显著性与溯源工具，跨模块复用。

![群体曲线：全部 44 株的「仅掩膜」冠层面积 / 纵向范围（真实、脱敏裁剪）——
一次统一调用即可从单帧幼苗扩展到整批群体](docs/assets/fig4_population_curves.png)

---

## 📚 Documentation

**English**

- [docs/tuning.md](./docs/tuning.md) — every segmentation knob, and *why* the defaults are what they are
- [docs/export_formats.md](./docs/export_formats.md) — mask / ISAT / CSV / QA layout
- [docs/adapter_guide.md](./docs/adapter_guide.md) — write your own data adapter
- [SKILL.md](./SKILL.md) — agent skill (WorkBuddy / Claude Code / Codex)
- [skills/phenocv-phenotype-port/](./skills/phenocv-phenotype-port/) — WorkBuddy skill for porting a phenotype pipeline into PhenoCV

**中文**

- [docs/tuning.md](./docs/tuning.md) —— 每个分割旋钮，以及*为什么*默认值是这样
- [docs/export_formats.md](./docs/export_formats.md) —— 掩膜 / ISAT / CSV / QA 布局
- [docs/adapter_guide.md](./docs/adapter_guide.md) —— 编写你自己的数据适配器
- [SKILL.md](./SKILL.md) —— 智能体技能（WorkBuddy / Claude Code / Codex）
- [skills/phenocv-phenotype-port/](./skills/phenocv-phenotype-port/) —— 把一个表型流水线移植进 PhenoCV 的 WorkBuddy 技能

---

## 🤝 Contributing

**English**

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [docs/extending.md](./docs/extending.md).
CPU-only setup, run `pytest`, open a PR.

**中文**

见 [CONTRIBUTING.md](./CONTRIBUTING.md) 与 [docs/extending.md](./docs/extending.md)。
CPU 环境即可开发，跑 `pytest`，开 PR。

---

## 📜 Citation

**English**

```bibtex
@software{phenocv2026,
  title  = {PhenoCV: Open-source computer-vision toolkit for plant phenotyping},
  author = {perseus-wy},
  year   = {2026},
  url    = {https://github.com/perseus-wy/PhenoCV},
  license = {MIT}
}
```

**中文**

```bibtex
@software{phenocv2026,
  title  = {PhenoCV: 开源植物表型计算机视觉工具箱},
  author = {perseus-wy},
  year   = {2026},
  url    = {https://github.com/perseus-wy/PhenoCV},
  license = {MIT}
}
```

---

## 📄 License

**English**

[MIT](./LICENSE) © 2026 perseus-wy.

**中文**

[MIT](./LICENSE) © 2026 perseus-wy.
