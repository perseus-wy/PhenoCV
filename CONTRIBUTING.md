# Contributing to PhenoCV / 为 PhenoCV 做贡献

Thanks for your interest in improving PhenoCV! This document explains the
**module contract** for multi-contributor development, how to set up an
environment, the testing rules, and how to submit changes.

感谢你为 PhenoCV 做出贡献！本文说明**多贡献者开发的模块契约**、开发环境搭建、
测试规则，以及如何提交改动。

PhenoCV is a *composable* toolkit: each capability lives in its own
`phenocv.<name>` package and plugs into one shared core (`phenocv.core`). You
rarely need to touch the core to add a feature — you add a package, register
your tools, and they become visible to the trait engine, the CLI and
`phenocv.list_modules()` automatically.

PhenoCV 是一个**可组合**的工具箱：每种能力各自位于 `phenocv.<name>` 包中，并接入
同一个共享核心（`phenocv.core`）。新增功能通常无需改动核心——你只需新增一个包、注册
你的工具，trait 引擎、CLI 与 `phenocv.list_modules()` 便会自动发现它们。

---

## Module contract — adding a `phenocv.xxx` package / 模块契约：新增一个包

**English.** A module is a sibling package under `src/phenocv/` that:

* has an `__init__.py` that **exports its public API** (the functions / classes
  users should reach for) and performs **no heavy imports at import time**
  (keep `torch` / `sam2` / `matplotlib` lazy — import them inside the functions
  that need them, so the package is importable on a CPU-only host);
* **registers** its compute tools with `phenocv.core.registry` via `@register`
  so the shared trait engine can discover and run them;
* is **data-agnostic**: it never reads a specific dataset layout; paths and
  column maps are supplied by the caller (an adapter / the CLI).

**中文.** 一个模块是 `src/phenocv/` 下的同级包，应满足：

* 其 `__init__.py` **导出公共 API**（用户直接使用的函数/类），且**导入时不加载
  重型依赖**（`torch` / `sam2` / `matplotlib` 保持惰性——仅在所需函数内部导入），
  以便在纯 CPU 主机上也能导入；
* 通过 `@register` 将计算工具注册到 `phenocv.core.registry`，供共享 trait 引擎
  发现与运行；
* **数据无关**：绝不读写特定的数据集布局；路径与列映射由调用方（适配器/CLI）提供。

Minimal package layout / 最小包结构:

```
src/phenocv/counting/
    __init__.py        # export public API; register extractors
    extractors.py      # TraitExtractor subclasses + @register
    io.py              # readers/writers (cv2/numpy/pillow only)
```

`src/phenocv/counting/__init__.py` should look like / 其 `__init__.py` 形如:

```python
from __future__ import annotations
from .extractors import CountingExtractor          # a registered trait extractor
from .io import load_count_mask                    # public IO helper

__all__ = ["CountingExtractor", "load_count_mask"]
```

---

## Extension points / 扩展点

PhenoCV grows through three well-defined extension points.

PhenoCV 通过三个定义清晰的扩展点成长。

### 1. Trait extractor (core registry) / 表型提取器（核心注册表）

**English.** Subclass `TraitExtractor` from `phenocv.core.registry`, set the
class attributes `name`, `description`, `requires` (the input tags you need:
`INPUT_MASK`, `INPUT_RGB`, `INPUT_DEPTH`, `INPUT_CALIB`, `INPUT_MULTISPECTRAL`,
or module-local tags such as `thermal` / `ambient`), and `tier` (1=mask-only …
4=multispectral), implement `extract` (return a flat column dict; use
`float('nan')` for unobservable outputs — never fabricate), then decorate with
`@register`. The orchestrator (`compute_traits` / `compute_thermal_traits`) runs
every registered extractor whose `requires` ⊆ the inputs you actually pass.

**中文.** 继承 `phenocv.core.registry` 的 `TraitExtractor`，设置类属性 `name`、
`description`、`requires`（所需输入标签：`INPUT_MASK`、`INPUT_RGB`、`INPUT_DEPTH`、
`INPUT_CALIB`、`INPUT_MULTISPECTRAL`，或模块内标签如 `thermal` / `ambient`）与
`tier`（1=仅掩膜 … 4=多光谱），实现 `extract`（返回扁平列字典；不可观测输出用
`float('nan')`，绝不编造），再用 `@register` 装饰。编排器（`compute_traits` /
`compute_thermal_traits`）会运行所有 `requires` ⊆ 你所提供输入的已注册提取器。

```python
from phenocv.core.registry import TraitExtractor, register, INPUT_MASK, INPUT_THERMAL

@register
class CanopyTemperatureExtractor(TraitExtractor):
    name = "canopy_temperature"
    description = "Whole-canopy robust temperature statistics."
    requires = [INPUT_MASK, INPUT_THERMAL]
    tier = 1

    def extract(self, *, mask=None, temperature=None, **ctx):
        # ... compute ... return {"canopy_temp_median_c": ...}
        return {}
```

### 2. Segmentation algorithm backend / 分割算法后端

**English.** *How* the canopy masks are computed is a swappable **algorithm
backend** behind `BaseSegmenter` (in `phenocv.segmentation.base`). The production
backend is `SAM2Segmenter` (it delegates to `run_sam2_video_temporal`);
placeholders `ClassicalSegmenter` and `YOLOSegmenter` document the interface and
raise `NotImplementedError`. Add a new algorithm by subclassing `BaseSegmenter`,
implementing `run`, and registering it in `build_segmenter` — you never touch the
adapters, exporters, or CLI. Drive any backend through the dispatcher
`run_segmentation(sequences, output, checkpoint, backend=...)`, which also reads
a `backend:` key from the YAML config. This is the *design-redundancy* extension
point: the orchestration is algorithm-agnostic.

**中文.** 冠层掩膜的*计算方式*是一个可替换的**算法后端**，抽象基类为
`phenocv.segmentation.base` 中的 `BaseSegmenter`。生产后端是 `SAM2Segmenter`（委托给
`run_sam2_video_temporal`）；占位后端 `ClassicalSegmenter`、`YOLOSegmenter` 记录了接口
约定并抛出 `NotImplementedError`。新增算法只需继承 `BaseSegmenter`、实现 `run` 并在
`build_segmenter` 中注册——无需改动适配器、导出器或 CLI。任意后端都经由分发器
`run_segmentation(sequences, output, checkpoint, backend=...)` 调用（该分发器也会读取
YAML 配置中的 `backend:` 字段）。这是*设计冗余*扩展点：编排逻辑与具体算法解耦。

```python
# src/phenocv/segmentation/backends/my_backend.py
from phenocv.segmentation.base import BaseSegmenter
from phenocv.segmentation.engine import DEFAULT_SAM2_CONFIG

class ClassicalSegmenter(BaseSegmenter):
    backend_name = "classical"

    def run(self, sequences, output_root, checkpoint,
            model_cfg=DEFAULT_SAM2_CONFIG, device="cpu", **kw):
        # implement a pure-CPU OpenCV pipeline here; same I/O contract as SAM2
        raise NotImplementedError("implement thresholding + watershed / GrabCut")
```

```python
from phenocv.segmentation.base import run_segmentation
result = run_segmentation(seqs, out, ckpt, backend="classical")
```

### 3. Data adapter / 数据适配器

**English.** *My dataset layout* is a separate extension point: a **data
adapter** that translates your on-disk layout into engine-ready `PlantSequence`
objects. Subclass `BaseAdapter` from `phenocv.segmentation.adapters.base` and
implement `build_sequences`, which returns a list of `PlantSequence` (each with
time-ordered `frame_paths` and sparse `AnchorFrame` masks in **full-image
boolean** coordinates). Pass the result to `run_segmentation(...)`. This is the
extension point for "how do I feed my own data layout into segmentation" — do
**not** patch the core engine. (This is distinct from the algorithm backend in
§2: the adapter decides *what frames/masks to feed*; the backend decides *how
they are propagated*.)

**中文.** *我的数据布局*是另一个独立的扩展点：通过**数据适配器**把你的磁盘布局转换为引擎
可用的 `PlantSequence` 对象。继承 `phenocv.segmentation.adapters.base` 的
`BaseAdapter` 并实现 `build_sequences`，返回 `PlantSequence` 列表（每个含按时间排序的
`frame_paths` 与稀疏的 `AnchorFrame` 掩膜，掩膜为**全图像布尔**坐标）。将结果传入
`run_segmentation(...)`。这是"如何把我的数据布局接入分割"的扩展点——**不要**改动核心引擎。
（这与 §2 的算法后端不同：适配器决定*喂哪些帧/掩膜*，后端决定*如何传播*。）

```python
from phenocv.segmentation.adapters import BaseAdapter
from phenocv.segmentation.engine import PlantSequence, AnchorFrame
import cv2, numpy as np, os, glob

class MyAdapter(BaseAdapter):
    def __init__(self, root, min_anchors=2):
        self.root, self.min_anchors = root, min_anchors

    def build_sequences(self, **kwargs):
        sequences = []
        for plant_dir in sorted(os.listdir(self.root)):
            frames = sorted(glob.glob(os.path.join(self.root, plant_dir, "rgb", "*.png")))
            anchors = []
            for i, fp in enumerate(frames):
                mp = fp.replace("rgb", "mask_manual")
                if os.path.exists(mp):
                    anchors.append(AnchorFrame(frame_idx=i, mask=cv2.imread(mp, 0) > 127))
            if len(anchors) < self.min_anchors:
                continue   # skip, but report skips (a progress callback), never drop silently
            sequences.append(PlantSequence(key=plant_dir, frame_paths=frames, anchors=anchors))
        return sequences
```

### 4. Modality reader / 模态读取器

**English.** A new sensing modality (e.g. multispectral, LiDAR, hyperspectral,
thermal, fluorescence) plugs into a **centralised registry** in
`phenocv.core.modalities`. Subclass `ModalityReader`, set `name` / `description`,
implement `read(path) -> numpy array`, and decorate with `@register_modality`.
Look readers up with `get_modality(name)`, `all_modalities()`, or
`available_modalities()`. The four reference modalities (`mask`, `rgb`, `depth`,
`multispectral`) and an example `lidar` reader are pre-registered; when the
modality needs a new trait, also register a `TraitExtractor` declaring its input
tag. Keep modality code under its own `phenocv.<modality>` package.

**中文.** 接入新传感模态（如多光谱、LiDAR、高光谱、热红外、荧光）使用
`phenocv.core.modalities` 中的**集中式注册表**。继承 `ModalityReader`，设置
`name` / `description`，实现 `read(path) -> numpy array`，并用 `@register_modality`
装饰。通过 `get_modality(name)`、`all_modalities()` 或 `available_modalities()` 查找。
四种参考模态（`mask`、`rgb`、`depth`、`multispectral`）与一个示例 `lidar` 读取器已预注册；
若该模态需要新表型，请同时注册一个声明其输入标签的 `TraitExtractor`。模态代码请放在
独立的 `phenocv.<modality>` 包下。

```python
from phenocv.core.modalities import ModalityReader, register_modality

@register_modality
class HyperspectralReader(ModalityReader):
    name = "hyperspectral"
    description = "Load an ENVI/numpy spectral cube."
    def read(self, path):
        import numpy as np
        return np.load(path)  # [H, W, bands]

# get_modality("hyperspectral").read("cube.npy")
```

### Extension-points list / 扩展点清单

| Extension point | Where / 位置 | How to add / 添加方式 |
| --- | --- | --- |
| Trait extractor | `phenocv.core.registry` | subclass `TraitExtractor` + `@register` |
| Segmentation algorithm backend | `phenocv.segmentation.base.BaseSegmenter` | subclass + implement `run`, register in `build_segmenter` |
| Data adapter | `phenocv.segmentation.adapters.base.BaseAdapter` | subclass + implement `build_sequences` |
| Modality reader | `phenocv.core.modalities.ModalityReader` | subclass + `@register_modality` |
| Config | `configs/*.yaml` + a `load_*_config` helper | flat dict, `from_mapping` |

---

## Testing rules / 测试规则

**English.**
* **CPU-only.** Tests must run without a GPU or downloaded weights; the GPU
  propagation layer (`torch` / `sam2`) is imported lazily and may be mocked.
* Use **pytest**. Keep it fast; prefer **synthetic, deterministic data**
  (`numpy.random.default_rng(seed)`) over fixtures that need real captures.
* **Keep `torch` / `sam2` lazy** — never import them at module top level, so the
  CPU logic stays importable and testable.
* Never write outputs to the `C:` drive in tests or tooling; use a temp dir.
* Verify numeric examples import cleanly (no `matplotlib` at import time).

**中文.**
* **仅 CPU。** 测试必须能在无 GPU、无下载权重的条件下运行；GPU 传播层
  （`torch` / `sam2`）惰性导入，可被打桩。
* 使用 **pytest**。保持快速；优先使用**合成、确定性数据**
  （`numpy.random.default_rng(seed)`），而非需要真实采集的夹具。
* **保持 `torch` / `sam2` 惰性**——绝不在模块顶层导入，以便 CPU 逻辑可导入、可测试。
* 测试或工具中切勿向 `C:` 盘写入；请使用临时目录。
* 校验数值示例能干净导入（导入时不触发 `matplotlib`）。

```bash
pytest
```

The suite is CPU-only and mocks the GPU layer, so it runs anywhere.

测试套件仅依赖 CPU 并对 GPU 层打桩，因而可在任意环境运行。

---

## "Under construction" convention / "建设中"约定

**English.** Partially-implemented modules are allowed and welcome. Mark them
clearly so users and the CLI know what is production-ready:

* add a module-level docstring / `.. note::` stating *what is implemented vs TODO*;
* raise `NotImplementedError` (or return `NaN` + `missing_reason`) for stubs
  rather than pretending they work;
* keep such modules discoverable (importable) so they appear in
  `phenocv.list_modules()`, but document their status.

**中文.** 我们欢迎"部分实现"的模块。请明确标注，让用户与 CLI 知道哪些是生产可用的：

* 在模块 docstring / `.. note::` 中说明*已实现 vs 待办*；
* 对占位实现抛 `NotImplementedError`（或返回 `NaN` + `missing_reason`），不要假装
  可用；
* 保持模块可被导入（可被 `phenocv.list_modules()` 发现），并注明其状态。

---

## Coding style / 代码风格

* Format with **black** and lint with **ruff** (config in `pyproject.toml`).
* **Bilingual docstrings** (English + 中文) for public functions/classes.
* Prefer **data-agnostic APIs**: new dataset formats = new **adapters**, not
  edits to the core engine.
* **Never write to `C:`** in source, tests, or tools — use temp dirs / the repo.
* Keep dependencies optional where possible (heavy deps lazy-imported).

**中文。** 用 **black** 格式化、**ruff** 检查（`pyproject.toml` 中配置）。公共函数/类使用
**双语 docstring**（英文 + 中文）。优先**数据无关 API**：新数据集格式 = 新**适配器**，
而非改动核心引擎。**源码、测试、工具中切勿写入 `C:`**——使用临时目录或仓库内路径。重型
依赖尽量可选（惰性导入）。

---

## Development setup / 开发环境搭建

PhenoCV keeps heavy GPU dependencies optional so that CPU-only contributors
and CI stay fast.

PhenoCV 将重型 GPU 依赖设为可选，使纯 CPU 贡献者与 CI 保持轻快。

```bash
# Clone / 克隆
git clone https://github.com/perseus-wy/PhenoCV.git
cd PhenoCV

# Create an environment (any tool you like) / 创建虚拟环境（任意工具）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install in editable mode with dev + optional video deps
# 以可编辑模式安装，含 dev 与可选 video 依赖
pip install -e ".[dev,video]"
```

If you only want to work on the CPU logic (adapters, ROI math, QA metrics,
thermal traits), `pip install -e ".[dev]"` is enough — `torch`/`sam2` are only
needed for the actual GPU video propagation layer, which is lazily imported.

若只做 CPU 逻辑（适配器、ROI 数学、QA 指标、热表型），`pip install -e ".[dev]"`
即可——`torch`/`sam2` 仅在实际 GPU 视频传播层需要，该层惰性导入。

---

## Running tests / 运行测试

```bash
pytest
```

The test suite is CPU-only and mocks the GPU propagation layer, so it runs
without a GPU or downloaded weights.

测试套件仅依赖 CPU 并对 GPU 传播层打桩，无需 GPU 或下载权重即可运行。

---

## Pull requests / 提交 PR

1. Fork the repo and create a topic branch (`feat/...`, `fix/...`).
   复刻仓库并创建主题分支（`feat/...`、`fix/...`）。
2. Keep changes focused; add/extend tests for new behavior.
   改动保持聚焦；为新行为补充/扩展测试。
3. Run `pytest` locally before pushing. 推送前本地运行 `pytest`。
4. Open a PR against `main` with a clear description of the change and its
   motivation. 向 `main` 提交 PR，清楚说明改动及其动机。

For new modules, also add a short bilingual note in your module docstring and a
docs page under `docs/` if it ships user-facing APIs.

新增模块时，请在模块 docstring 中补充简短双语说明；若含面向用户的 API，请在 `docs/`
下增加文档页。

---

## Reporting issues / 报告问题

Open a GitHub issue with a minimal reproduction (synthetic sample preferred)
and your environment (`phenocv` version, Python, OS, GPU/driver if relevant).

请通过 GitHub issue 提交最小复现（优先合成样例）及你的环境（phenocv 版本、Python、
操作系统、相关 GPU/驱动）。

---

## Code of Conduct / 行为准则

By participating, you agree to uphold our
[Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to
the maintainers via GitHub issues.

参与即表示你同意遵守我们的[行为准则](CODE_OF_CONDUCT.md)。如有不当行为，
请通过 GitHub issue 向维护者举报。
