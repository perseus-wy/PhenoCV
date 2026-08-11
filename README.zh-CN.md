# PhenoCV

> 面向植物表型的**开源计算机视觉工具箱** —— 由可组合的独立模块构成（时序冠层分割、四级表型引擎，并可持续扩展），全部共享同一个 `core`。

[![PyPI version](https://img.shields.io/pypi/v/phenocv.svg)](https://pypi.org/project/phenocv/)
[![Python versions](https://img.shields.io/pypi/pyversions/phenocv.svg)](https://pypi.org/project/phenocv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml/badge.svg)](https://github.com/perseus-wy/PhenoCV/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub-blue.svg)](./docs/)

PhenoCV 不是单个算法，而是一套**可组合、可插拔**的植物表型视觉模块。当前已提供两个：

- **`phenocv.segmentation`**：用**少量人工标注的关键帧**，借助 [SAM 2](https://github.com/facebookresearch/sam2)
  视频传播，得到**完整时序的分割掩膜**。引擎**与数据源无关**，任何数据集格式都可通过轻量**适配器（adapter）**接入。
- **`phenocv.phenotypes`**：**四级、失败即留痕的表型引擎**，从一张植株掩膜（外加可选 RGB / 深度 / 多光谱）算出一张扁平的表型表：
  2D 形状 → RGB 植被指数 → 3D 高度/体积 → 多光谱指数。它只运行你**实际提供输入**的那些提取器。

两个模块都构建在共享的 **`phenocv.core`**（表型提取器注册表 + IO 工具）之上；
因此新增第三个模块（例如 `phenocv.counting`）只需「写一个包、注册你的工具」——核心无需改动。

---

## 🌟 为什么是 PhenoCV

- **可组合，非单体**：分割模块、表型引擎，各自独立、可单独导入使用。
- **时序一致性（分割）**：每隔几天标一帧即可，SAM 2 将标注传播到整条序列，掩膜平滑、无漂移。
- **幼苗友好**：ROI 裁剪把早期幼苗的有效分辨率提升约 10×（否则 SAM 2 会把每帧缩放到 1024px，把幼苗压没）。
- **失败即留痕、可审计（表型）**：每个表型行都记录 `pred_source` / `_inputs` / `_extractors_run`，不可观测的输出为 `NaN` + `missing_reason`——绝不伪造。
- **无需额外标注的 QA**：留一法（Leave-One-Out, LOO）在你真实的锚帧上报告 IoU / Boundary-F1，无需任何额外标注。
- **CPU 可测**：整个逻辑层（ROI 运算、阈值阶梯、救援、IoU/BF1、表型引擎）无需 CUDA —— CI 与贡献者都不需要 GPU。

## ✨ 模块

| 模块 | 能力 |
|---|---|
| `phenocv.segmentation` | SAM 2 视频传播，双向（正向+反向）logits 平均，阈值阶梯回退 + 点救援，LOO IoU/BF1 QA，可插拔适配器，ISAT/CSV/QA 导出 |
| `phenocv.phenotypes` | 四级表型引擎：2D 形状（面积/ bbox/ 实心度…）、RGB 植被指数（ExG/ExR/VARI…）、3D 高度/体积（mm，需深度+内参）、多光谱指数（12 个 + 反射率统计） |
| `phenocv.core` | 所有模块共享的表型提取器**注册表**（`@register`）+ 极简 IO 工具（掩膜 / RGB / 深度 / 多光谱读取器） |

---

## 💿 安装

```bash
# 核心（仅 CPU，引擎/适配器/测试均不需要 torch）
pip install phenocv

# 源码安装 + 开发工具（pytest）
pip install -e ".[dev]"

# 可选：GPU 视频传播（SAM 2）
pip install "phenocv[video]"
```

> **说明：** `video` 额外依赖会拉入 `torch` + `sam2`。实际跑分割需要 SAM 2 权重
> （如 `sam2.1_hiera_l.pt`）及其模型配置（如 `sam2.1_hiera_l.yaml`，随 `sam2` 包提供）。
> 其余部分 —— 适配器、配置、CPU 单元测试 —— 都不需要它。

## 🚀 快速开始

### 1. 生成合成示例（无真实数据，纯 CPU）

```bash
python tools/make_demo_sample.py --out samples/demo
```

生成 `samples/demo/{frames,masks,manifest.csv}` —— 一个随时间增大的绿色圆盘，共 6 帧、3 个稀疏锚帧。

### 2. 运行分割

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

### 3. 编程 API

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
print(result["loo_summary_interior"])  # IoU / BF1 中位数
```

```python
import cv2
from phenocv.phenotypes import compute_traits

mask = (cv2.imread("mask.png", 0) > 0)
rgb = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)
row = compute_traits(mask=mask, rgb=rgb)   # 只运行你传入输入所满足的提取器
print(row)
```

### 4. 纯 CPU 冒烟测试（无 GPU、无 SAM 2）

```bash
pip install -e ".[dev]"
pytest                      # 30 个测试，全部 CPU
python -c "import phenocv; print(phenocv.list_modules())"
```

---

## 🧩 适配器契约

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
自定义适配器见 [docs/adapter_guide.md](./docs/adapter_guide.md)。

## 🔧 预设

预设位于 `configs/default.yaml` 的 `presets:` 块，用 `--preset <名称>`（或
`load_config(path, preset=...)`）应用。

| 预设 | 适用场景 | 关键参数 |
|---|---|---|
| `plant_phenotyping` | 盆栽大豆时序（参考配置） | ROI 余量 1.9，开启阈值阶梯，开启救援 |
| `rigid_object` | 边界清晰、尺度稳定的物体 | ROI 收紧 1.3，无阶梯，无救援，类别 `object` |
| `high_recall` | 弱目标 / 易丢失目标 | ROI 放大 2.2，更深阶梯（至 −8.0），更小 `isat_min_area` |

每个 `TemporalPropagationConfig` 字段都可覆盖 —— 见 [docs/tuning.md](./docs/tuning.md)。

## 🏗️ 架构

```
phenocv/
├── __init__.py / __main__.py   # 工具箱入口；phenocv.list_modules()
├── cli.py                      # `phenocv segment|phenotype|list-traits`
├── core/                       # 所有模块共享的【基础】
│   ├── registry.py             # TraitExtractor + @register + available_for
│   └── io.py                   # 掩膜/RGB/深度/多光谱 读取器
├── segmentation/               # 模块 1 —— 时序冠层分割
│   ├── engine.py               # ROI、传播、阈值阶梯、救援、
│   │                           #   LOO、ISAT 导出（与数据源无关）
│   ├── config.py               # YAML + 预设加载
│   └── adapters/               # BaseAdapter + CsvManifest + 盆栽示例
└── phenotypes/                 # 模块 2 —— 四级表型引擎
    ├── base.py                 # 对 phenocv.core.registry 的再导出
    ├── compute_traits.py       # 按层级编排、失败即留痕
    ├── shape2d.py / rgb_indices.py / geometry3d.py / multispectral.py
    └── calib.py                # 相机内参
```

分割引擎从不直接读取你的文件系统布局 —— 它只看到 `PlantSequence` 对象。
表型引擎从不硬编码某个表型 —— 它只运行已注册的工具。正是这种分离让 PhenoCV
可复用、可组合。

## 📊 输出与 QA

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

## 🗺️ 路线图

PhenoCV 是一个持续生长的模块箱。计划 / 欢迎贡献的模块（每个都是 `phenocv/` 下的同级包，并把自己的工具注册到 `phenocv.core.registry`）：

- **`phenocv.counting`** —— 基于掩膜的花 / 果 / 分蘖计数。
- **`phenocv.disease`** —— 病斑 / 病征分割与评分。
- **`phenocv.growth`** —— 基于表型引擎长表的生育期分级与生长曲线拟合。

## 📚 文档

- [docs/tuning.md](./docs/tuning.md) —— 每个分割旋钮，以及*为什么*默认值是这样
- [docs/export_formats.md](./docs/export_formats.md) —— 掩膜 / ISAT / CSV / QA 布局
- [docs/adapter_guide.md](./docs/adapter_guide.md) —— 编写你自己的数据适配器
- [SKILL.md](./SKILL.md) —— 智能体技能（WorkBuddy / Claude Code / Codex）
- [skills/phenocv-phenotype-port/](./skills/phenocv-phenotype-port/) —— 把一个表型流水线移植进 PhenoCV 的 WorkBuddy 技能

## 🤝 贡献

见 [CONTRIBUTING.md](./CONTRIBUTING.md)。CPU 环境即可开发，跑 `pytest`，开 PR。

## 📜 引用

```bibtex
@software{phenocv2026,
  title  = {PhenoCV: Open-source vision toolkit for plant phenotyping},
  author = {perseus-wy},
  year   = {2026},
  url    = {https://github.com/perseus-wy/PhenoCV},
  license = {MIT}
}
```

## 📄 许可证

[MIT](./LICENSE) © 2026 perseus-wy.
