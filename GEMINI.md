# GEMINI.md — PhenoCV

PhenoCV 是一个**可组合的开源植物表型计算机视觉工具箱** —— 由共享同一个
`phenocv.core` 的独立、可插拔模块构成。
PhenoCV is a **composable open-source computer-vision toolkit for plant
phenotyping** — a set of independent, pluggable modules that share one
`phenocv.core`.

当前已发布三个模块 (Three modules ship today):
- **`phenocv.segmentation`** —— 时序冠层分割（SAM 2 是默认的可选后端，Classical / YOLO 为可插拔替代）：
  用少量人工标注的关键帧，得到完整时序的分割掩膜。 / temporal canopy segmentation (SAM 2 is the
  default, optional, pluggable backend; Classical / YOLO are drop-in alternatives) from sparse
  manual keyframes.
- **`phenocv.phenotypes`** —— 可插拔、失败即留痕（fail-closed）的**四级表型引擎**
  （2D 形状 / RGB 植被指数 / 3D 株高 / 多光谱指数）：从一张掩膜（+ 可选 RGB / 深度+标定 /
  多光谱）算出一张扁平表型表。 / a pluggable, fail-closed 4-tier trait engine.
- **`phenocv.thermal`** —— **纯 CPU** 的热红外（FLIR）表型模块（io / traits /
  environment / stress + 可选的懒加载 SAM 2 分割层）：温度表型、按相对高度的分层、
  环境传感器时序对齐、前后对照的胁迫/复水分析（bootstrap + HAC 不确定性）。
  **热红外核心无需 GPU 即可导入与测试**（`numpy` + `cv2` + `pandas` + `scipy` +
  `statsmodels` 懒加载；`torch`/`sam2` 仅在可选分割层内懒加载）。 /
  pure-CPU thermal (FLIR) phenotyping module — importable & testable without a GPU.

**完整技能文档 / Full skill doc:** 见 [`SKILL.md`](./SKILL.md) —— 触发条件、概念、CLI/API、
适配器契约、预设、表型层级、QA 约定均在其中。本文件是轻量入口，请勿在此重复技能正文。
This file is a thin entry point — do not duplicate the skill body here.

## Quick commands / 快速命令

```bash
pip install -e ".[dev]"                       # CPU-only core + tests
pytest                                        # 30 CPU tests
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

## Contribution / test conventions / 贡献与测试约定

- **多模块布局 (Multi-module layout):** 每个能力在 `src/phenocv/<module>/` 下为同级包；
  共享基础是 `src/phenocv/core/`（注册表 + IO）。
- 引擎逻辑必须保持**无需 CUDA 即可导入**；`torch`/`sam2` 在 `Sam2VideoPropagator` 内懒加载。
- 核心引擎保持**数据源无关** —— 绝不在 `engine.py` 中直接读取数据集布局；新格式通过
  `adapters/` 接入。
- 每一帧必须带有 `pred_source` 溯源标签（manual / propagated / propagated_lowthr /
  point_rescue / failed_empty）。
- 表型保持 **CPU-only**（numpy + cv2，无 torch）且 **fail-closed** —— 缺失/不可观测 →
  `NaN` + `missing_reason`（或 `<name>_error`），绝不编造数值。新增表型只需写一个
  `TraitExtractor` 子类并加 `@register`，不要改编排器。
- 新增模块（如 `phenocv.counting`）：建包并把工具注册到 `phenocv.core.registry.register`，
  即可被 `phenocv.list_modules()` / CLI 自动发现。
- 开 PR 前运行 `pytest`（纯 CPU）。详见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## Repo map / 仓库结构

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
scripts/ tools/               # demo generator, batch phenotype compute
docs/{tuning,export_formats,adapter_guide}.md
skills/phenocv-phenotype-port/  # WorkBuddy skill: port a phenotype pipeline into PhenoCV
```

> 中文为主、英文为辅的说明贯穿全仓库；`SKILL.md` 与两份 `README` 均为中英双语。
> Chinese-first, English-secondary docs throughout; `SKILL.md` and both `README` files are bilingual.
