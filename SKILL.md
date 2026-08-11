---
name: phenocv
description: >-
  PhenoCV — open-source vision toolkit for plant phenotyping; temporal canopy
  segmentation via SAM 2 video propagation from sparse manual keyframes.
  Use when a user wants to (a) segment a plant/object time-series from a few
  labeled frames, (b) build a phenotyping mask dataset, (c) write a data
  adapter for a new dataset format, or (d) tune ROI / threshold / rescue
  parameters.  PhenoCV —— 面向植物表型的开源视觉工具；基于 SAM 2 视频传播，
  从少量人工关键帧得到完整时序冠层分割。当用户需要（a）用少量标注帧分割植物/
  物体时序，（b）构建表型掩膜数据集，（c）为新数据集格式编写适配器，或
  （d）调整 ROI / 阈值 / 救援参数时使用。
---

# PhenoCV Skill

> Open-source vision toolkit for plant phenotyping. Temporal canopy segmentation
> via SAM 2 video propagation: a few manually labeled keyframes → a fully
> segmented time series. 面向植物表型的开源视觉工具，基于 SAM 2 视频传播：
> 少量人工关键帧 → 完整时序分割。

## When to use / 何时触发

- User has a **sequence of plant/object images over time** and wants a mask for
  every frame. 用户有一组**随时间变化的植物/物体图像序列**，想要每帧的掩膜。
- User wants **annotation-free accuracy** (LOO IoU / Boundary-F1) on their own
  anchors. 用户想要对其锚帧做**无需额外标注的精度评估**（LOO IoU / Boundary-F1）。
- User is building a **phenotyping dataset** (canopy area over time, growth
  curves). 用户正在构建**表型数据集**（时序冠层面积、生长曲线）。
- User needs to **plug in a new dataset format** (write an adapter). 用户需要
  **接入新的数据集格式**（编写适配器）。

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
from phenocv.adapters import CsvManifestAdapter
from phenocv.config import load_config
from phenocv.engine import run_sam2_video_temporal

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

## References / 参考

- README.md / README.zh-CN.md
- docs/tuning.md, docs/export_formats.md, docs/adapter_guide.md
- Engine source: `src/phenocv/engine.py` (CPU logic layer is importable & testable
  without CUDA; `torch`/`sam2` imported lazily only inside `Sam2VideoPropagator`).
