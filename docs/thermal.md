# Thermal (FLIR) phenotyping with `phenocv.thermal`

**English.** `phenocv.thermal` is a pure-CPU thermal-infrared module: it turns a
2D temperature matrix (°C) plus a canopy mask into robust temperature traits,
aligns an environment time-series onto FLIR frame timestamps, and runs
before/after stress / rewatering analysis with block-bootstrap and HAC
uncertainty. Every renderer uses **cv2 only** (no matplotlib), so the whole
module is importable and testable on a CPU-only host. This tutorial embeds
five real-data figures under `docs/assets/` and shows runnable snippets.

**中文。** `phenocv.thermal` 是一个纯 CPU 的热红外模块：它把二维温度矩阵（°C）与冠层掩膜转换为稳健的温度表型，把环境时序对齐到 FLIR 帧时刻，并做前后胁迫/复水分析（移动块 bootstrap 与 HAC 不确定性）。所有渲染器**仅用 cv2**（不依赖 matplotlib），因而可在纯 CPU 主机上导入与测试。本教程嵌入 `docs/assets/` 下的五张真实数据示意图，并给出可运行代码片段。

> Figures below are generated from real FLIR data included in the repo
> (`samples/demo/thermal/`).
> 下文展示的示意图均基于 repo 中包含的真实 FLIR 数据生成（`samples/demo/thermal/`）。

---

## 1. The thermal scene, the mask, and overlays — `io`

![Real FLIR thermal scene](docs/assets/fig_thermal_scene.png)

**English.** A thermal image is just a `float32 [H, W]` temperature matrix.
`load_temperature` reads it from a `.npy`; `thermal_feature_image` combines
absolute temperature / local ΔT / gradient into a 3-channel feature (a good
GrabCut prompt for mask refinement); `polygons_to_mask` rasterises a canopy
polygon; `make_overlay` renders temperature on a fixed scale with a mask
contour. Real FLIR data lives under `samples/demo/thermal/`.

**中文。** 一张热红外图像本质上就是一个 `float32 [H, W]` 温度矩阵。
`load_temperature` 从 `.npy` 读取；`thermal_feature_image` 把绝对温度 / 局部
温差 / 梯度组合成 3 通道特征（可作为掩膜精修的 GrabCut 提示）；
`polygons_to_mask` 将冠层多边形栅格化；`make_overlay` 在固定温标上渲染温度并
叠加掩膜轮廓。真实 FLIR 数据位于 `samples/demo/thermal/`。

```python
import numpy as np
import phenocv.thermal as T

# --- Load the real temperature matrix (float64 [H,W], °C) ---
# 加载真实温度矩阵（float64 [H,W]，单位 °C）
temp = T.load_temperature("samples/demo/thermal/temperature_0000.npy")

# --- Build a 3-channel thermal feature image (absolute / local ΔT / gradient) ---
# 构建 3 通道热特征图（绝对温度 / 局部温差 / 梯度）
feat = T.thermal_feature_image(temp)          # uint8 [H,W,3]

# --- Rasterise a canopy polygon into a boolean mask ---
# 把冠层多边形栅格化为布尔掩膜
theta = np.linspace(0, 2 * np.pi, 24)
poly = np.column_stack([160 + 70 * np.cos(theta), 120 + 60 * np.sin(theta)]).tolist()
mask = T.polygons_to_mask(temp.shape, [poly])   # bool [H,W]

# --- Render temperature on a fixed scale with a mask overlay (cv2 only) ---
# 固定温标渲染温度并叠加掩膜（仅 cv2）
overlay = T.make_overlay(temp, mask, vmin=22.0, vmax=34.0)   # BGR [H,W,3]
```

![Canopy mask overlay](docs/assets/fig_thermal_overlay.png)

*Figure: the canopy disc drawn as a green contour over the thermal scene
(rendered with `cv2.drawContours`). 图：用 cv2 在热场景上以绿色轮廓绘出冠层掩膜。*

---

## 2. Canopy temperature traits — `traits`

![Canopy layer partition](docs/assets/fig_thermal_layers.png)

**English.** `compute_thermal_traits` runs every registered extractor whose
inputs you provide: whole-canopy temperature statistics, upper/middle/lower
layer temperatures (via `partition_canopy_by_relative_height`), and canopy
ΔT = canopy temperature − ambient. Missing inputs return `NaN` +
`missing_reason` (fail-closed), never a fabricated value.

**中文.** `compute_thermal_traits` 会运行你提供的所有输入的已注册提取器：整株冠层
温度统计、上/中/下层温度（`partition_canopy_by_relative_height`）、以及冠层
ΔT = 冠层温度 − 环境温度。缺失输入时返回 `NaN` + `missing_reason`（fail-closed），
绝不编造数值。

```python
import numpy as np
import phenocv.thermal as T

H, W = 240, 320
rng = np.random.default_rng(0)
temp = rng.normal(28.0, 1.5, size=(H, W)).astype(np.float32)
theta = np.linspace(0, 2 * np.pi, 24)
poly = np.column_stack([160 + 70 * np.cos(theta), 120 + 60 * np.sin(theta)]).tolist()
mask = T.polygons_to_mask((H, W), [poly])

# --- One call computes every applicable trait for this (mask, temperature) ---
# 一次调用即算出该（掩膜, 温度）下所有可计算的表型
row = T.compute_thermal_traits(mask=mask, temperature=temp, ambient=24.5)
print(row["canopy_temp_median_c"], row["canopy_delta_t_c"])

# --- Or call the building blocks directly ---
# 也可以直接调用底层函数
summary = T.summarize_masked_temperature(temp, mask)        # dict of temp stats
layers = T.partition_canopy_by_relative_height(mask)       # {"upper","middle","lower"}
print(layers.keys())                                        # dict_keys(['upper','middle','lower'])
```

*The three colour bands (cyan / green / magenta) are the upper, middle and
lower canopy thirds, split by the vertical bbox of the whole-canopy mask — the
same split `partition_canopy_by_relative_height` performs.
三层颜色（青/绿/洋红）即上、中、下冠层，按整株掩膜垂直 bbox 三等分，与
`partition_canopy_by_relative_height` 的切分一致。*

---

## 3. Aligning an environment time-series to frames — `environment`

![Environment alignment](docs/assets/fig_thermal_envalign.png)

**English.** FLIR frames carry a capture timestamp but no ambient conditions.
`read_environment_workbook` reads a sensor workbook (with a column map to
semantic names); `align_environment_to_frames` linearly interpolates the
requested columns onto each frame timestamp. Two hard, fail-closed rules:
**no extrapolation** (frames outside the sensor range get `NaN` +
`qc_flag="outside_sensor_range"`) and a **gap guard** (`NaN` +
`qc_flag="sensor_gap_exceeds_limit"` when bracketing observations are too far
apart). Nothing is silently invented.

**中文.** FLIR 帧带有采集时刻但没有环境条件。`read_environment_workbook` 读取传感器
工作簿（通过列映射得到语义名）；`align_environment_to_frames` 把所需列按各帧时刻
线性插值对齐。两条 fail-closed 硬规则：**禁止外推**（超出传感器范围的帧返回
`NaN` + `qc_flag="outside_sensor_range"`）与**缺口保护**（前后观测间隔过大时返回
`NaN` + `qc_flag="sensor_gap_exceeds_limit"`）。绝不静默编造数据。

```python
import pandas as pd
import phenocv.thermal as T

# --- Load the demo environment workbook ---
# 加载示例环境工作簿
env = T.read_environment_workbook(
    "samples/demo/thermal/environment.csv",
    column_map={"ambient_temp_c": "ambient_c"},   # source column -> semantic name
)

# --- Align ambient temperature onto two frame timestamps ---
# 把环境温度对齐到两个帧时刻
aligned = T.align_environment_to_frames(
    frame_timestamps=["2026-01-01 08:05:00", "2026-01-01 08:25:00"],
    env_dataframe=env,
    value_columns=["ambient_c"],
    max_gap_sec=600,
)
print(aligned[["ambient_c", "qc_flag"]])
```

*Real-data figure: a diurnal ambient-temperature curve (blue) with red markers
at the FLIR frame timestamps that were aligned onto it.
真实数据图：日变化环境温度曲线（蓝）与对齐到其上的 FLIR 帧时刻红点标记。*

---

## 4. Before/after stress response — `stress`

![Stress response](docs/assets/fig_thermal_stress.png)

**English.** `analyze_stress_response` contrasts a metric (e.g. canopy ΔT) before
vs after an event (irrigation / rewatering). It reports, per photoperiod phase,
the effect with a **block-bootstrap 95% CI** and a Welch t-test, runs a
**HAC-adjusted** regression (`metric ~ post`, robust to autocorrelation) on the
lit phase, and treats the **dark phase as an internal negative control**
(transpiration-driven cooling should vanish without light). Optional recovery
kinetics bins summarise post-event dynamics.

**中文.** `analyze_stress_response` 围绕事件（灌溉/复水）对指标（如冠层 ΔT）做前后
对照。它按光暗阶段分别报告效应量与**移动块 bootstrap 95% 区间**及 Welch t 检验，
并在有光阶段运行**HAC 校正回归**（`metric ~ post`，对自相关稳健）；**暗期作为内部
阴性对照**（无光时蒸腾降温应消失）。可选的恢复动力学分箱汇总事件后动态。

```python
import numpy as np
import pandas as pd
import phenocv.thermal as T

n = 120
ts = pd.date_range("2026-01-01 00:00:00", periods=n, freq="10min")
event = pd.Timestamp("2026-01-01 10:00:00")
rng = np.random.default_rng(1)
metric = np.concatenate([
    rng.normal(5.0, 0.4, 60),   # pre-event: water deficit -> high ΔT
    rng.normal(1.5, 0.4, 60),   # post-event: rewatering -> low ΔT
])
light = (ts.hour >= 6) & (ts.hour < 20)
df = pd.DataFrame({
    "timestamp": ts,
    "canopy_dt_c": metric,
    "light_phase": np.where(light, "light", "dark"),
})

res = T.analyze_stress_response(
    df, event, "canopy_dt_c",
    phase_column="light_phase", lit_value="light",
    covariate_columns=None, random_seed=42,
)
print(res["phase_contrast"])     # per-phase effect + bootstrap CI + p
print(res["hac_adjusted"])       # regression post coefficient + HAC CI
```

*Real-data figure: paired bars of canopy ΔT before vs after an irrigation event,
with the negative delta annotated — recovered transpiration lowers ΔT.
真实数据图：灌溉事件前后冠层 ΔT 的配对柱状图，标注负向差值——复水后蒸腾恢复使 ΔT 降低。*

---

## 5. Where the data comes from — `samples/demo/thermal/`

**English.** The thermal sample under `samples/demo/thermal/` consists of 6 REAL
FLIR frames (`temperature_0000..0005.npy`, `float64` at 480×640) from a real
soybean canopy experiment spanning a rewatering event at 2026-07-28 16:46 (soil
moisture 18.5% → 59.3%). It also includes real SAM 2 canopy masks
(`masks/0000..0005.png`) and a real environment sensor log (`environment.csv`
with columns `timestamp, ambient_temp_c, ambient_rh_pct, co2_ppm,
soil_moisture_pct, light_lux`). The figures above are generated from these
committed data files — no generation script needed for the thermal sample.

**中文.** `samples/demo/thermal/` 下的热成像样本由 6 幅真实 FLIR 帧
（`temperature_0000..0005.npy`，`float64`, 尺寸 480×640）组成，来源于一次真实
大豆冠层试验，跨过一次发生于 2026-07-28 16:46 的复水事件（土壤水分 18.5% →
59.3%）。同时包含真实的 SAM 2 冠层掩膜（`masks/0000..0005.png`）与真实环境传感
器日志（`environment.csv`, 列包括 `timestamp, ambient_temp_c, ambient_rh_pct,
co2_ppm, soil_moisture_pct, light_lux`）。以上示意图均由这些已提交的真实数据生成，
热成像样本无需生成脚本。

```bash
# This script generates ONLY the synthetic RGB/segmentation demo
# (samples/demo/frames/, masks/, manifest.csv).
# 该脚本仅生成合成 RGB/分割演示数据（samples/demo/frames/、masks/、manifest.csv）。
python tools/make_demo_sample.py
# The thermal sample (samples/demo/thermal/) is committed real data — no
# generation needed.
# 热成像样本（samples/demo/thermal/）为已提交的真实数据——无需生成。
```

**Extension points.** To add a new thermal trait, subclass `TraitExtractor` from
`phenocv.core.registry` and decorate it with `@register` (see
[`docs/extending.md`](extending.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md)).
The trait engine, environment aligner and stress tools all stay data-agnostic:
paths and column maps are supplied by the caller, never hard-coded.

**扩展点。** 要新增热表型，只需继承 `phenocv.core.registry` 的 `TraitExtractor` 并
用 `@register` 装饰（见 [`docs/extending.md`](extending.md) 与
[`CONTRIBUTING.md`](../CONTRIBUTING.md)）。表型引擎、环境对齐器与胁迫工具均保持
数据无关：路径与列映射由调用方提供，绝不硬编码。
