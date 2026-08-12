# How to contribute a module to PhenoCV / 如何为 PhenoCV 贡献一个模块

**English.** This guide is the hands-on companion to
[`CONTRIBUTING.md`](../CONTRIBUTING.md). It walks through adding a brand-new
`phenocv.<name>` package with a minimal but complete example: a registered trait
extractor, a registered data adapter (the §3 data-extension point), and a test.
Everything
below is runnable and synthetic — no real data, no GPU required.

**中文.** 本指南是 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 的实操补充。它用一个最小但完整的
示例，带你新增一个 `phenocv.<name>` 包：一个已注册的表型提取器、一个已注册的数据适配器
（§3 数据扩展点）以及对应的测试。以下全部可直接运行且为合成数据——无需真实数据，也无需 GPU。

---

## 1. Package layout / 包结构

```
src/phenocv/growth/
    __init__.py      # export public API; importing the submodules registers extractors
    extractors.py    # TraitExtractor subclasses decorated with @register
    io.py            # readers/writers (cv2 / numpy / pillow only)
    adapter.py       # optional data adapter (BaseAdapter)
tests/
    test_growth.py   # CPU-only, synthetic, deterministic
```

`src/phenocv/growth/__init__.py`:

```python
from __future__ import annotations

from .extractors import PlantHeightRateExtractor
from .io import load_height_series

__all__ = ["PlantHeightRateExtractor", "load_height_series"]
```

---

## 2. Minimal registered trait extractor / 最小已注册表型提取器

**English.** Subclass `TraitExtractor` from `phenocv.core.registry`, declare your
input dependencies and tier, implement `extract`, and decorate with `@register`.
The orchestrator runs your extractor automatically whenever its `requires` are
satisfied. Return `float('nan')` for anything unobservable — never fabricate.

**中文.** 继承 `phenocv.core.registry` 的 `TraitExtractor`，声明输入依赖与 tier，实现
`extract`，并用 `@register` 装饰。只要 `requires` 被满足，编排器便会自动运行你的提取器。
不可观测的结果返回 `float('nan')`——切勿编造。

```python
# src/phenocv/growth/extractors.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from phenocv.core.registry import INPUT_MASK, TraitExtractor, register


@register
class PlantHeightRateExtractor(TraitExtractor):
    name = "plant_height_rate"
    description = "Mean daily height growth rate from a height series."
    requires = [INPUT_MASK]            # we only need a mask to anchor the plant
    tier = 1

    def extract(self, *, mask=None, height_series=None, **ctx) -> Dict[str, Any]:
        if height_series is None or len(height_series) < 2:
            return {"height_rate_mm_day": float("nan"),
                    "missing_reason": "height_series_too_short"}
        hs = np.asarray(height_series, dtype=float)
        rate = float(np.mean(np.diff(hs)))          # mm per step (synthetic)
        return {"height_rate_mm_day": rate}
```

Register it by importing the package once:

```python
import phenocv.growth  # importing registers PlantHeightRateExtractor
from phenocv.core.registry import all_extractors
print("plant_height_rate" in all_extractors())   # True
```

---

## 3. Minimal data adapter / 最小已注册数据适配器

**English.** A data adapter translates *your* dataset layout into the
`PlantSequence` objects the engine consumes. (Distinct from the *algorithm*
backend in §2 of CONTRIBUTING.md: the adapter decides what frames/masks to feed;
the backend decides how they are propagated.) Subclass `BaseAdapter` from
`phenocv.segmentation.adapters.base` and implement `build_sequences`. Keep masks
boolean and in **full-image** coordinates; sort frames by time; skip (but report)
sequences with too few anchors.

**中文.** 数据适配器把*你的*数据集布局转换为引擎所用的 `PlantSequence` 对象。（与
CONTRIBUTING.md §2 的*算法*后端不同：适配器决定喂哪些帧/掩膜，后端决定如何传播。）继承
`phenocv.segmentation.adapters.base` 的 `BaseAdapter` 并实现 `build_sequences`。掩膜保持布尔且
为**全图像**坐标；按时间排序帧；对锚点不足的序列应跳过（但需报告）。

```python
# src/phenocv/growth/adapter.py
from __future__ import annotations
import os, glob
from typing import List

import cv2
import numpy as np
from phenocv.segmentation.adapters import BaseAdapter
from phenocv.segmentation.engine import AnchorFrame, PlantSequence


class PotGrowthAdapter(BaseAdapter):
    def __init__(self, root: str, min_anchors: int = 2):
        self.root, self.min_anchors = root, min_anchors

    def build_sequences(self, **kwargs) -> List[PlantSequence]:
        sequences: List[PlantSequence] = []
        for plant_dir in sorted(os.listdir(self.root)):
            frames = sorted(glob.glob(os.path.join(self.root, plant_dir, "rgb", "*.png")))
            anchors = []
            for i, fp in enumerate(frames):
                mp = fp.replace("rgb", "mask_manual")
                if os.path.exists(mp):
                    anchors.append(AnchorFrame(frame_idx=i,
                                              mask=cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127))
            if len(anchors) < self.min_anchors:
                continue  # report skips via a progress callback in real code
            sequences.append(PlantSequence(key=plant_dir, frame_paths=frames, anchors=anchors))
        return sequences
```

Drive it through the engine (GPU layer lazy — mocked in tests):

```python
from phenocv.growth.backend import PotGrowthAdapter
seqs = PotGrowthAdapter("/data/potted_growth").build_sequences()
```

---

## 4. Minimal test / 最小测试

**English.** Tests are CPU-only and synthetic. The extractor test needs no files;
the adapter test builds a tiny in-memory dataset. No `torch`/`sam2`/`matplotlib`
are imported.

**中文.** 测试仅依赖 CPU 且使用合成数据。提取器测试无需文件；适配器测试构建极小的内存
数据集。不导入 `torch`/`sam2`/`matplotlib`。

```python
# tests/test_growth.py
import numpy as np
import phenocv.growth                       # registers the extractor
from phenocv.core.registry import available_for, INPUT_MASK
from phenocv.growth.extractors import PlantHeightRateExtractor


def test_extractor_runs_when_inputs_present():
    mask = np.zeros((10, 10), bool); mask[3:7, 3:7] = True
    ext = PlantHeightRateExtractor()
    out = ext.extract(mask=mask, height_series=[10.0, 12.0, 15.0])
    assert np.isfinite(out["height_rate_mm_day"])


def test_extractor_fail_closed_on_short_series():
    mask = np.zeros((10, 10), bool); mask[3:7, 3:7] = True
    out = PlantHeightRateExtractor().extract(mask=mask, height_series=[1.0])
    assert "missing_reason" in out                     # fail-closed, not fabricated


def test_registry_discovers_extractor():
    assert any(e.name == "plant_height_rate" for e in available_for({INPUT_MASK}))
```

Run with / 运行：

```bash
pytest tests/test_growth.py
```

---

## 5. Hotspot roadmap — where to plug in / 热点路线图：从何处接入

**English.** PhenoCV is intentionally a skeleton for many phenotyping hotspots.
Good first modules to contribute (each its own `phenocv.<name>` package + traits
+ tests):

* **Multi-modal fusion** — fuse RGB + thermal + depth into joint traits.
* **Stress / disease** — extend `phenocv.thermal.stress` with new contrast designs.
* **Growth** — temporal height/volume trajectories (`phenocv.phenotypes` Tier-3).
* **Counting** — organ/fruit counting backends and adapters.
* **Root** — root-system image analysis.
* **Postharvest** — quality/sorting traits.
* **Yield / G2P** — link phenotypes to genotypes.
* **Foundation models** — plug a lazy-loaded foundation model as a segmentation backend.
* **Edge** — CPU-only / on-device inference adapters.
* **Uncertainty** — reusable CI / HAC / bootstrap tooling (see `phenocv.thermal.stress`).

**中文.** PhenoCV 刻意做成可容纳众多表型热点的骨架。值得优先贡献的模块（每个都是独立的
`phenocv.<name>` 包 + 表型 + 测试）：

* **多模态融合** — 融合 RGB + 热红外 + 深度为联合表型。
* **胁迫 / 病害** — 以新的对照设计扩展 `phenocv.thermal.stress`。
* **生长** — 时序株高/体积轨迹（`phenocv.phenotypes` 第三层）。
* **计数** — 器官/果实计数后端与适配器。
* **根系** — 根系图像分析。
* **采后** — 品质/分选表型。
* **产量 / G2P** — 表型与基因型关联。
* **基础模型** — 以惰性加载的基础模型作为分割后端。
* **边缘端** — 纯 CPU / 端侧推理适配器。
* **不确定性** — 可复用的 CI / HAC / bootstrap 工具（参见 `phenocv.thermal.stress`）。

---

## 6. Checklist before opening a PR / 提交 PR 前清单

- [ ] New module is a sibling `phenocv.<name>` package with a public `__init__.py`.
      新模块为同级 `phenocv.<name>` 包，且 `__init__.py` 导出公共 API。
- [ ] Trait tools registered via `@register`; no core edits needed.
      表型工具通过 `@register` 注册；无需改动核心。
- [ ] No `torch` / `sam2` / `matplotlib` imported at module top level.
      模块顶层未导入 `torch` / `sam2` / `matplotlib`。
- [ ] Docstrings are bilingual (EN + 中文); APIs are data-agnostic.
      docstring 双语（英 + 中）；API 数据无关。
- [ ] Tests are CPU-only, synthetic, deterministic; `pytest` passes.
      测试仅 CPU、合成、确定性；`pytest` 通过。
- [ ] Nothing written to `C:`; temp dirs used instead.
      未向 `C:` 写入；改用临时目录。
- [ ] Partially-implemented pieces marked "under construction".
      部分实现的部分已标注"建设中"。
