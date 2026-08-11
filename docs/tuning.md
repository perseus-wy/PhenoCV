# 🔧 Tuning PhenoCV

Every tunable lives in `TemporalPropagationConfig` (see `phenocv/engine.py`).
Load them from `configs/default.yaml` under `propagation:` (base, always applied)
and `presets:` (overlay via `--preset`), or pass a dict directly to
`run_sam2_video_temporal(config=...)`.

> **Design rule:** unknown keys in the YAML are ignored
> (`TemporalPropagationConfig.from_mapping`), so config files stay
> forward-compatible across engine versions.

## The knobs and *why* the defaults are what they are

### ROI cropping (mandatory)

SAM 2 resizes every input frame to 1024px internally. Feeding a full
1280×720 frame crushes a small seedling (hundreds of pixels) out of existence.
So we crop a **square ROI** around the union bbox of the anchor masks (padded)
before propagation. This lifts a small seedling's effective resolution by
roughly an order of magnitude.

| Field | Default | Meaning |
|---|---|---|
| `roi_pad_ratio` | `1.9` | padding multiplier on the union bbox. A canopy grows many-fold over a season, so leave generous headroom. |
| `roi_min_size` | `384` | min ROI side; avoids too-small context for early shoots. |
| `roi_max_size` | `null` | max ROI side; `null` = bounded by image height. |

**Tune when:** targets are stable-scale and sharp (use `rigid_object` preset,
`roi_pad_ratio: 1.3`); or targets are tiny / easily lost (`high_recall`,
`roi_pad_ratio: 2.2`, `roi_min_size: 448`).

### Bidirectional propagation

| Field | Default | Meaning |
|---|---|---|
| `bidirectional` | `true` | run forward **and** reverse, then average logits before thresholding. |

Unidirectional propagation drifts monotonically away from the anchors near the
sequence endpoints. Averaging cancels most of that drift. Keep `true` unless
you have a reason not to.

### Threshold ladder (recall fallback)

| Field | Default | Meaning |
|---|---|---|
| `base_threshold` | `0.0` | base binarization threshold on the propagated logits. |
| `threshold_ladder` | `(-0.5, -1.0, -2.0, -4.0)` | fallback thresholds tried **in order** when the base yields an empty mask. |
| `min_valid_area` | `1` | minimum pixel count to count as non-empty. |

SAM 2's object logits are ~0 at the contour, so a hard `0.0` cutoff drops many
small early-stage targets as "empty". The ladder steps the threshold *down*
(lower = more inclusive) until a non-empty mask appears. Deeper ladders
(`high_recall`: down to `-8.0`) recover weaker targets at the cost of more
over-segmentation. For sharp, bounded objects, set `threshold_ladder: []` to
disable the fallback (`rigid_object`).

### Point-rescue (last resort for empty frames)

| Field | Default | Meaning |
|---|---|---|
| `rescue_enabled` | `true` | when the ladder still yields empty, prompt with the nearest anchor's centroid point. |
| `rescue_box_ratio` | `0.65` | box expansion factor relative to the nearest anchor bbox half-extent. |
| `rescue_min_box` | `8.0` | minimum box width/height (px) for the rescue prompt. |

**Critical:** point-rescue **must** be box-constrained. An unconstrained
centroid point makes SAM 2 grab background in a far corner. We expand the
nearest anchor bbox and push out-of-box logits to `-1e9`, forcing the mask
inside the expected region. Disable (`rescue_enabled: false`) only for
sharp-boundary objects where an empty frame should stay empty (`rigid_object`).

### Export / QA

| Field | Default | Meaning |
|---|---|---|
| `jpeg_quality` | `95` | ROI frame-stack JPEG quality (SAM 2 accepts JPEG only). |
| `isat_category` | `plant` | ISAT object category name. |
| `isat_simplify_eps` | `2.0` | `approxPolyDP` tolerance; larger = fewer contour points. |
| `isat_min_area` | `10.0` | drop connected components smaller than this (de-noise). |
| `qa_grid_cols` | `6` | columns in the QA overview grid. |
| `qa_grid_tile` | `200` | tile size (px) in the QA grid. |

## Worked: override one field without touching code

```yaml
# my_experiment.yaml
propagation:
  threshold_ladder: [-0.5, -1.0, -2.0, -4.0, -8.0, -16.0]
  rescue_box_ratio: 0.8
```

```bash
phenocv segment --manifest data/manifest.csv \
  --config my_experiment.yaml \
  --checkpoint sam2.1_hiera_l.pt --output results/exp
```

Or programmatically:

```python
from phenocv.segmentation.config import load_config
from phenocv.segmentation.engine import TemporalPropagationConfig

cfg = load_config("configs/default.yaml", preset="plant_phenotyping")
cfg["rescue_box_ratio"] = 0.8          # dict form from load_config
# or build directly:
custom = TemporalPropagationConfig(threshold_ladder=(-0.5, -1.0, -2.0, -4.0, -8.0))
```
