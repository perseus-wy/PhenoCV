# 📦 Export formats

`phenocv segment` writes a fixed, machine-readable layout under `--output`.
All mask PNGs are **full-image** (pasted back from ROI space); ISAT JSONs and
CSVs are aligned by frame `stem` (the source frame filename without extension).

```
<output>/
├── run_manifest.json        # full run record + LOO summary (machine-readable)
├── loo_quality.csv          # per-anchor Leave-One-Out metrics
├── frame_manifest.csv       # every frame, one row
├── sequence_summary.csv     # per-sequence rollup
└── <sequence_key>/
    ├── masks/<stem>.png     # boolean mask, 0/255, full-image
    ├── jsons/<stem>.json    # ISAT annotation (if export_isat)
    ├── area.csv             # per-frame area + pred_source + extras
    └── qa_grid.png          # frames × masks overview (if export_qa)
```

## Masks (`masks/<stem>.png`)

- Single-channel, 0 (background) / 255 (plant). Full-image resolution.
- One file per frame, named by the source frame stem.

## ISAT annotation (`jsons/<stem>.json`)

[ISAT](https://github.com/yatengLG/ISAT_with_segment_anything) is the native
annotation format of the SAM ecosystem. Each JSON carries one `plant` object
with a polygon (simplified via `approxPolyDP`, tolerance `isat_simplify_eps`)
and `iscrowd=0`. Components smaller than `isat_min_area` are dropped as noise.
The `note` field records `sequence | frame_idx | label | src`, so an annotator
can immediately see how the mask was produced.

Minimal shape:

```json
{
  "description": "",
  "imageHeight": 720,
  "imageWidth": 1280,
  "imagepath": "/abs/path/to/frame.png",
  "shapes": [
    {
      "label": "plant",
      "points": [[x1, y1], [x2, y2], "..."],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flag": {},
      "iscrowd": 0
    }
  ]
}
```

## CSV exports

### `frame_manifest.csv` (every frame)

| Column | Meaning |
|---|---|
| `sequence` | sequence key |
| `frame_idx` | 0-based index |
| `frame_label` | human-readable label |
| `stem` | source frame stem |
| `pred_source` | `manual` / `propagated` / `propagated_lowthr` / `point_rescue` / `failed_empty` |
| `thr` | threshold actually used |
| *(extras)* | any `frame_extras` columns from the adapter, passed through |

### `area.csv` (per sequence)

Same columns as `frame_manifest.csv` plus `area_px` (mask pixel count). Use this
directly for **canopy-area-over-time** curves.

### `loo_quality.csv` (per anchor, Leave-One-Out)

| Column | Meaning |
|---|---|
| `sequence` | sequence key |
| `frame_idx` | the held-out anchor index |
| `iou` | mask IoU vs the held-out manual mask |
| `bf1` | Boundary-F1 vs the manual mask |
| `pred_source` | how the LOO prediction was produced |
| `thr` | threshold used |

The held-out anchor is re-predicted from the *other* anchors; interior anchors
(those with neighbors on both sides) are the most reliable signal.
`run_manifest.json` carries `loo_summary_interior` (medians over interior
anchors) and `loo_summary_all`.

### `sequence_summary.csv`

| Column | Meaning |
|---|---|
| `sequence` | sequence key |
| `n_frames` / `n_anchors` | counts |
| `roi` | `x0,y0,x1,y1` ROI used |
| `area_first` / `area_last` / `area_max` | canopy area trajectory |
| `n_manual` / `n_propagated` / `n_lowthr` / `n_rescue` / `n_failed` | `pred_source` histogram |

## QA grid (`qa_grid.png`)

A `qa_grid_cols` × ⌈N/cols⌉ grid of tiles (`qa_grid_tile` px each), each showing
the frame with its mask overlaid. One glance to spot drift, leakage, or empty
frames before trusting the run.

## Provenance (`pred_source`)

Every mask is tagged with how it was produced — this is the audit trail:

| `pred_source` | Produced by |
|---|---|
| `manual` | human-labeled anchor, copied through |
| `propagated` | SAM 2 propagation at `base_threshold` |
| `propagated_lowthr` | empty at base → recovered by the threshold ladder |
| `point_rescue` | empty after ladder → box-constrained point prompt |
| `failed_empty` | no mask after all fallbacks |

Filter on `pred_source` to quarantine low-confidence frames for human review
instead of silently trusting them.
