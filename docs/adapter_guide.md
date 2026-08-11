# 📚 Adapter Guide

An **adapter** translates *your* dataset layout into the
`PlantSequence` objects the core engine consumes. The engine itself never
touches your filesystem — it only sees sequences + anchor masks. That is what
makes PhenoCV reusable across domains.

## The contract

```python
from phenocv.segmentation.engine import PlantSequence, AnchorFrame

PlantSequence(
    key="plant_01",                       # sequence id
    frame_paths=["/abs/frame_0000.png",   # time-ordered RGB paths
                 "/abs/frame_0001.png", ...],
    anchors=[AnchorFrame(frame_idx=0, mask=<bool ndarray>),  # sparse manual labels
             AnchorFrame(frame_idx=5, mask=<bool ndarray>)],
    frame_labels=["2026-05-01", ...],     # optional human-readable labels
    frame_extras=[{"das": 1, "ts": "..."}, ...],  # optional per-frame metadata
    meta={...},                           # optional sequence-level metadata
)
```

`AnchorFrame.mask` is a **boolean** (`True` = plant) array in **full-image**
coordinates. `frame_idx` is 0-based and must fall within `frame_paths`.
`frame_labels` / `frame_extras` default to index-derived stubs but are carried
through verbatim into `area.csv` / `frame_manifest.csv` / `loo_quality.csv`, so
they are your extension point for domain metadata (date, DAS, genotype, …).

## Option A — `CsvManifestAdapter` (default, zero code)

Write **one manifest CSV** (or JSON) and point the CLI at it:

```bash
phenocv segment --adapter csv --manifest my_manifest.csv \
  --config configs/default.yaml --checkpoint sam2.1_hiera_l.pt --output out
```

Columns: `sequence_key`, `frame_idx`, `frame_path`, `frame_label`, `is_anchor`,
`mask_path`, plus any extra columns (passed through as `frame_extras`).

```csv
sequence_key,frame_idx,frame_path,frame_label,is_anchor,mask_path,das
plant_01,0,/data/p01/0000.png,DAS1,1,/masks/p01/0000.png,1
plant_01,1,/data/p01/0001.png,DAS2,0,,2
plant_01,2,/data/p01/0002.png,DAS3,1,/masks/p01/0002.png,3
```

JSON is also accepted — a list of row dicts, or `{"frames": [...]}`.

## Option B — subclass `BaseAdapter`

For anything the manifest can't express (custom filename parsing, remote
storage, multi-camera rigs), subclass `BaseAdapter`:

```python
from phenocv.segmentation.adapters import BaseAdapter
from phenocv.segmentation.engine import PlantSequence, AnchorFrame
import cv2, numpy as np, os

class MyAdapter(BaseAdapter):
    def __init__(self, root: str, min_anchors: int = 2):
        self.root = root
        self.min_anchors = min_anchors

    def build_sequences(self, **kwargs):
        sequences = []
        for plant_dir in sorted(os.listdir(self.root)):
            frames = sorted(glob(os.path.join(self.root, plant_dir, "rgb", "*.png")))
            anchors = []
            for i, fp in enumerate(frames):
                mask_p = fp.replace("rgb", "mask_manual")
                if os.path.exists(mask_p):
                    anchors.append(AnchorFrame(
                        frame_idx=i,
                        mask=cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) > 127))
            if len(anchors) < self.min_anchors:
                continue  # silently skip; report skips via a progress callback
            sequences.append(PlantSequence(
                key=plant_dir, frame_paths=frames, anchors=anchors,
                frame_labels=[os.path.basename(f) for f in frames]))
        return sequences
```

Then:

```python
from phenocv.segmentation.config import load_config
from phenocv.segmentation.engine import run_sam2_video_temporal
from mymodule import MyAdapter

seqs = MyAdapter("/data/my_dataset").build_sequences()
cfg = load_config("configs/default.yaml", preset="plant_phenotyping")
run_sam2_video_temporal(seqs, "out", checkpoint="sam2.1_hiera_l.pt",
                        model_cfg="sam2.1_hiera_l.yaml", config=cfg)
```

## Worked example shipped with PhenoCV

`phenocv.segmentation.adapters.PlantPhenotypingAdapter` is a real example for **potted
soybean temporal data**: it reads a frame-index CSV plus a per-plant manual-mask
directory and builds sequences. Use it as a template:

```bash
phenocv segment --adapter plant \
  --index data/frame_index.csv \
  --anchor-root data/manual_masks \
  --rgb-root /local/mirror/of/frames \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint sam2.1_hiera_l.pt --output out
```

## Rules of thumb

- **Always sort frames by time** before building `frame_paths` (the engine
  assumes temporal order).
- **Skip, don't crash**, on sequences with too few anchors — but report skips
  (a `progress` callback or a log line) so nothing disappears silently.
- **Keep masks boolean and full-image.** ROI cropping is the engine's job.
- **Use `frame_extras`** for anything you'll want in the output CSVs later
  (DAS, genotype, treatment) — it costs you nothing at inference time.
