# samples/

This directory holds **example datasets**. Real datasets are **never committed**
(see `.gitignore`). To try PhenoCV end-to-end with zero real data, generate a
synthetic sample and run the quickstart:

```bash
python tools/make_demo_sample.py --out samples/demo
phenocv segment --adapter csv \
    --manifest samples/demo/manifest.csv \
    --checkpoint /path/to/sam2.1_hiera_l.pt \
    --output runs/demo
```

`make_demo_sample.py` writes:

- `frames/` — synthetic RGB frames (a green disc that grows over time)
- `masks/` — anchor masks for the labeled frames
- `manifest.csv` — a `CsvManifestAdapter`-compatible manifest

To bring your own data, write a manifest CSV with the columns documented in
`src/phenocv/adapters/csv_manifest.py`, or implement a custom adapter that
subclasses `phenocv.adapters.BaseAdapter`.
