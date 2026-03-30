# Climbing-Hold-Segmentation

This repo supports training from one or more Roboflow COCO exports, then
re-splitting them locally so the final `train` / `valid` / `test` split is under
our control.

The training pipeline is manifest-driven. It expects a dataset config JSON at
`data/dataset_sources.json` or at the path provided by `YOLO_DATASET_CONFIG`.

Current training assumptions:
- Train 2 segmentation classes: `hold`, `volume`
- Current standard training preset uses `yolo26m-seg.pt`
- App/demo inference can filter to class `0` (`hold`) only
- Training already uses full RGB color images; there is no grayscale conversion in the training pipeline.
- Dataset sources are configured in [`data/dataset_sources.json`](./data/dataset_sources.json)

Canonical training entrypoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_segmentation.ps1
```

`scripts/train_segmentation.ps1` is the supported way to launch training. It sets
the repo's current training preset, points at the repo-local `.venv`, and then
calls [`ml/model.py`](./ml/model.py) underneath.

Recommended overnight run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_segmentation.ps1
```

Current default training preset in [`scripts/train_segmentation.ps1`](./scripts/train_segmentation.ps1):
- `yolo26m-seg.pt`
- `imgsz=1440`
- `batch=2`
- `epochs=200`
- `patience=15`
- Lower-level optimizer and augmentation settings are baked into the repo's fixed preset in [`ml/model.py`](./ml/model.py)

If you hit GPU memory limits, retry with a lower image size or batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_segmentation.ps1 -ImageSize 1152 -Batch 1
```

The underlying implementation lives in [`ml/model.py`](./ml/model.py). Use it
directly only when you intentionally want the lower-level env-var driven path.
Its built-in defaults match the current script preset.

Useful environment overrides when calling `ml/model.py` directly:
- `YOLO_DATASET_CONFIG` example: `data/dataset_sources.json`
- `YOLO_WEIGHTS` example: `yolo26n-seg.pt`, `yolo26m-seg.pt`
- `YOLO_EPOCHS`
- `YOLO_PATIENCE`
- `YOLO_BATCH`
- `YOLO_IMGSZ`
- `YOLO_DEVICE`
- `YOLO_CACHE`
- `YOLO_WORKERS`
- `YOLO_RUN_NAME`
- Lower-level optimizer and augmentation settings now live in the repo's fixed training preset instead of being exposed as first-class env knobs.

If you only want to rebuild the training dataset without training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_dataset.ps1
```

You can still use the direct env-var route if needed:

```powershell
$env:YOLO_PREPARE_ONLY = "1"
python .\ml\model.py
```

## Dataset config

The default config in
[`data/dataset_sources.json`](./data/dataset_sources.json) uses Roboflow share
URLs only. Each source must provide either:
- `download_url`
- or `download_url_env` that resolves to a Roboflow share URL in `.env`

The share URL usually downloads a flat COCO export with images plus a root
`_annotations.coco.json`. The prep pipeline still supports both flat and split
COCO layouts, and if `split.force_resplit` is `true`, everything is flattened
into one pool and then re-split locally anyway.

To pool multiple exports together, add more entries to
[`data/dataset_sources.json`](./data/dataset_sources.json).

Roboflow share URL sources look like this:

```json
{
  "name": "primary_dataset",
  "type": "roboflow_share_url",
  "download_url_env": "ROBOFLOW_DOWNLOAD_URL"
}
```

Roboflow downloads are cached under `data/roboflow_downloads/`.

The merged YOLO-ready dataset is rebuilt into `data/training_dataset/`.

## Model artifacts

Training runs are generated under `ml/model/runs/` and are treated as local
artifacts, not committed source files.

Base `.pt` files in `ml/model/` are also ignored. The intended shareable place
for checkpoints is `releases/`.

To copy a finished run's `best.pt` and `last.pt` into a commit-friendly release
folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\publish_release.ps1 -RunName your_run_name -ReleaseName latest
```

If `-RunName` is omitted, the script uses the newest run in `ml/model/runs/`.
It writes:
- `releases/<release-name>/best.pt`
- `releases/<release-name>/last.pt`
- `releases/<release-name>/metrics.json`
- `releases/<release-name>/release.json`

Run inference with hold-only output:

```powershell
python .\ml\inference.py --image path\to\image.jpg --classes 0
```

Run a full visual review pass over a dataset split with boxes and masks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\review_test_predictions.ps1
```

That reads [`data/training_dataset/data.yaml`](./data/training_dataset/data.yaml) by default,
runs inference over the `test` split, and saves rendered review images under
`data/review_predictions/latest_test_review/`.
