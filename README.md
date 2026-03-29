# Climbing-Hold-Segmentation

This repo supports training from one or more Roboflow COCO exports, then
re-splitting them locally so the final `train` / `valid` / `test` split is under
our control.

The training pipeline is manifest-driven. It expects a dataset config JSON at
`data/dataset_sources.json` or at the path provided by `YOLO_DATASET_CONFIG`.

Current training assumptions:
- Train 2 segmentation classes: `hold`, `volume`
- Direct `python .\ml\model.py` default weights: `yolo26s-seg.pt`
- `scripts/train_segmentation.ps1` default weights: `yolo26m-seg.pt`
- App/demo inference can filter to class `0` (`hold`) only
- Training already uses full RGB color images; there is no grayscale conversion in the training pipeline.
- Dataset sources are configured in [`data/dataset_sources.json`](./data/dataset_sources.json)

Run training:

```powershell
python .\ml\model.py
```

Useful environment overrides:
- `YOLO_DATASET_CONFIG` example: `data/dataset_sources.json`
- `YOLO_WEIGHTS` example: `yolo26n-seg.pt`, `yolo26s-seg.pt`
- `YOLO_EPOCHS`
- `YOLO_PATIENCE`
- `YOLO_BATCH`
- `YOLO_IMGSZ`
- `YOLO_DEVICE`
- `YOLO_RUN_NAME`

Recommended overnight run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_segmentation.ps1
```

Current default training preset in [`scripts/train_segmentation.ps1`](./scripts/train_segmentation.ps1):
- `yolo26m-seg.pt`
- `imgsz=1600`
- `batch=2`
- `epochs=200`
- `patience=15`
- `cache=ram`
- `workers=6`

If you hit GPU memory limits, retry with a lower image size or batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_segmentation.ps1 -ImageSize 1152 -Batch 1
```

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
[`data/dataset_sources.json`](./data/dataset_sources.json) uses a Roboflow
dataset source. It can download either from:
- a direct Roboflow share URL in `.env`
- or the standard workspace / project / version values in `.env`

Download mode behavior:
- `ROBOFLOW_DOWNLOAD_URL` is tried first when present. This usually downloads a flat COCO export with images plus a root `_annotations.coco.json`.
- If `ROBOFLOW_DOWNLOAD_URL` is not set, the pipeline falls back to the Roboflow SDK path using `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, and `ROBOFLOW_VERSION`. That path often returns a split export with `train/`, `valid/`, and `test/`.
- The prep pipeline supports both layouts.
- If `split.force_resplit` is `true`, both layouts are flattened into one pool and then re-split locally anyway.

To pool multiple exports together, add more entries to
[`data/dataset_sources.json`](./data/dataset_sources.json). Local COCO exports
look like this:

```json
{
  "name": "forked_2026_03_26",
  "type": "local_coco_export",
  "path": "Climbing Hold Detection-Forked on 3-26-2026.coco"
}
```

Roboflow-hosted sources can be added like this:

```json
{
  "name": "primary_dataset",
  "type": "roboflow",
  "download_url_env": "ROBOFLOW_DOWNLOAD_URL",
  "workspace_env": "ROBOFLOW_WORKSPACE",
  "project_env": "ROBOFLOW_PROJECT",
  "version_env": "ROBOFLOW_VERSION",
  "model_format": "coco-segmentation"
}
```

If `ROBOFLOW_DOWNLOAD_URL` is present, the pipeline uses that direct share link
first. Otherwise it falls back to `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`,
`ROBOFLOW_PROJECT`, and `ROBOFLOW_VERSION`.

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
