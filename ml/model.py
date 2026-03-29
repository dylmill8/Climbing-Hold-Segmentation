import json
import os
from pathlib import Path

from dotenv import load_dotenv

from utils import (
    download_roboflow_coco_dataset,
    download_roboflow_coco_dataset_from_url,
    load_yolo,
    prepare_yolo_dataset_from_coco,
    prepare_yolo_dataset_from_coco_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = REPO_ROOT / "data" / "dataset_sources.json"
DEFAULT_ROBOFLOW_DOWNLOAD_ROOT = REPO_ROOT / "data" / "roboflow_downloads"
DEFAULT_PREPARED_DATASET = REPO_ROOT / "data" / "training_dataset"
DEFAULT_RUNS_DIR = REPO_ROOT / "ml" / "model" / "runs"
DEFAULT_CLASS_NAMES = ("hold", "volume")


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dataset_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_source_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _resolve_coco_sources(config: dict) -> tuple[list[str], list[str], list[dict[str, str] | None], tuple[str, ...], dict]:
    class_names = tuple(config.get("class_names", DEFAULT_CLASS_NAMES))
    split_config = config.get("split", {})
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("Dataset config must contain at least one source.")

    resolved_paths: list[str] = []
    resolved_names: list[str] = []
    resolved_category_maps: list[dict[str, str] | None] = []
    for index, source in enumerate(sources, start=1):
        source_type = source.get("type", "local_coco_export")
        source_name = source.get("name") or f"source_{index}"

        if source_type == "local_coco_export":
            raw_path = source.get("path")
            if not raw_path:
                raise ValueError(f"Dataset source '{source_name}' is missing 'path'.")
            source_path = _resolve_source_path(raw_path)
        elif source_type == "roboflow":
            download_root = _resolve_source_path(
                source.get("download_root", str(DEFAULT_ROBOFLOW_DOWNLOAD_ROOT.relative_to(REPO_ROOT)))
            )
            download_url = source.get("download_url")
            if download_url is None:
                download_url_env = source.get("download_url_env", "ROBOFLOW_DOWNLOAD_URL")
                download_url = os.getenv(download_url_env)

            if download_url:
                source_path = Path(
                    download_roboflow_coco_dataset_from_url(
                        download_url=download_url,
                        download_root=str(download_root),
                        dataset_name=source_name,
                        overwrite=bool(source.get("overwrite", False)),
                    )
                )
            else:
                api_key_env = source.get("api_key_env", "ROBOFLOW_API_KEY")
                api_key = os.getenv(api_key_env)
                if not api_key:
                    raise RuntimeError(
                        f"Dataset source '{source_name}' requires environment variable '{api_key_env}'."
                    )

                workspace = source.get("workspace")
                if workspace is None:
                    workspace_env = source.get("workspace_env", "ROBOFLOW_WORKSPACE")
                    workspace = os.getenv(workspace_env)

                project = source.get("project")
                if project is None:
                    project_env = source.get("project_env", "ROBOFLOW_PROJECT")
                    project = os.getenv(project_env)

                version = source.get("version")
                if version is None:
                    version_env = source.get("version_env", "ROBOFLOW_VERSION")
                    version = os.getenv(version_env)
                if not workspace or not project or version is None:
                    raise ValueError(
                        f"Dataset source '{source_name}' must include workspace, project, and version."
                    )

                source_path = Path(
                    download_roboflow_coco_dataset(
                        api_key=api_key,
                        workspace=workspace,
                        project=project,
                        version=int(version),
                        download_root=str(download_root),
                        dataset_name=source_name,
                        model_format=source.get("model_format", "coco-segmentation"),
                        overwrite=bool(source.get("overwrite", False)),
                    )
                )
        else:
            raise ValueError(f"Unsupported dataset source type: {source_type}")

        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source not found: {source_path}")

        resolved_paths.append(str(source_path))
        resolved_names.append(source_name)
        raw_category_map = source.get("category_name_map")
        resolved_category_maps.append(dict(raw_category_map) if isinstance(raw_category_map, dict) else None)

    return resolved_paths, resolved_names, resolved_category_maps, class_names, split_config


def prepare_dataset() -> tuple[str, tuple[str, ...]]:
    dataset_config_path = Path(os.getenv("YOLO_DATASET_CONFIG", str(DEFAULT_DATASET_CONFIG)))
    prepared_dataset = Path(os.getenv("YOLO_PREPARED_DATASET", str(DEFAULT_PREPARED_DATASET)))

    if not dataset_config_path.exists():
        raise FileNotFoundError(
            "Dataset config not found. Expected YOLO_DATASET_CONFIG to point to a JSON file, "
            f"for example: {dataset_config_path}"
        )

    dataset_config = _load_dataset_config(dataset_config_path)
    source_paths, source_names, source_category_maps, class_names, split_config = _resolve_coco_sources(dataset_config)

    if len(source_paths) == 1 and not split_config.get("force_resplit", True):
        data_yaml = prepare_yolo_dataset_from_coco(
            source_root=source_paths[0],
            output_root=str(prepared_dataset),
            class_names=class_names,
            category_name_map=source_category_maps[0],
        )
    else:
        data_yaml = prepare_yolo_dataset_from_coco_sources(
            source_roots=source_paths,
            source_names=source_names,
            source_category_maps=source_category_maps,
            output_root=str(prepared_dataset),
            class_names=class_names,
            train_split=float(split_config.get("train", 0.8)),
            valid_split=float(split_config.get("valid", 0.5)),
            null_fraction=float(split_config.get("null_fraction", 0.0)),
            seed=int(split_config.get("seed", getenv_int("YOLO_SEED", 42))),
        )

    return data_yaml, class_names


def main() -> None:
    load_dotenv()

    weights_spec = os.getenv("YOLO_WEIGHTS", "yolo26s-seg.pt")
    run_name = os.getenv("YOLO_RUN_NAME", "y26s_seg_2class_1024")

    data_yaml, class_names = prepare_dataset()
    print(f"Prepared classes: {', '.join(class_names)}")

    if getenv_bool("YOLO_PREPARE_ONLY", False):
        print(f"Dataset prepared only. YAML saved to: {data_yaml}")
        return

    model = load_yolo(weights_spec)

    result = model.train(
        resume=getenv_bool("YOLO_RESUME", False),
        task="segment",
        data=data_yaml,
        imgsz=getenv_int("YOLO_IMGSZ", 1024),
        batch=getenv_int("YOLO_BATCH", -1),
        nbs=getenv_int("YOLO_NBS", 64),
        epochs=getenv_int("YOLO_EPOCHS", 80),
        patience=getenv_int("YOLO_PATIENCE", 20),
        seed=getenv_int("YOLO_SEED", 42),
        device=os.getenv("YOLO_DEVICE"),
        amp=getenv_bool("YOLO_AMP", True),
        workers=getenv_int("YOLO_WORKERS", 0),
        optimizer=os.getenv("YOLO_OPTIMIZER", "AdamW"),
        lr0=getenv_float("YOLO_LR0", 0.003),
        lrf=getenv_float("YOLO_LRF", 0.12),
        weight_decay=getenv_float("YOLO_WEIGHT_DECAY", 0.0005),
        cos_lr=getenv_bool("YOLO_COS_LR", True),
        warmup_epochs=getenv_int("YOLO_WARMUP_EPOCHS", 5),
        hsv_h=getenv_float("YOLO_HSV_H", 0.015),
        hsv_s=getenv_float("YOLO_HSV_S", 0.7),
        hsv_v=getenv_float("YOLO_HSV_V", 0.4),
        degrees=getenv_float("YOLO_DEGREES", 10.0),
        translate=getenv_float("YOLO_TRANSLATE", 0.05),
        scale=getenv_float("YOLO_SCALE", 0.5),
        shear=getenv_float("YOLO_SHEAR", 0.0),
        perspective=getenv_float("YOLO_PERSPECTIVE", 0.0),
        fliplr=getenv_float("YOLO_FLIPLR", 0.5),
        flipud=getenv_float("YOLO_FLIPUD", 0.0),
        mosaic=getenv_float("YOLO_MOSAIC", 0.12),
        mixup=getenv_float("YOLO_MIXUP", 0.0),
        copy_paste=getenv_float("YOLO_COPY_PASTE", 0.20),
        erasing=getenv_float("YOLO_ERASING", 0.15),
        close_mosaic=getenv_int("YOLO_CLOSE_MOSAIC", 18),
        multi_scale=getenv_bool("YOLO_MULTI_SCALE", False),
        cache=os.getenv("YOLO_CACHE", "ram"),
        max_det=getenv_int("YOLO_MAX_DET", 1000),
        project=str(DEFAULT_RUNS_DIR),
        name=run_name,
    )

    print(result)
    print(f"Training finished. Prepared dataset: {data_yaml}")
    print(f"Runs saved to: {DEFAULT_RUNS_DIR / run_name}")


if __name__ == "__main__":
    main()
