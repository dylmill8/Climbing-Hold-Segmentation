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
DEFAULT_WEIGHTS = "yolo26m-seg.pt"
DEFAULT_RUN_NAME = "y26m_seg_resplit_1440"
DEFAULT_TRAINING_CONFIG = {
    "imgsz": 1440,
    "batch": 2,
    "nbs": 64,
    "epochs": 200,
    "patience": 15,
    "seed": 42,
    "workers": 6,
    "optimizer": "AdamW",
    "lr0": 0.0012,
    "lrf": 0.12,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.05,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.05,
    "mixup": 0.0,
    "copy_paste": 0.03,
    "erasing": 0.15,
    "close_mosaic": 10,
    "cache": "ram",
    "max_det": 1000,
}


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


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
            seed=int(split_config.get("seed", DEFAULT_TRAINING_CONFIG["seed"])),
        )

    return data_yaml, class_names


def main() -> None:
    load_dotenv()

    weights_spec = os.getenv("YOLO_WEIGHTS", DEFAULT_WEIGHTS)
    run_name = os.getenv("YOLO_RUN_NAME", DEFAULT_RUN_NAME)

    data_yaml, class_names = prepare_dataset()
    print(f"Prepared classes: {', '.join(class_names)}")

    if getenv_bool("YOLO_PREPARE_ONLY", False):
        print(f"Dataset prepared only. YAML saved to: {data_yaml}")
        return

    model = load_yolo(weights_spec)

    result = model.train(
        resume=False,
        task="segment",
        data=data_yaml,
        imgsz=getenv_int("YOLO_IMGSZ", DEFAULT_TRAINING_CONFIG["imgsz"]),
        batch=getenv_int("YOLO_BATCH", DEFAULT_TRAINING_CONFIG["batch"]),
        nbs=DEFAULT_TRAINING_CONFIG["nbs"],
        epochs=getenv_int("YOLO_EPOCHS", DEFAULT_TRAINING_CONFIG["epochs"]),
        patience=getenv_int("YOLO_PATIENCE", DEFAULT_TRAINING_CONFIG["patience"]),
        seed=DEFAULT_TRAINING_CONFIG["seed"],
        device=os.getenv("YOLO_DEVICE"),
        amp=True,
        workers=getenv_int("YOLO_WORKERS", DEFAULT_TRAINING_CONFIG["workers"]),
        optimizer=DEFAULT_TRAINING_CONFIG["optimizer"],
        lr0=DEFAULT_TRAINING_CONFIG["lr0"],
        lrf=DEFAULT_TRAINING_CONFIG["lrf"],
        weight_decay=DEFAULT_TRAINING_CONFIG["weight_decay"],
        cos_lr=True,
        warmup_epochs=DEFAULT_TRAINING_CONFIG["warmup_epochs"],
        hsv_h=DEFAULT_TRAINING_CONFIG["hsv_h"],
        hsv_s=DEFAULT_TRAINING_CONFIG["hsv_s"],
        hsv_v=DEFAULT_TRAINING_CONFIG["hsv_v"],
        degrees=DEFAULT_TRAINING_CONFIG["degrees"],
        translate=DEFAULT_TRAINING_CONFIG["translate"],
        scale=DEFAULT_TRAINING_CONFIG["scale"],
        shear=DEFAULT_TRAINING_CONFIG["shear"],
        perspective=DEFAULT_TRAINING_CONFIG["perspective"],
        fliplr=DEFAULT_TRAINING_CONFIG["fliplr"],
        flipud=DEFAULT_TRAINING_CONFIG["flipud"],
        mosaic=DEFAULT_TRAINING_CONFIG["mosaic"],
        mixup=DEFAULT_TRAINING_CONFIG["mixup"],
        copy_paste=DEFAULT_TRAINING_CONFIG["copy_paste"],
        erasing=DEFAULT_TRAINING_CONFIG["erasing"],
        close_mosaic=DEFAULT_TRAINING_CONFIG["close_mosaic"],
        multi_scale=False,
        cache=os.getenv("YOLO_CACHE", DEFAULT_TRAINING_CONFIG["cache"]),
        max_det=DEFAULT_TRAINING_CONFIG["max_det"],
        project=str(DEFAULT_RUNS_DIR),
        name=run_name,
    )

    print(result)
    print(f"Training finished. Prepared dataset: {data_yaml}")
    print(f"Runs saved to: {DEFAULT_RUNS_DIR / run_name}")


if __name__ == "__main__":
    main()
