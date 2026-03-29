import argparse
from pathlib import Path

from PIL import Image
import torch
import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "releases" / "latest" / "best.pt"
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "training_dataset" / "data.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "review_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run segmentation inference over a dataset split and save visual review images."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to YOLO weights.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "valid", "test", "val"),
        help="Dataset split to review.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1600, help="Inference image size.")
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated class indices to keep. Empty means all classes.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for rendered review images.",
    )
    parser.add_argument(
        "--run-name",
        default="latest_test_review",
        help="Subdirectory name under output-root.",
    )
    return parser.parse_args()


def parse_classes(raw_value: str) -> list[int] | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def resolve_split_images_dir(data_yaml_path: Path, split: str) -> Path:
    normalized_split = "valid" if split == "val" else split
    with data_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    root = Path(data.get("path", data_yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()

    split_value = data.get(normalized_split)
    if split_value is None:
        if normalized_split == "valid":
            split_value = data.get("val")
        elif normalized_split == "test":
            split_value = "images/test"
        elif normalized_split == "train":
            split_value = "images/train"

    if split_value is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml_path}")

    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Split images directory not found: {images_dir}")
    return images_dir


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    data_yaml_path = Path(args.data).resolve()
    images_dir = resolve_split_images_dir(data_yaml_path, args.split)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    classes = parse_classes(args.classes)

    results = model.predict(
        source=str(images_dir),
        task="segment",
        conf=args.conf,
        imgsz=args.imgsz,
        classes=classes,
        save=False,
        save_txt=False,
        save_conf=False,
        stream=True,
        verbose=False,
    )

    processed = 0
    for result in results:
        rendered = result.cpu().plot(
            boxes=True,
            labels=True,
            conf=True,
        )
        output_path = output_dir / Path(result.path).name
        Image.fromarray(rendered[..., ::-1]).save(output_path)
        processed += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Reviewed split: {args.split}")
    print(f"Source images: {images_dir}")
    print(f"Saved review images: {output_dir}")
    print(f"Images processed: {processed}")


if __name__ == "__main__":
    main()
