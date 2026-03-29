import argparse
import os
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "releases" / "latest" / "best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "images" / "inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run climbing hold segmentation inference.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained YOLO weights.")
    parser.add_argument(
        "--classes",
        default="0",
        help="Comma-separated class indices to keep. Default '0' means hold only.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for rendered inference output.",
    )
    return parser.parse_args()


def parse_classes(raw_value: str) -> list[int] | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def label_image(
    image_path: str,
    model_path: str,
    output_dir: str,
    classes: list[int] | None = None,
    conf: float = 0.25,
    imgsz: int = 1024,
) -> None:
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=image_path,
        task="segment",
        conf=conf,
        imgsz=imgsz,
        classes=classes,
        save=False,
    )

    for index, result in enumerate(results):
        output_path = os.path.join(output_dir, f"prediction_{index}.jpg")
        result.save(filename=output_path)
        print(f"Saved prediction to {output_path}")


def main() -> None:
    args = parse_args()
    label_image(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        classes=parse_classes(args.classes),
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
