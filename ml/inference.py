import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
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
    parser.add_argument("--imgsz", type=int, default=1440, help="Inference image size.")
    parser.add_argument(
        "--retina-masks",
        dest="retina_masks",
        action="store_true",
        help="Upsample masks back to image resolution for cleaner edges.",
    )
    parser.add_argument(
        "--no-retina-masks",
        dest="retina_masks",
        action="store_false",
        help="Disable full-resolution mask upsampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for rendered inference output.",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save a merged grayscale mask PNG at the original image resolution.",
    )
    parser.set_defaults(retina_masks=True)
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
    imgsz: int = 1440,
    retina_masks: bool = True,
    save_mask: bool = False,
) -> None:
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=image_path,
        task="segment",
        conf=conf,
        imgsz=imgsz,
        classes=classes,
        retina_masks=retina_masks,
        save=False,
    )

    for index, result in enumerate(results):
        output_path = os.path.join(output_dir, f"prediction_{index}.jpg")
        result.save(filename=output_path)
        print(f"Saved prediction to {output_path}")

        if save_mask:
            orig_height, orig_width = result.orig_shape
            mask_array = np.zeros((orig_height, orig_width), dtype=np.float32)
            if result.masks is not None and result.boxes is not None and len(result.boxes.cls) > 0:
                selected = np.ones(len(result.boxes.cls), dtype=bool)
                if classes is not None:
                    class_ids = result.boxes.cls.to(dtype=torch.int32).cpu().numpy()
                    selected = np.isin(class_ids, np.asarray(classes, dtype=np.int32))
                if np.any(selected):
                    masks = result.masks.data[selected].float()
                    if masks.ndim == 3 and (masks.shape[1] != orig_height or masks.shape[2] != orig_width):
                        masks = torch.nn.functional.interpolate(
                            masks.unsqueeze(1),
                            size=(orig_height, orig_width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    mask_array = masks.max(dim=0).values.cpu().numpy().astype(np.float32)

            mask_output_path = os.path.join(output_dir, f"prediction_{index}_mask.png")
            grayscale = np.clip(mask_array * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(grayscale, mode="L").save(mask_output_path, format="PNG")
            print(f"Saved full-resolution mask to {mask_output_path}")


def main() -> None:
    args = parse_args()
    label_image(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        classes=parse_classes(args.classes),
        conf=args.conf,
        imgsz=args.imgsz,
        retina_masks=args.retina_masks,
        save_mask=args.save_mask,
    )


if __name__ == "__main__":
    main()
