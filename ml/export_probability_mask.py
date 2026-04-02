import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a merged probability mask from a YOLO segmentation model.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output grayscale PNG path.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights.")
    parser.add_argument("--class-index", type=int, default=0, help="Class index to keep.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1440, help="Inference image size.")
    parser.add_argument(
        "--retina-masks",
        dest="retina_masks",
        action="store_true",
        help="Upsample masks back to image resolution before exporting.",
    )
    parser.add_argument(
        "--no-retina-masks",
        dest="retina_masks",
        action="store_false",
        help="Disable full-resolution mask upsampling.",
    )
    parser.set_defaults(retina_masks=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=args.image,
        task="segment",
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[args.class_index],
        retina_masks=args.retina_masks,
        save=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO inference returned no results.")

    result = results[0]
    orig_height, orig_width = result.orig_shape
    mask_array = np.zeros((orig_height, orig_width), dtype=np.float32)

    if result.masks is not None and result.boxes is not None and len(result.boxes.cls) > 0:
        selected = result.boxes.cls.to(dtype=torch.int32).cpu().numpy() == int(args.class_index)
        if np.any(selected):
            masks = result.masks.data.cpu().numpy()[selected]
            if masks.size > 0:
                mask_array = masks.max(axis=0).astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grayscale = np.clip(mask_array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(grayscale, mode="L").save(output_path, format="PNG")

    payload = {
        "width": int(mask_array.shape[1]),
        "height": int(mask_array.shape[0]),
        "model_version": Path(args.model).name,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
