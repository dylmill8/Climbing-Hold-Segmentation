import os
import shutil
import random
import json
import re
import tempfile
import zipfile
from typing import Iterable

import requests
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

def ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def link_or_copy_file(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        os.remove(destination)

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)

def _normalize_split_name(split: str) -> str:
    if split == "val":
        return "valid"
    return split

def _normalize_category_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized

def _flatten_segmentation_points(segmentation: list) -> list[list[float]]:
    polygons: list[list[float]] = []
    if not isinstance(segmentation, list):
        return polygons

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        polygons.append(polygon)
    return polygons

def _slugify_source_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "dataset"

def _iter_coco_subsets(source_root: str) -> list[tuple[str, str, str]]:
    root_coco_json = os.path.join(source_root, "_annotations.coco.json")
    if os.path.isfile(root_coco_json):
        return [("all", source_root, root_coco_json)]

    subsets: list[tuple[str, str, str]] = []
    for raw_split in ("train", "valid", "test"):
        split = _normalize_split_name(raw_split)
        split_source = os.path.join(source_root, split)
        coco_json_path = os.path.join(split_source, "_annotations.coco.json")
        if not os.path.isfile(coco_json_path):
            raise FileNotFoundError(f"Missing COCO annotation file: {coco_json_path}")
        subsets.append((split, split_source, coco_json_path))
    return subsets

def convert_coco_segmentations_to_yolo(
    coco_json_path: str,
    image_root: str,
    output_images_dir: str,
    output_labels_dir: str,
    category_names: Iterable[str],
    file_name_prefix: str = "",
) -> dict[str, int]:
    with open(coco_json_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    requested_categories = [_normalize_category_name(name) for name in category_names]
    requested_set = set(requested_categories)

    category_lookup: dict[int, str] = {
        int(category["id"]): _normalize_category_name(category["name"])
        for category in coco.get("categories", [])
    }
    category_to_index = {
        name: idx for idx, name in enumerate(requested_categories)
    }

    image_lookup = {
        int(image["id"]): image
        for image in coco.get("images", [])
    }
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        category_name = category_lookup.get(int(annotation["category_id"]))
        if category_name not in requested_set:
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    stats = {
        "images_total": len(image_lookup),
        "images_written": 0,
        "labels_written": 0,
        "polygons_written": 0,
        "duplicate_labels_removed": 0,
    }
    for name in requested_categories:
        stats[f"class_{name}"] = 0

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for image_id, image in tqdm(image_lookup.items(), desc=f"Converting {os.path.basename(coco_json_path)}", unit="img"):
        file_name = image["file_name"]
        output_file_name = f"{file_name_prefix}{file_name}"
        width = float(image["width"])
        height = float(image["height"])
        source_image = os.path.join(image_root, file_name)
        target_image = os.path.join(output_images_dir, output_file_name)
        target_label = os.path.join(output_labels_dir, os.path.splitext(output_file_name)[0] + ".txt")

        if not os.path.isfile(source_image):
            print(f"Warning: missing image referenced by COCO file: {source_image}")
            continue

        link_or_copy_file(source_image, target_image)
        stats["images_written"] += 1

        label_lines: list[str] = []
        seen_lines: set[str] = set()
        for annotation in annotations_by_image.get(image_id, []):
            category_name = category_lookup[int(annotation["category_id"])]
            polygons = _flatten_segmentation_points(annotation.get("segmentation", []))
            if not polygons:
                continue

            class_index = category_to_index[category_name]
            for polygon in polygons:
                normalized_points: list[str] = []
                for idx in range(0, len(polygon), 2):
                    x = min(max(float(polygon[idx]) / width, 0.0), 1.0)
                    y = min(max(float(polygon[idx + 1]) / height, 0.0), 1.0)
                    normalized_points.extend((f"{x:.6f}", f"{y:.6f}"))
                line = f"{class_index} " + " ".join(normalized_points)
                if line in seen_lines:
                    stats["duplicate_labels_removed"] += 1
                    continue
                seen_lines.add(line)
                label_lines.append(line)
                stats["polygons_written"] += 1
                stats[f"class_{category_name}"] += 1

        with open(target_label, "w", encoding="utf-8") as fh:
            fh.write("\n".join(label_lines))
        stats["labels_written"] += 1

    return stats

def write_yolo_data_yaml(output_root: str, class_names: Iterable[str]) -> str:
    normalized_classes = [_normalize_category_name(name) for name in class_names]
    yaml_path = os.path.join(output_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {output_root.replace(os.sep, '/')}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/valid\n")
        fh.write("test: images/test\n")
        fh.write(f"nc: {len(normalized_classes)}\n")
        names_str = ", ".join(f"'{name}'" for name in normalized_classes)
        fh.write(f"names: [{names_str}]\n")
    return yaml_path

def prepare_yolo_dataset_from_coco(
    source_root: str,
    output_root: str,
    class_names: Iterable[str],
) -> str:
    normalized_classes = [_normalize_category_name(name) for name in class_names]
    ensure_clean_dir(output_root)

    images_root = os.path.join(output_root, "images")
    labels_root = os.path.join(output_root, "labels")

    all_stats: dict[str, dict[str, int]] = {}
    for split, split_source, coco_json_path in _iter_coco_subsets(source_root):
        output_images_dir = os.path.join(images_root, split)
        output_labels_dir = os.path.join(labels_root, split)
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        all_stats[split] = convert_coco_segmentations_to_yolo(
            coco_json_path=coco_json_path,
            image_root=split_source,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            category_names=normalized_classes,
        )

    yaml_path = write_yolo_data_yaml(output_root, normalized_classes)

    print("Prepared YOLO dataset from COCO export:")
    for split, split_stats in all_stats.items():
        print(
            f"  {split}: images={split_stats['images_written']}, "
            f"labels={split_stats['labels_written']}, polygons={split_stats['polygons_written']}, "
            f"duplicates_removed={split_stats['duplicate_labels_removed']}"
        )
        for class_name in normalized_classes:
            print(f"    {class_name}: {split_stats[f'class_{class_name}']}")

    return yaml_path

def stage_coco_sources_for_resplit(
    source_roots: Iterable[str],
    output_root: str,
    class_names: Iterable[str],
    source_names: Iterable[str] | None = None,
) -> str:
    normalized_classes = [_normalize_category_name(name) for name in class_names]
    ensure_clean_dir(output_root)

    images_root = os.path.join(output_root, "images")
    labels_root = os.path.join(output_root, "labels")
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    resolved_source_names = list(source_names) if source_names is not None else [
        os.path.basename(os.path.normpath(source_root)) for source_root in source_roots
    ]

    all_stats: dict[str, dict[str, int]] = {}
    for source_root, source_name in zip(source_roots, resolved_source_names):
        prefix = f"{_slugify_source_name(source_name)}__"
        source_stats = {
            "images_total": 0,
            "images_written": 0,
            "labels_written": 0,
            "polygons_written": 0,
            "duplicate_labels_removed": 0,
        }
        for class_name in normalized_classes:
            source_stats[f"class_{class_name}"] = 0

        for _, split_source, coco_json_path in _iter_coco_subsets(source_root):
            split_stats = convert_coco_segmentations_to_yolo(
                coco_json_path=coco_json_path,
                image_root=split_source,
                output_images_dir=images_root,
                output_labels_dir=labels_root,
                category_names=normalized_classes,
                file_name_prefix=prefix,
            )

            for key, value in split_stats.items():
                source_stats[key] = source_stats.get(key, 0) + value

        all_stats[source_name] = source_stats

    print("Staged COCO exports for local split:")
    for source_name, stats in all_stats.items():
        print(
            f"  {source_name}: images={stats['images_written']}, "
            f"labels={stats['labels_written']}, polygons={stats['polygons_written']}, "
            f"duplicates_removed={stats['duplicate_labels_removed']}"
        )
        for class_name in normalized_classes:
            print(f"    {class_name}: {stats[f'class_{class_name}']}")

    return output_root

def prepare_yolo_dataset_from_coco_sources(
    source_roots: Iterable[str],
    output_root: str,
    class_names: Iterable[str],
    source_names: Iterable[str] | None = None,
    train_split: float = 0.8,
    valid_split: float = 0.5,
    null_fraction: float = 0.0,
    seed: int | None = None,
) -> str:
    stage_coco_sources_for_resplit(
        source_roots=source_roots,
        output_root=output_root,
        class_names=class_names,
        source_names=source_names,
    )
    train_test_split(
        img_dir=os.path.join(output_root, "images"),
        label_dir=os.path.join(output_root, "labels"),
        train_split=train_split,
        valid_split=valid_split,
        null_fraction=null_fraction,
        seed=seed,
    )
    return write_yolo_data_yaml(output_root, class_names)

def train_test_split(
    img_dir: str,
    label_dir: str,
    train_split: float = 0.8,
    valid_split: float = 0.5,
    null_fraction: float = 0.1,
    seed: int | None = None,
) -> None:
    """
    Moves paired images/labels into train/valid/test splits for YOLOv11 training.

    @param img_dir: Directory containing input images.
    @param label_dir: Directory containing label files.
    @param train_split: Fraction of data to use for training.
    @param valid_split: Fraction of data to use for validation.
    @param null_fraction: Fraction of empty-label images to use as nulls.
    @param seed: Random seed for reproducibility.
    """
    # Validate inputs
    if not (0.0 < train_split < 1.0):
        raise ValueError("train_split must be in (0,1)")
    if not (0.0 <= valid_split <= 1.0):
        raise ValueError("valid_split must be in [0,1]")
    if not (0.0 <= null_fraction <= 1.0):
        raise ValueError("null_fraction must be in [0,1]")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Labels dir not found: {label_dir}")

    if seed is not None:
        random.seed(seed)

    # Prepare split dirs (clear old split folders only)
    def reset_dir(p: str):
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    img_splits = {s: os.path.join(img_dir, s) for s in ("train", "valid", "test")}
    lbl_splits = {s: os.path.join(label_dir, s) for s in ("train", "valid", "test")}
    for s in ("train", "valid", "test"):
        reset_dir(img_splits[s])
        reset_dir(lbl_splits[s])

    # Collect images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    annotated_pairs = []   # (img_path, lbl_path)
    empty_label_pairs = [] # (img_path, lbl_path)
    for img_name in tqdm(images, desc="Scanning images/labels", unit="img"):
        stem, _ = os.path.splitext(img_name)
        ip = os.path.join(img_dir, img_name)
        lp = os.path.join(label_dir, f"{stem}.txt")
        if not os.path.isfile(lp):
            # Skip images without a corresponding label file
            print(f"Warning: missing label for {ip}; skipping")
            continue
        if os.path.getsize(lp) > 0:
            annotated_pairs.append((ip, lp))
        else:
            empty_label_pairs.append((ip, lp))

    # Select ~null_fraction of empty-label pairs as null images
    selected_empty = []
    if empty_label_pairs and null_fraction > 0:
        k = int(round(null_fraction * len(empty_label_pairs)))
        k = max(0, min(k, len(empty_label_pairs)))
        if k > 0:
            selected_empty = random.sample(empty_label_pairs, k)

    pairs = annotated_pairs + selected_empty
    if not pairs:
        raise RuntimeError("No data found to split: no annotated images and no selected nulls.")

    # Shuffle and split
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(train_split * n)
    rem = n - n_train
    n_valid = int(round(valid_split * rem))
    n_test = rem - n_valid

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test":  pairs[n_train + n_valid:],
    }

    # Move files (no label creation)
    for split, items in splits.items():
        for ip, lp in tqdm(items, desc=f"Moving {split}", unit="pair"):
            dst_img = os.path.join(img_splits[split], os.path.basename(ip))
            if os.path.abspath(ip) != os.path.abspath(dst_img):
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                shutil.move(ip, dst_img)

            dst_lbl = os.path.join(lbl_splits[split], os.path.basename(lp))
            if os.path.abspath(lp) != os.path.abspath(dst_lbl):
                if os.path.exists(dst_lbl):
                    os.remove(dst_lbl)
                shutil.move(lp, dst_lbl)

    print(f"Moved: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

def filter_annotations(
        input_csv: str, 
        output_csv: str = "annotation-filtered.csv",
        location: str = "./data/"
) -> None:
    """
    Filters out empty annotations from the input CSV and saves the result to the output CSV.

    @param input_csv: Path to the input CSV file
    @param output_csv: Path to the output CSV file
    @param location: Directory to move the output CSV file to
    """
    df = pd.read_csv(input_csv)
    filtered = df[df['region_shape_attributes'].notna() & (df['region_shape_attributes'] != '{}')]
    filtered.to_csv(output_csv, index=False)
    shutil.move(output_csv, os.path.join(location, output_csv))

def load_yolo(weights_spec: str | None = None, model_dir: str = "./ml/model") -> YOLO:
    """
    Loads a YOLO model from the specified weights file or downloads the default if none provided.

    @param weights_spec: Path to the weights file or None
    @param model_dir: Directory to store the model weights
    @return: Loaded YOLO model
    """
    os.makedirs(model_dir, exist_ok=True)

    def load_or_download(base: str) -> YOLO:
        target = os.path.join(model_dir, os.path.basename(base))
        if os.path.isfile(target):
            return YOLO(target)

        if os.path.isfile(base):
            return YOLO(base)

        # Trigger a download to CWD, then move into model_dir for reuse.
        y = YOLO(base)
        cwd_file = os.path.abspath(os.path.basename(base))
        if os.path.isfile(cwd_file):
            shutil.move(cwd_file, target)
            return YOLO(target)
        return y

    if not weights_spec:
        return load_or_download("yolo11m-seg.pt")

    # Explicit local path provided.
    if os.path.isfile(weights_spec):
        return YOLO(weights_spec)

    # Named published checkpoint, e.g. yolo26s-seg.pt.
    if os.path.basename(weights_spec) == weights_spec:
        return load_or_download(weights_spec)

    raise FileNotFoundError(f"Specified weights not found: {weights_spec}")

def download_roboflow_coco_dataset(
    *,
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    download_root: str,
    dataset_name: str,
    model_format: str = "coco-segmentation",
    overwrite: bool = False,
) -> str:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow download requested, but the 'roboflow' package is not installed."
        ) from exc

    target_dir = os.path.join(download_root, _slugify_source_name(dataset_name))
    marker_path = os.path.join(target_dir, ".download_complete.json")

    if overwrite and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    if os.path.isdir(target_dir) and os.path.isfile(marker_path):
        return target_dir

    os.makedirs(download_root, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(model_format=model_format)
    downloaded_location = dataset.location

    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(downloaded_location, target_dir)

    with open(marker_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "workspace": workspace,
                "project": project,
                "version": version,
                "model_format": model_format,
            },
            fh,
            indent=2,
        )

    return target_dir

def download_roboflow_coco_dataset_from_url(
    *,
    download_url: str,
    download_root: str,
    dataset_name: str,
    overwrite: bool = False,
    timeout_seconds: int = 300,
) -> str:
    target_dir = os.path.join(download_root, _slugify_source_name(dataset_name))
    marker_path = os.path.join(target_dir, ".download_complete.json")

    if overwrite and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    if os.path.isdir(target_dir) and os.path.isfile(marker_path):
        return target_dir

    os.makedirs(download_root, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=download_root) as temp_dir:
        zip_path = os.path.join(temp_dir, "dataset.zip")
        extract_dir = os.path.join(temp_dir, "extracted")

        with requests.get(download_url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        extracted_entries = [
            os.path.join(extract_dir, entry)
            for entry in os.listdir(extract_dir)
        ]
        extracted_dirs = [entry for entry in extracted_entries if os.path.isdir(entry)]
        dataset_dir = extracted_dirs[0] if len(extracted_dirs) == 1 else extract_dir

        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(dataset_dir, target_dir)

    with open(marker_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "download_url": download_url,
                "source": "roboflow_share_url",
            },
            fh,
            indent=2,
        )

    return target_dir
