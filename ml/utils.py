import os
import shutil
import random
from typing import List

from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

def cp_files(file_list: List[str], destination: str) -> None:
    os.makedirs(destination, exist_ok=True)
    for f in tqdm(file_list, desc="Copying files", unit="file"):
        shutil.copyfile(f, os.path.join(destination, os.path.split(f)[-1]))

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

    if not weights_spec:
        base = "yolo11m-seg.pt"
        target = os.path.join(model_dir, base)
        if os.path.isfile(target):
            return YOLO(target)

        # Trigger a download to CWD, then move into model_dir
        y = YOLO(base)  # downloads as ./yolo11m-seg.pt in CWD
        cwd_file = os.path.abspath(base)
        if os.path.isfile(cwd_file):
            shutil.move(cwd_file, target)  # single pinned copy under model_dir
        else:
            # Fallback: persist the in-memory model to target
            y.save(target)
        return YOLO(target)

    # Path provided: load only if it exists
    if os.path.isfile(weights_spec):
        return YOLO(weights_spec)

    raise FileNotFoundError(f"Specified weights not found: {weights_spec}")