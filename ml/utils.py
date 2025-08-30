import json
import os
import shutil
from typing import List

import torch
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd

def cp_files(file_list: List[str], destination: str) -> None:
    for f in file_list:
        shutil.copyfile(f, os.path.join(destination, os.path.split(f)[-1]))

def train_test_split(
    img_dir: str,
    label_dir: str,
    train_split: float = 0.8,
    valid_split: float = 0.5,
    null_fraction: float = 0.1,
) -> None:
    """
    Splits paired images/labels into train/valid/test sets for YOLOv11 training.

    @param img_dir: Directory containing source images
    @param label_dir: Directory containing corresponding YOLO .txt labels
    @param train_split: Fraction of selected images for training (0 < train_split < 1)
    @param valid_split: Fraction of the remaining (non-train) data assigned to validation (0 <= valid_split <= 1)
    @param null_fraction: Fraction of empty-label images to include as negatives (0 <= null_fraction <= 1)
    """
    if not (0.0 < train_split < 1.0):
        raise ValueError("train_split must be between 0 and 1 (exclusive)")
    if not (0.0 <= null_fraction <= 1.0):
        raise ValueError("null_fraction must be between 0 and 1 (inclusive)")
    if not (0.0 <= valid_split <= 1.0):
        raise ValueError("valid_split must be between 0 and 1 (inclusive)")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    # Build annotated and empty-label pools (labels must exist)
    annotated_pairs = []
    empty_label_pairs = []
    for img_name in images:
        stem, _ = os.path.splitext(img_name)
        label_path = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue  # skip images without a label file
        img_path = os.path.join(img_dir, img_name)
        if os.path.getsize(label_path) > 0:
            annotated_pairs.append((img_path, label_path))
        else:
            empty_label_pairs.append((img_path, label_path))

    # Select ~null_fraction of empty-label pairs as null images
    selected_empty = []
    if empty_label_pairs and null_fraction > 0:
        gen_empty = torch.Generator().manual_seed(42)
        perm_e = torch.randperm(len(empty_label_pairs), generator=gen_empty).tolist()
        n_keep = int(round(null_fraction * len(empty_label_pairs)))
        selected_empty = [empty_label_pairs[i] for i in perm_e[:n_keep]]

    pairs = annotated_pairs + selected_empty
    if not pairs:
        print("No images with labels found. Nothing to split.")
        return

    # Deterministic shuffle and split
    n = len(pairs)
    train_size = int(train_split * n)
    gen_shuffle = torch.Generator().manual_seed(43)
    perm = torch.randperm(n, generator=gen_shuffle).tolist()
    pairs = [pairs[i] for i in perm]

    remaining = n - train_size
    valid_size = int(valid_split * remaining)

    train_pairs = pairs[:train_size]
    valid_pairs = pairs[train_size:train_size + valid_size]
    test_pairs = pairs[train_size + valid_size:]

    # Prepare output directories
    base_out = os.path.join(".", "data")
    img_train = os.path.join(base_out, "images", "train")
    img_valid = os.path.join(base_out, "images", "valid")
    img_test = os.path.join(base_out, "images", "test")
    lbl_train = os.path.join(base_out, "labels", "train")
    lbl_valid = os.path.join(base_out, "labels", "valid")
    lbl_test = os.path.join(base_out, "labels", "test")

    for d in (img_train, img_valid, img_test, lbl_train, lbl_valid, lbl_test):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Copy files
    cp_files([i for i, _ in train_pairs], img_train)
    cp_files([l for _, l in train_pairs], lbl_train)
    cp_files([i for i, _ in valid_pairs], img_valid)
    cp_files([l for _, l in valid_pairs], lbl_valid)
    cp_files([i for i, _ in test_pairs], img_test)
    cp_files([l for _, l in test_pairs], lbl_test)

    print(
        f"Split {n} files (annotated: {len(annotated_pairs)}, null: {len(selected_empty)}) "
        f"-> train: {len(train_pairs)}, valid: {len(valid_pairs)}, test: {len(test_pairs)}"
    )

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