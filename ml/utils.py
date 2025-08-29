import json
import os
import shutil
from typing import List

import torch
from torch.utils.data import random_split

def cp_files(file_list: List[str], destination: str) -> None:
    for f in file_list:
        shutil.copyfile(f, os.path.join(destination, os.path.split(f)[-1]))

def move_train_test(
    img_dir: str,
    annotation_filename: str = "annotation.json",
    train_split: float = 0.8,
) -> None:
    """
    Copies data into train/test directories and creates train/test annotation files 

    @param img_dir: Image dir
    @param annotation_filename: filename of annotation json
    @param train_split: Percentage of images for train set
    """
    annotation_json = os.path.join(img_dir, annotation_filename)
    dataset_filenames = []
    with open(annotation_json, "r") as f:
        data = json.load(f)
        img_annotations = data["_via_img_metadata"]
    for k, image in img_annotations.items():
        if not image["regions"]:
            # Skip images that have no region annotation, i.e. not annotated images
            continue
        if not all(
            [
                region["region_attributes"]["label_type"] == "handlabeled"
                for region in image["regions"]
            ]
        ):
            # Skip images which have regions that are not handlabeled
            continue
        dataset_filenames.append({k: image})
    train_size = int(train_split * len(dataset_filenames))
    test_size = len(dataset_filenames) - train_size
    train_dataset_files, test_dataset_files = random_split(
        dataset_filenames,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset_filenames = [
        os.path.join(img_dir, values["filename"])
        for f in train_dataset_files
        for values in f.values()
    ]
    test_dataset_filenames = [
        os.path.join(img_dir, values["filename"])
        for f in test_dataset_files
        for values in f.values()
    ]
    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "test"), exist_ok=True)
    cp_files(train_dataset_filenames, os.path.join(img_dir, "train"))
    cp_files(test_dataset_filenames, os.path.join(img_dir, "test"))
    with open(os.path.join(img_dir, "train", annotation_filename), "w") as f:
        data["_via_img_metadata"] = {
            k: v for f in train_dataset_files for k, v in f.items()
        }
        json.dump(data, f)
    with open(os.path.join(img_dir, "test", annotation_filename), "w") as f:
        data["_via_img_metadata"] = {
            k: v for f in test_dataset_files for k, v in f.items()
        }
        json.dump(data, f)