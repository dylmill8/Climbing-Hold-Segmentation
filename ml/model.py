from dotenv import load_dotenv
import os
from roboflow import Roboflow
import shutil

from utils import (
    train_test_split,
    load_yolo,
)

def main():
    load_dotenv()

    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
    ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")

    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(3)
    dataset = version.download(model_format="yolov11")

    # Target directories
    target_dir = "./data/"
    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")

    # Clear old images/labels directories
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # For each split (train/valid/test), move the contents into images and labels directories
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(dataset.location, split)
        if os.path.exists(split_path):
            # Move images
            split_images = os.path.join(split_path, "images")
            if os.path.exists(split_images):
                for f in os.listdir(split_images):
                    shutil.move(os.path.join(split_images, f), images_dir)
            # Move labels
            split_labels = os.path.join(split_path, "labels")
            if os.path.exists(split_labels):
                for f in os.listdir(split_labels):
                    shutil.move(os.path.join(split_labels, f), labels_dir)

    yaml_src = os.path.join(dataset.location, "data.yaml")
    yaml_dst = os.path.join(target_dir, "data.yaml")
    if os.path.exists(yaml_src):
        if os.path.exists(yaml_dst):
            # Avoid overwriting existing data.yaml
            shutil.move(yaml_src, os.path.join(target_dir, "data_roboflow.yaml"))
            print("data.yaml exists — saved Roboflow's as data_roboflow.yaml")
        else:
            shutil.move(yaml_src, yaml_dst)

    shutil.rmtree(dataset.location)

    # TODO: remove random seed and check generalization
    # Split (moves files)
    train_test_split("./data/images", "./data/labels", seed=42)

    # WEIGHTS_SPEC = "yolo11m-seg.pt" or None   # download, move to ./ml/model
    # WEIGHTS_SPEC = r"./ml/model/runs/y11m_seg_1024/weights/last.pt"   # resume/fine-tune
    # WEIGHTS_SPEC = r"./ml/model/runs/y11m_seg_1024/weights/best.pt"   # resume/fine-tune
    WEIGHTS_SPEC = None #"./ml/model/yolo11m-seg.pt"  # use ./ml/model/yolo11m-seg.pt (auto-download if missing)
    model = load_yolo(WEIGHTS_SPEC)

    result = model.train(
        resume=False, # TODO: Make True to resume training from a checkpoint
        task="segment",
        data="./data/data.yaml",
        imgsz=1024,
        batch=-1,
        nbs=64,
        epochs=120,
        patience=30,
        seed=42,
        device=0,
        amp=True,
        workers=0,
        optimizer="AdamW",
        lr0=0.003, lrf=0.12, weight_decay=0.0005, cos_lr=True,
        warmup_epochs=5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.05, scale=0.5, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=0.12, mixup=0.0, copy_paste=0.20, erasing=0.15, close_mosaic=18,
        multi_scale=False,
        cache="ram",
        max_det=1000,
        project="./ml/model/runs/",
        name="y11m_seg_1024"
    )

    # Remove YOLO artifacts
    _ = [os.remove(p) for p in ("yolo11n.pt","yolo11m.pt","yolo11n-seg.pt","yolo11m-seg.pt") if os.path.isfile(p)]

    # TODO: pre-label, rename annotations, & upload on Roboflow
    # TODO: calculate statistics to measure model performance & visualize results

if __name__ == "__main__":
    main()