from dotenv import load_dotenv
import os

from utils import (
    move_train_test,
)

from ultralytics import YOLO

from roboflow import Roboflow
import shutil

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")

if not ROBOFLOW_API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY is not set")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(3)
dataset = version.download(model_format="yolov8-obb")

# Target directory
target_dir = ".\\data\\"
images_dir = os.path.join(target_dir, "images")
labels_dir = os.path.join(target_dir, "labels")

# Always clear out old images/labels folders first
if os.path.exists(images_dir):
    shutil.rmtree(images_dir)
if os.path.exists(labels_dir):
    shutil.rmtree(labels_dir)

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# For each split (train/valid/test), move the contents into images/ and labels/
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

# Copy the data.yaml if you want to keep class names etc.
yaml_src = os.path.join(dataset.location, "data.yaml")
yaml_dst = os.path.join(target_dir, "data.yaml")

if os.path.exists(yaml_src):
    if os.path.exists(yaml_dst):
        # Don't overwrite — save Roboflow's version separately
        shutil.move(yaml_src, os.path.join(target_dir, "data_roboflow.yaml"))
        print("⚠️  data.yaml already exists — preserved your file, saved Roboflow's as data_roboflow.yaml")
    else:
        shutil.move(yaml_src, yaml_dst)

# Finally, remove the original parent folder
shutil.rmtree(dataset.location)

move_train_test("./data/bh")

model = YOLO("./data/model/yolo11m-seg.pt")