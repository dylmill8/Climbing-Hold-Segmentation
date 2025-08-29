import os

from utils import (
    move_train_test,
)

from ultralytics import YOLO

move_train_test(".\\data\\bh")

model = YOLO("yolov11m-seg.pt")