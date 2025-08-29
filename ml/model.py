from utils import (
    move_train_test,
)

#from ultralytics import YOLO

#model = YOLO("yolov12m.pt")

move_train_test(".\\data\\bh", "bh-annotation.json", 0.8)