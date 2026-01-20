from pathlib import Path

from ultralytics import YOLO

# Older model

exp_root = Path("experiments/exp007")

exp_config = {
    "project": exp_root / "logs",
    "seed": 1212,
    "epochs": 40,
    "patience": 6,
    "batch": 16,
}

model = YOLO("yolov8n.pt")
results = model.train(data="./datasets/seven_seg_merge/data_wh.yaml", **exp_config)
