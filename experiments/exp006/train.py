from pathlib import Path

from ultralytics import YOLO

# Bigger model

exp_root = Path("experiments/exp006")

exp_config = {
    "project": exp_root / "logs",
    "seed": 1212,
    "epochs": 40,
    "patience": 6,
    "batch": 16,
}

model = YOLO("yolo11s.pt")
results = model.train(data="./datasets/seven_seg_merge/data_wh.yaml", **exp_config)
