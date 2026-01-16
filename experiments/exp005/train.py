from pathlib import Path

from ultralytics import YOLO

# Class weights and transforms

exp_root = Path("experiments/exp005")

exp_config = {
    "project": exp_root / "logs",
    "seed": 1212,
    "epochs": 40,
    "patience": 6,
    "batch": 16,
    "mosaic": 0.3,
    "hsv_h": 0.01,
    "hsv_s": 0.2,
    "hsv_v": 0.2,
}

model = YOLO("yolo11n.pt")
results = model.train(data="./datasets/seven_seg_merge/data_wh.yaml", **exp_config)
