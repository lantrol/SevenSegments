from pathlib import Path

from ultralytics import YOLO

exp_root = Path("experiments/exp001")

exp_config = {
    "project": exp_root / "logs",
    "seed": 1212,
    "epochs": 25,
    "patience": 5,
    "batch": 16,
}

model = YOLO("yolo11n.pt")
results = model.train(data="./datasets/seven_seg_2/data.yaml", **exp_config)
