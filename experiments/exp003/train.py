from pathlib import Path

from ultralytics import YOLO

# Increased augments

exp_root = Path("experiments/exp003")

exp_config = {
    "project": exp_root / "logs",
    "seed": 1212,
    "epochs": 40,
    "patience": 6,
    "batch": 16,
    "hsv_h": 0.01,  # slight color shift (LED / lighting variance)
    "hsv_s": 0.2,
    "hsv_v": 0.2,
    "scale": 0.2,  # zoom in/out (camera distance changes)
    "translate": 0.05,  # small framing shifts
    "degrees": 5,  # DO NOT rotate digits
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,  # DO NOT flip digits
    "flipud": 0.0,
    "mosaic": 0.4,  # helps detection robustness
    "mixup": 0.1,  # very light
    "copy_paste": 0.0,
    "erasing": 0.1,  # simulates broken segments
}

model = YOLO("yolo11n.pt")
results = model.train(data="./datasets/seven_seg_merge/data.yaml", **exp_config)
