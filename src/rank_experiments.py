import os
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

EXPERIMENTS_DIR = "experiments"
RESULTS_REL_PATH = os.path.join("logs", "train", "results.csv")
MODEL_REL_PATH = os.path.join("logs", "train", "weights")
METRIC_TO_OPTIMIZE = "metrics/mAP50-95(B)"


def find_experiments(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def load_best_metric(results_csv, metric):
    df = pd.read_csv(results_csv)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    # Use best epoch instead of last epoch
    return df[metric].max()


def main():
    experiments = find_experiments(EXPERIMENTS_DIR)
    rankings = []

    for exp_path in experiments:
        exp_name = os.path.basename(exp_path)
        results_csv = os.path.join(exp_path, RESULTS_REL_PATH)

        if not os.path.isfile(results_csv):
            print(f"[WARN] Missing results.csv for {exp_name}")
            continue

        try:
            best_score = load_best_metric(results_csv, METRIC_TO_OPTIMIZE)
            rankings.append({"experiment": exp_name, METRIC_TO_OPTIMIZE: best_score})
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")

    if not rankings:
        print("No valid experiments found.")
        return

    # Rank from best to worst
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values(
        by=METRIC_TO_OPTIMIZE, ascending=False
    ).reset_index(drop=True)

    print("\n=== Experiment Ranking (Best to Worst) ===")
    for idx, row in rankings_df.iterrows():
        print(
            f"{idx + 1:02d}. {row['experiment']} | "
            f"{METRIC_TO_OPTIMIZE} = {row[METRIC_TO_OPTIMIZE]:.4f}"
        )

    best = rankings_df.iloc[0]
    print("\nBest Experiment")
    print(
        f"{best['experiment']} with "
        f"{METRIC_TO_OPTIMIZE} = {best[METRIC_TO_OPTIMIZE]:.4f}"
    )

    model_path = Path(EXPERIMENTS_DIR) / best["experiment"] / MODEL_REL_PATH
    model = YOLO(model_path / "best.pt")

    os.makedirs("models", exist_ok=True)

    print("\n=== Exporting Embedded Models ===")

    model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        simplify=True,
        half=True,
        dynamic=False,
        device="cpu",
        name="onnx_fp16",
    )

    shutil.move(model_path / "best.onnx", "models/")


if __name__ == "__main__":
    main()
