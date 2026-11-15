"""Draw validation-only figures from saved experiment results, without refitting."""

import os
from pathlib import Path

from src.config import EXPERIMENTS_DIR, ROOT_DIR

os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".venv" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.io_utils import read_json
from src.ml.artifacts import file_sha256
from src.ml.comparison import choose_best, load_results
from src.ml.experiment import CONFIG_PATH
from src.ml.metrics import precision_recall_points

SHORT_NAMES = {"dummy": "Dummy", "logistic_regression": "Logistic", "random_forest": "Random Forest"}


def draw_figures():
    results = load_results(EXPERIMENTS_DIR / "results")
    config = read_json(CONFIG_PATH)
    selected = choose_best(results, config)
    directory = EXPERIMENTS_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    ordered = sorted(results, key=lambda run: (run["model_name"], run["imbalance_strategy"], run["tuning"]))
    labels = [f"{SHORT_NAMES[run['model_name']]} / {run['imbalance_strategy']} / {run['tuning']}" for run in ordered]
    fig, ax = plt.subplots(figsize=(11, max(5, len(ordered) * 0.35)))
    ax.barh(labels, [run["validation_metrics"]["average_precision"] for run in ordered], color="#2563a6")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average precision (AP), validation")
    ax.set_title("Same chronological split; fixed and CV-tuned configurations")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(directory / "validation_average_precision.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    selected_points = None
    for model in config["models"]:
        candidates = [run for run in results if run["model_name"] == model]
        if not candidates:
            continue
        feasible = [run for run in candidates if run["status"] == "ok"]
        representative = choose_best(feasible, config) if feasible else max(candidates, key=lambda run: run["validation_metrics"]["average_precision"])
        scores_path = ROOT_DIR / representative["scores_file"]
        if file_sha256(scores_path) != representative.get("scores_sha256"):
            raise ValueError(f"{representative['run_id']}: validation cache SHA-256 mismatch")
        frame = pd.read_parquet(scores_path)
        curve = precision_recall_points(frame["y_true"], frame["score"])
        metrics = representative["validation_metrics"]
        label = (f"{SHORT_NAMES[model]} / {representative['imbalance_strategy']} / {representative['tuning']}\n"
                 f"AP={metrics['average_precision']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
        ax.plot(curve["recall"], curve["precision"], label=label)
        if representative["run_id"] == selected["run_id"]:
            selected_points = curve
    ax.axhline(config["min_precision"], color="black", linestyle="--", linewidth=1, label="Required precision = 0.85")
    ax.scatter([selected["validation_metrics"]["recall"]], [selected["validation_metrics"]["precision"]], color="black", zorder=5, label="Selected threshold")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Validation precision–recall curves")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(directory / "validation_precision_recall.png", dpi=150)
    plt.close(fig)
    if selected_points is not None:
        pd.DataFrame({"precision": selected_points["precision"], "recall": selected_points["recall"],
                      "threshold": selected_points["thresholds"] + [np.nan]}).to_csv(directory / "selected_validation_pr_curve.csv", index=False)

    metrics = selected["validation_metrics"]
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"], xlabel="Predicted", ylabel="Actual", title="Selected threshold: validation only")
    fig.tight_layout()
    fig.savefig(directory / "validation_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return sorted(str(path.relative_to(ROOT_DIR)) for path in directory.iterdir())


if __name__ == "__main__":
    for path in draw_figures():
        print(path)
