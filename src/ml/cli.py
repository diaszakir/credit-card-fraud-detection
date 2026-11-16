"""Commands for the ML stages; every fit runs in its own process."""

import argparse
import json
import os
import subprocess
import sys

from src.config import EXPERIMENTS_DIR, ROOT_DIR


def run_one(model, strategy="none", tune=False, force=False, stage=None):
    command = [sys.executable, "-m", "src.ml.experiment", "--model", model, "--strategy", strategy]
    if tune:
        command.append("--tune")
    if force:
        command.append("--force")
    if stage is not None:
        command.extend(["--stage", str(stage)])
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    subprocess.run(command, cwd=ROOT_DIR, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one fixed or CV-tuned train/validation experiment")
    run.add_argument("--model", choices=["dummy", "logistic_regression", "random_forest"], required=True)
    run.add_argument("--strategy", choices=["none", "class_weight", "smote", "undersampling"], default="none")
    run.add_argument("--tune", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--stage", type=int)
    commands.add_parser("tune", help="CV tuning for both learned models and all four strategies")
    commands.add_parser("compare", help="export validation results and choose a feasible model")
    args = parser.parse_args()
    if args.command == "run":
        run_one(args.model, args.strategy, args.tune, args.force, args.stage)
    elif args.command == "tune":
        for model in ("logistic_regression", "random_forest"):
            for strategy in ("none", "class_weight", "smote", "undersampling"):
                run_one(model, strategy, tune=True, stage=12)
    else:
        from src.data.io_utils import read_json
        from src.ml.comparison import choose_best, export_results, load_results
        from src.ml.experiment import CONFIG_PATH, write_atomic_json
        results = load_results(EXPERIMENTS_DIR / "results")
        paths = export_results(results, EXPERIMENTS_DIR)
        best = choose_best(results, read_json(CONFIG_PATH))
        write_atomic_json(best, EXPERIMENTS_DIR / "selected_run.json")
        print(json.dumps({"selected_run": best["run_id"], "threshold": best["threshold"], "files": paths}, default=str))


if __name__ == "__main__":
    main()
