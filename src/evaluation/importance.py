"""Permutation importance замороженной модели на validation.

Важность признака = насколько падает average precision, если случайно перемешать этот
признак. Считается на validation (test не трогается), модель не переобучается.
Расчёт долгий (Random Forest × 29 признаков × повторы), поэтому выполняется командой,
а приложение показывает сохранённый результат:

    python -m src.evaluation.importance            # 5 повторов
    python -m src.evaluation.importance --repeats 10
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

RESULT_CSV = "permutation_importance_validation.csv"
RESULT_JSON = "permutation_importance_validation.json"


def importance_table(pipeline, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Среднее и стандартное отклонение падения AP по повторам; сортировка по убыванию."""
    if n_repeats < 2:
        raise ValueError("нужно хотя бы 2 повтора, чтобы оценить разброс")
    result = permutation_importance(pipeline, X, np.asarray(y), scoring="average_precision",
                                    n_repeats=n_repeats, random_state=seed, n_jobs=1)
    table = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    raw = pd.DataFrame(result.importances, columns=[f"repeat_{i}" for i in range(n_repeats)])
    table = pd.concat([table, raw], axis=1)
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_selected(root=None, n_repeats: int = 5, seed: int = 42, allow_non_real: bool = False) -> tuple[pd.DataFrame, dict]:
    """Посчитать важности для выбранной модели и проверить, что данные те же, что у Диаса."""
    from src.data.api import load_xy
    from src.evaluation import sources
    from src.evaluation.contract import FrozenPipelinePredictor
    from src.ml.prediction import validate_features

    paths = sources.project_paths(root)
    predictor = FrozenPipelinePredictor.load(paths.model, paths.selection_lock)
    metadata = predictor.metadata
    if metadata.get("provenance") != "real" and not allow_non_real:
        raise ValueError("permutation importance сохраняется только для модели с provenance='real'")
    if not paths.transactions.is_file():
        raise FileNotFoundError(f"Нет {paths.transactions}. Подготовьте данные: python -m src.data.cli split.")
    X_raw, y = load_xy("validation", path=paths.transactions)
    X = validate_features(X_raw, metadata)

    baseline = float(average_precision_score(y, predictor.pipeline.predict_proba(X)[:, 1]))
    saved = metadata.get("validation_metrics", {}).get("average_precision")
    if saved is not None and abs(baseline - saved) > 1e-12:
        raise ValueError(f"AP на загруженных данных {baseline} не равна сохранённой {saved}: данные или модель другие")

    started = time.perf_counter()
    table = importance_table(predictor.pipeline, X, y, n_repeats=n_repeats, seed=seed)
    info = {
        "run_id": metadata.get("run_id"),
        "split": "validation",
        "provenance": metadata.get("provenance"),
        "scoring": "average_precision",
        "baseline_average_precision": baseline,
        "n_repeats": n_repeats,
        "seed": seed,
        "n_rows": int(len(X)),
        "n_fraud": int(np.asarray(y).sum()),
        "seconds": round(time.perf_counter() - started, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return table, info


def save(table: pd.DataFrame, info: dict, out_dir: Path) -> tuple[Path, Path]:
    if info.get("provenance") != "real":
        raise ValueError("сохраняются только результаты с provenance='real'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / RESULT_CSV, out_dir / RESULT_JSON
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def load_saved(out_dir: Path) -> tuple[pd.DataFrame, dict] | None:
    csv_path, json_path = Path(out_dir) / RESULT_CSV, Path(out_dir) / RESULT_JSON
    if not csv_path.is_file() or not json_path.is_file():
        return None
    return pd.read_csv(csv_path), json.loads(json_path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Permutation importance выбранной модели на validation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from src.evaluation.sources import project_paths

    print(f"Считаю permutation importance ({args.repeats} повторов × 29 признаков), это займёт несколько минут…")
    table, info = compute_selected(n_repeats=args.repeats, seed=args.seed)
    csv_path, _ = save(table, info, project_paths().evaluation_dir)
    print(table[["feature", "importance_mean", "importance_std"]].head(10).to_string(index=False))
    print(f"Базовая AP {info['baseline_average_precision']:.6f} совпадает с сохранённой. "
          f"Время {info['seconds']} с. Записано: {csv_path}")
    return 0


if __name__ == "__main__":
    from src.evaluation.cli_utils import run_cli

    raise SystemExit(run_cli(main))
