"""Save a small on-disk checkpoint after every stage, including before long runs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR

PROGRESS_PATH = Path(__file__).with_name("progress.json")
STAGES = {
    4: "Окружение и фиксированные зависимости",
    5: "Схема данных, разбиение и временные CV-фолды",
    6: "Baseline DummyClassifier",
    7: "LogisticRegression без балансировки",
    8: "RandomForest без балансировки",
    9: "Эксперименты class_weight",
    10: "Эксперименты SMOTE",
    11: "Эксперименты undersampling",
    12: "Подбор гиперпараметров внутри train",
    13: "Сравнение моделей и выбор порога на validation",
    14: "Замороженный пайплайн и метаданные",
    15: "Проверка загрузки и предсказания",
    16: "Передача Адаму и итоговые проверки ML",
}


def save_checkpoint(stage, status, note, checks=None):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if PROGRESS_PATH.exists():
        state = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "objective": "Finish Dias ML stages with local commits; never push.",
            "constraints": [
                "min_precision=0.85; chronological 70/15/15 after exact raw deduplication",
                "Do not use test rows for ML selection or final evaluation; hand off to Adam",
                "Use simple code in the style of Assignment 3 and 4",
                "Use configured Git identity; commit explicit files after each stage",
                "Create DIAS_ML_REPORT.md only at the end; do not commit or ignore it",
            ],
            "resume": "Read AGENTS.md, AGENTS.dias.md, this file and git log. Resume the first incomplete stage. Run python -m src.ml.cli --help for saved commands.",
            "stages": {str(key): {"name": name, "status": "pending"} for key, name in STAGES.items()},
        }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["stages"][str(stage)].update(status=status, note=note, checks=checks or [])
    temporary = PROGRESS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(PROGRESS_PATH)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=int)
    parser.add_argument("status", choices=["in_progress", "complete", "failed"])
    parser.add_argument("--note", required=True)
    parser.add_argument("--check", action="append", default=[])
    args = parser.parse_args()
    save_checkpoint(args.stage, args.status, args.note, args.check)
    print(PROGRESS_PATH.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
