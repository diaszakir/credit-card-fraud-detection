"""Error Analysis: полные метрики и исходы TP/FP/FN/TN выбранной модели на validation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views.common import fmt, paths, show_missing
from src.evaluation import sources
from src.evaluation.metrics import compare_with_saved, evaluate_predictions_table

OUTCOME_NAMES = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}


def outcomes(table: pd.DataFrame) -> pd.Series:
    """Тип исхода каждой строки по y_true и y_pred."""
    pairs = zip(table["y_true"].astype(int), table["y_pred"].astype(int))
    return pd.Series([OUTCOME_NAMES[pair] for pair in pairs], index=table.index, name="outcome")


def confusion_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
        index=pd.Index(["Факт: обычная", "Факт: fraud"], name=""),
        columns=["Прогноз: обычная", "Прогноз: fraud"],
    )


def show_metrics(result: dict, saved: dict) -> None:
    st.subheader("Метрики на validation")
    first = st.columns(4)
    first[0].metric("AP", fmt(result["average_precision"]))
    first[1].metric("PR-AUC", fmt(result["pr_auc"]))
    first[2].metric("ROC-AUC", fmt(result["roc_auc"]))
    first[3].metric("F1", fmt(result.get("f1")))
    second = st.columns(4)
    for column, key, label in zip(second, ["precision", "recall", "specificity", "false_positive_rate"],
                                  ["Precision", "Recall", "Specificity", "FPR"]):
        ci = result.get(f"{key}_ci")
        column.metric(label, fmt(result.get(key)), f"95% ДИ {ci[0]:.3f}–{ci[1]:.3f}" if ci else None, delta_color="off")
    st.dataframe(confusion_table(result))
    problems = compare_with_saved(result, saved)
    if problems:
        st.warning("Метрики не совпали с сохранёнными в модели: " + "; ".join(problems))
    else:
        st.caption(f"{result['n']} транзакций, из них fraud {result['n_positive']}. AP, PR-AUC, precision, recall и "
                   "матрица ошибок совпадают с сохранёнными результатами Диаса. ROC-AUC при 0.13% fraud выглядит "
                   "оптимистично, поэтому основная метрика — AP.")


def render() -> None:
    st.header("Error Analysis")
    project = paths()
    metadata = sources.load_model_metadata(project)
    if not metadata.available:
        show_missing(metadata, "warning")
        return
    predictions = sources.load_validation_predictions(project, metadata.data)
    if not predictions.available:
        show_missing(predictions)
        st.caption("Раздел заработает после восстановления модели на этом компьютере; результаты не подставляются.")
        return

    try:
        result = evaluate_predictions_table(predictions.data, metadata.data, split="validation")
    except ValueError as exc:
        st.error(f"Прогнозы не прошли проверку: {exc}")
        return
    show_metrics(result, metadata.data.get("validation_metrics", {}))

    st.subheader("Ошибки модели")
    table = predictions.data.assign(outcome=lambda t: outcomes(t))
    counts = table["outcome"].value_counts().reindex(["TP", "FP", "FN", "TN"], fill_value=0)
    columns = st.columns(4)
    for column, name in zip(columns, counts.index):
        column.metric(name, int(counts[name]))
    st.caption(f"Запуск {metadata.data.get('run_id')}, порог {fmt(metadata.data.get('threshold'), 6)}. "
               f"Источник: {predictions.path.name}.")

    kind = st.radio("Ошибки", ["FN", "FP"], horizontal=True, key="error_kind",
                    format_func=lambda k: {"FN": "Пропущенный fraud (FN)", "FP": "Ложная тревога (FP)"}[k])
    errors = table[table["outcome"] == kind].sort_values("score", ascending=(kind == "FN"))
    if errors.empty:
        st.write("Строк этого типа нет.")
    else:
        st.dataframe(errors[["transaction_id", "score", "y_true", "y_pred"]], hide_index=True)
    st.info("Сравнение признаков FP/FN с TP/TN, SQL JOIN с transactions и permutation importance — этап 4.")
