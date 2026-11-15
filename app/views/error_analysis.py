"""Error Analysis: исходы TP/FP/FN/TN выбранной модели на validation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views.common import fmt, paths, show_missing
from src.evaluation import sources

OUTCOME_NAMES = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}


def outcomes(table: pd.DataFrame) -> pd.Series:
    """Тип исхода каждой строки по y_true и y_pred."""
    pairs = zip(table["y_true"].astype(int), table["y_pred"].astype(int))
    return pd.Series([OUTCOME_NAMES[pair] for pair in pairs], index=table.index, name="outcome")


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

    table = predictions.data.assign(outcome=lambda t: outcomes(t))
    counts = table["outcome"].value_counts().reindex(["TP", "FP", "FN", "TN"], fill_value=0)
    columns = st.columns(4)
    for column, name in zip(columns, counts.index):
        column.metric(name, int(counts[name]))
    st.caption(f"Validation, запуск {metadata.data.get('run_id')}, порог {fmt(metadata.data.get('threshold'), 6)}. "
               f"Источник: {predictions.path.name}.")

    kind = st.radio("Ошибки", ["FN", "FP"], horizontal=True, key="error_kind",
                    format_func=lambda k: {"FN": "Пропущенный fraud (FN)", "FP": "Ложная тревога (FP)"}[k])
    errors = table[table["outcome"] == kind].sort_values("score", ascending=(kind == "FN"))
    if errors.empty:
        st.write("Строк этого типа нет.")
    else:
        st.dataframe(errors[["transaction_id", "score", "y_true", "y_pred"]], hide_index=True)
    st.info("Сравнение признаков FP/FN с TP/TN, SQL JOIN с transactions и permutation importance — этап 4.")
