"""Fraud Prediction: проверка схемы входа и прогноз сохранённой моделью."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views.common import fmt, model_label, predictor_status
from app.views.error_analysis import outcomes
from app.views import prediction_lab
from src.config import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from src.evaluation.contract import STUB_ENV_VAR, PredictorStatus, check_input_schema, split_input

MAX_ROWS = 50_000


def show_status(status: PredictorStatus) -> None:
    if status.mode == "frozen":
        info = status.predictor.info
        st.success(f"Модель: {model_label(info.model_name, info.imbalance_strategy)}. "
                   f"Порог {fmt(info.threshold, 6)} (выбран на validation, при прогнозе не меняется).")
    elif status.mode == "stub":
        st.error("ЗАГЛУШКА. score — хеш значений строки, а не прогноз модели. Режим только для разработки интерфейса. "
                 + status.message)
    else:
        st.warning(f"Модель недоступна: {status.message}")
        st.caption("Проверка схемы CSV работает и без модели. Восстановление модели: "
                   "python -m src.ml.reproduce, затем python -m src.ml.handoff verify. "
                   f"Для разработки интерфейса без модели: {STUB_ENV_VAR}=1.")


def read_csv(uploaded) -> pd.DataFrame | None:
    try:
        uploaded.seek(0)
        frame = pd.read_csv(uploaded)
    except (OSError, ValueError, UnicodeError) as exc:
        st.error(f"Не удалось прочитать CSV: {exc}. Нужен разделитель-запятая и строка заголовков.")
        return None
    if len(frame) > MAX_ROWS:
        st.error(f"В файле {len(frame)} строк; за один раз можно проверить не больше {MAX_ROWS}.")
        return None
    return frame


def template_csv() -> bytes:
    return pd.DataFrame(columns=[ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]).to_csv(index=False).encode("utf-8")


def manual_frame() -> pd.DataFrame:
    """Одна строка для ручного ввода; значения задаёт пользователь."""
    start = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
    return st.data_editor(start, hide_index=True, key="manual_row", num_rows="fixed")


def show_result(frame: pd.DataFrame, status: PredictorStatus) -> None:
    report = check_input_schema(frame, FEATURE_COLUMNS)
    if not report.ok:
        st.error("Схема входных данных не прошла проверку:")
        for message in report.messages():
            st.write(f"- {message}")
        return
    st.success(f"Схема корректна: {report.n_rows} строк, 29 признаков"
               + (", есть transaction_id" if report.has_id else "")
               + (", есть Class" if report.has_label else "") + ".")
    if status.predictor is None:
        return
    X, y = split_input(frame, FEATURE_COLUMNS)
    try:
        predictions = status.predictor.predict(X)
    except ValueError as exc:
        st.error(f"Прогноз не выполнен: {exc}")
        return
    result = predictions.copy()
    result["Класс"] = result["y_pred"].map({0: "Обычная", 1: "Fraud"})
    if y is not None:
        result["y_true"] = y.to_numpy()
        result["Исход"] = outcomes(result).to_numpy()
    result.index.name = ID_COLUMN if report.has_id else "строка"
    st.dataframe(result.reset_index(), hide_index=True)
    fraud = int(result["y_pred"].sum())
    st.caption(f"Fraud: {fraud} из {len(result)}. Fraud означает score ≥ порога."
               + (" Исходы TP/FP/FN/TN посчитаны по вашей колонке Class и не являются оценкой качества модели."
                  if y is not None else ""))
    st.download_button("Скачать прогнозы CSV", result.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv", key="download_predictions")


def render() -> None:
    st.header("Fraud Prediction")
    status = predictor_status()
    show_status(status)
    lab_tab, upload_tab, manual_tab = st.tabs(["Интерактивная демонстрация", "CSV-файл", "Таблица признаков"])
    with lab_tab:
        prediction_lab.render(status)
    with upload_tab:
        st.caption(f"Колонки: V1–V28 и Amount; по желанию {ID_COLUMN} и {TARGET_COLUMN}.")
        st.download_button("Скачать шаблон CSV", template_csv(), file_name="transactions_template.csv",
                           mime="text/csv", key="download_template")
        uploaded = st.file_uploader("CSV с транзакциями", type=["csv"], key="prediction_csv")
        if st.button("Проверить и предсказать", key="predict_csv", disabled=uploaded is None):
            frame = read_csv(uploaded)
            if frame is not None:
                show_result(frame, status)
    with manual_tab:
        frame = manual_frame()
        if st.button("Предсказать", key="predict_manual"):
            show_result(frame, status)
