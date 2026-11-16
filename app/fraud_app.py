"""Streamlit-приложение Credit Card Fraud Detection (Адам).

Запуск из корня проекта:  streamlit run app/fraud_app.py
Без модели (только для разработки интерфейса):  FRAUD_APP_STUB=1 streamlit run app/fraud_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit добавляет в путь только папку app/
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.views import common, data_overview, error_analysis, fraud_prediction, model_comparison  # noqa: E402
from src.evaluation.sources import availability  # noqa: E402

SECTIONS = {
    "Data Overview": data_overview.render,
    "Model Comparison": model_comparison.render,
    "Error Analysis": error_analysis.render,
    "Fraud Prediction": fraud_prediction.render,
}


def sidebar() -> str:
    st.sidebar.title("Fraud Detection")
    section = st.sidebar.radio("Раздел", list(SECTIONS), index=3, key="section")
    status = common.predictor_status()
    labels = {"frozen": "модель загружена", "stub": "ЗАГЛУШКА", "unavailable": "модели нет"}
    st.sidebar.caption(f"Прогнозы: {labels[status.mode]}")
    with st.sidebar.expander("Артефакты на этом компьютере"):
        table = availability(common.paths())
        st.dataframe(table[["Артефакт", "Есть"]], hide_index=True)
    return section


def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    section = sidebar()
    try:
        SECTIONS[section]()
    except Exception as exc:  # последняя защита: страница не должна показывать traceback на защите
        st.error(f"Раздел «{section}» не смог отобразиться: {exc}")


main()
