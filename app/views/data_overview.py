"""Data Overview: разбиение, EDA-графики и SQL-аналитика Жандарбека."""

from __future__ import annotations

import streamlit as st

from app.views.common import SPLIT_NAMES, paths, show_missing
from src.evaluation import sources

SPLIT_COLUMNS = {
    "split": "Часть", "n_rows": "Транзакций", "n_fraud": "Fraud", "n_legitimate": "Обычных",
    "fraud_rate_percent": "Доля fraud, %", "share_of_rows_percent": "Доля строк, %",
    "time_min": "Time от, с", "time_max": "Time до, с",
}


def show_split(project) -> None:
    st.subheader("Разбиение данных")
    split = sources.load_split_summary(project)
    if not split.available:
        show_missing(split, "warning")
        return
    frame = split.data
    total_rows, total_fraud = int(frame["n_rows"].sum()), int(frame["n_fraud"].sum())
    left, middle, right = st.columns(3)
    left.metric("Транзакций после дедупликации", f"{total_rows:,}".replace(",", " "))
    middle.metric("Fraud", f"{total_fraud:,}".replace(",", " "))
    right.metric("Доля fraud", f"{100 * total_fraud / total_rows:.3f} %")
    shown = frame[[c for c in SPLIT_COLUMNS if c in frame.columns]].copy()
    shown["split"] = shown["split"].map(SPLIT_NAMES).fillna(shown["split"])
    st.dataframe(shown.rename(columns=SPLIT_COLUMNS), hide_index=True)
    st.caption("Хронологическое разбиение 70/15/15 после удаления точных дубликатов (docs/CONTRACTS.md, раздел 4).")


def show_eda(project) -> None:
    st.subheader("EDA")
    figures = sources.list_figures(project.eda_figures_dir, "eda_*.png")
    if not figures:
        st.info("EDA-графики не найдены. Постройте их: python -m src.data.cli eda.")
        return
    names = {path.stem: path for path in figures}
    chosen = st.selectbox("График", list(names), key="eda_figure")
    st.image(str(names[chosen]))


def show_sql(project) -> None:
    st.subheader("SQL-аналитика")
    listed = sources.list_sql_results(project)
    if not listed.available:
        show_missing(listed)
        return
    name = st.selectbox("Запрос", list(listed.data), key="sql_query")
    catalog = sources.describe_sql_queries()
    if catalog.available and "name" in catalog.data.columns:
        row = catalog.data[catalog.data["name"] == name]
        if not row.empty:
            info = row.iloc[0]
            st.markdown(f"**{info.get('title', name)}** — {info.get('question', '')}")
            constructs = info.get("detected_constructs") or info.get("declared_constructs") or ""
            if not isinstance(constructs, str):
                constructs = ", ".join(map(str, constructs))
            st.caption(f"Конструкции SQL: {constructs or 'не указаны'}. Данные: {info.get('scope', '')}.")
    live = project.sqlite.is_file() and st.toggle("Выполнить на fraud.db", key="sql_live",
                                                   help="Без переключателя показывается сохранённый CSV.")
    result = sources.run_live_query(project, name) if live else sources.load_sql_result(project, name)
    if not result.available:
        show_missing(result, "warning")
        return
    st.dataframe(result.data, hide_index=True)
    st.caption(f"Источник: {result.path.name}" + (" (живой запрос)" if live else " (сохранённый результат)"))


def render() -> None:
    st.header("Data Overview")
    project = paths()
    show_split(project)
    show_eda(project)
    show_sql(project)
