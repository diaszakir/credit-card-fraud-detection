"""Model Comparison: выбранная модель и все сохранённые validation-запуски Диаса."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views.common import MODEL_NAMES, STRATEGY_NAMES, fmt, model_label, paths, show_missing
from src.evaluation import curves, plots, sources
from src.evaluation.metrics import COUNT_KEYS, add_count_metrics, evaluate_predictions_table, metrics_from_counts

TABLE_COLUMNS = {
    "model_name": "Модель", "imbalance_strategy": "Балансировка", "tuning": "Настройка", "status": "Статус",
    "average_precision": "AP", "pr_auc": "PR-AUC", "precision": "Precision", "precision_low": "Precision 95% от",
    "precision_high": "Precision 95% до", "recall": "Recall", "recall_low": "Recall 95% от",
    "recall_high": "Recall 95% до", "f1": "F1", "threshold": "Порог", "fit_seconds": "Обучение, с",
    "peak_memory_mb": "Пик памяти, МБ", "model_size_bytes": "Размер модели, байт",
}


def interval(ci) -> str:
    return "—" if not ci else f"95% ДИ {ci[0]:.3f}–{ci[1]:.3f}"


def show_selected(project) -> None:
    st.subheader("Выбранная модель")
    metadata = sources.load_model_metadata(project)
    if not metadata.available:
        show_missing(metadata, "warning")
        return
    data = metadata.data
    metrics = data.get("validation_metrics", {})
    st.write(model_label(data.get("model_name", ""), data.get("imbalance_strategy", "")) + f" · `{data.get('run_id')}`")
    derived = {}
    if all(isinstance(metrics.get(k), int) for k in COUNT_KEYS):
        derived = metrics_from_counts(*(metrics[k] for k in COUNT_KEYS))
    columns = st.columns(5)
    columns[0].metric("AP", fmt(metrics.get("average_precision")))
    columns[1].metric("PR-AUC", fmt(metrics.get("pr_auc")))
    columns[2].metric("Precision", fmt(metrics.get("precision")), interval(derived.get("precision_ci")), delta_color="off")
    columns[3].metric("Recall", fmt(metrics.get("recall")), interval(derived.get("recall_ci")), delta_color="off")
    columns[4].metric("F1", fmt(derived.get("f1")))
    st.caption(f"Validation. Порог {data.get('threshold')} выбран Диасом при precision ≥ {data.get('min_precision')}. "
               "Интервалы Уилсона показывают неопределённость из-за малого числа fraud. "
               "Итоговая оценка test — отдельный этап.")


def show_table(project) -> None:
    st.subheader("Все запуски (validation)")
    comparison = sources.load_comparison(project)
    if not comparison.available:
        show_missing(comparison, "warning")
        return
    frame = add_count_metrics(comparison.data)
    models = st.multiselect("Модели", sorted(frame["model_name"].unique()), format_func=lambda v: MODEL_NAMES.get(v, v),
                            key="cmp_models")
    if models:
        frame = frame[frame["model_name"].isin(models)]
    shown = frame[[c for c in TABLE_COLUMNS if c in frame.columns]].copy()
    shown["model_name"] = shown["model_name"].map(MODEL_NAMES).fillna(shown["model_name"])
    shown["imbalance_strategy"] = shown["imbalance_strategy"].map(STRATEGY_NAMES).fillna(shown["imbalance_strategy"])
    shown = shown.sort_values("average_precision", ascending=False, na_position="last")
    st.dataframe(shown.rename(columns=TABLE_COLUMNS), hide_index=True)
    infeasible = int((frame["status"] != "ok").sum())
    st.caption(f"Запусков: {len(frame)}. Без допустимого порога: {infeasible} — у них пороговые метрики пусты, "
               "а не равны нулю. PR-AUC — трапеции, AP — average precision. F1 и интервалы посчитаны "
               "из сохранённых TP/FP/FN/TN каждого запуска.")
    selected = sources.load_model_metadata(project)
    selected_id = selected.data.get("run_id") if selected.available else None
    min_precision = selected.data.get("min_precision", 0.85) if selected.available else 0.85
    points_tab, ap_tab, cost_tab = st.tabs(["Рабочие точки", "Average precision", "Время и память"])
    with points_tab:
        points = curves.run_operating_points(comparison.data)
        st.plotly_chart(plots.operating_points_figure(points, selected_id, min_precision), key="fig_points")
        st.caption("Каждая точка — запуск с допустимым порогом; линии — 95% интервалы Уилсона. "
                   "Пересекающиеся интервалы означают, что на 55 fraud разница между запусками не надёжна.")
    with ap_tab:
        st.plotly_chart(plots.ap_figure(comparison.data, selected_id), key="fig_ap")
    with cost_tab:
        if {"fit_seconds", "peak_memory_mb"} <= set(frame.columns):
            chart = frame.assign(label=frame["run_id"]).set_index("label")[["fit_seconds"]]
            st.bar_chart(pd.DataFrame(chart), horizontal=True, x_label="Время обучения, с")


@st.cache_data(show_spinner="Строю кривые…")
def _curves(path: str, mtime: int, threshold: float, run_id: str):
    """Кривые по сохранённым прогнозам; кэш сбрасывается при изменении файла."""
    table = pd.read_parquet(path)
    if "run_id" in table.columns:
        table = table[table["run_id"] == run_id]
    y, score = table["y_true"].to_numpy(), table["score"].to_numpy()
    return (curves.pr_curve(y, score), curves.threshold_curve(y, score), curves.roc_points(y, score),
            curves.operating_point(y, score, threshold))


def show_curves(project) -> None:
    st.subheader("Кривые выбранной модели (validation)")
    metadata = sources.load_model_metadata(project)
    if not metadata.available:
        show_missing(metadata, "warning")
        return
    predictions = sources.load_validation_predictions(project, metadata.data)
    if not predictions.available:
        show_missing(predictions)
        return
    data = metadata.data
    result = evaluate_predictions_table(predictions.data, data, split="validation")
    pr, thr, roc, point = _curves(str(predictions.path), predictions.path.stat().st_mtime_ns,
                                  float(data["threshold"]), str(data["run_id"]))
    pr_tab, thr_tab, roc_tab = st.tabs(["Precision–Recall", "Precision/Recall от порога", "ROC"])
    with pr_tab:
        st.plotly_chart(plots.pr_figure(pr, point, data["min_precision"], result["prevalence"],
                                        result["average_precision"]), key="fig_pr")
    with thr_tab:
        st.plotly_chart(plots.threshold_figure(thr, data["threshold"], data["min_precision"]), key="fig_thr")
        st.caption("Порог не подбирается заново: вертикальная линия — порог, выбранный Диасом на validation.")
    with roc_tab:
        st.plotly_chart(plots.roc_figure(roc, point, result["roc_auc"]), key="fig_roc")
        st.caption(f"При доле fraud {100 * result['prevalence']:.2f}% ROC-кривая выглядит оптимистично; "
                   "для сравнения используется PR и AP.")


def render() -> None:
    st.header("Model Comparison")
    project = paths()
    show_selected(project)
    show_curves(project)
    show_table(project)
