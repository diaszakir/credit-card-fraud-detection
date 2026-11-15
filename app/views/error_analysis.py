"""Error Analysis: метрики, ошибки FP/FN, их признаки, SQL-срезы и permutation importance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.views.common import fmt, paths, show_missing
from src.evaluation import errors, importance, plots, sources
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
                   f"матрица ошибок совпадают с сохранёнными результатами Диаса. ROC-AUC при доле fraud "
                   f"{100 * result['prevalence']:.2f}% выглядит оптимистично, поэтому основная метрика — AP.")


def show_error_list(table: pd.DataFrame, metadata: dict, source_name: str) -> None:
    counts = table["outcome"].value_counts().reindex(["TP", "FP", "FN", "TN"], fill_value=0)
    columns = st.columns(4)
    for column, name in zip(columns, counts.index):
        column.metric(name, int(counts[name]))
    st.caption(f"Запуск {metadata.get('run_id')}, порог {fmt(metadata.get('threshold'), 6)}. Источник: {source_name}.")
    kind = st.radio("Ошибки", ["FN", "FP"], horizontal=True, key="error_kind",
                    format_func=lambda k: {"FN": "Пропущенный fraud (FN)", "FP": "Ложная тревога (FP)"}[k])
    selected = table[table["outcome"] == kind].sort_values("score", ascending=(kind == "FN"))
    if selected.empty:
        st.write("Строк этого типа нет.")
    else:
        st.dataframe(selected[["transaction_id", "score", "y_true", "y_pred"]], hide_index=True)


@st.cache_data(show_spinner="Читаю признаки validation…")
def _profile(transactions_path: str, mtime: int, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = errors.load_split_features(transactions_path, "validation")
    joined = errors.join_features(predictions, features)
    return errors.feature_profile(joined), errors.outcome_sizes(joined)


def smd_figure(profile: pd.DataFrame, column: str, title: str, top: int = 10) -> go.Figure:
    data = profile.dropna(subset=[column]).reindex(profile[column].abs().sort_values(ascending=False).index).head(top)
    data = data.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=data[column], y=data["feature"], orientation="h", marker={"color": plots.BLUE},
        text=[f"{v:+.2f}" for v in data[column]], textposition="outside", textfont={"color": plots.TEXT},
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line={"color": plots.TEXT_2, "width": 1})
    fig.update_layout(showlegend=False, bargap=0.45)
    return plots._layout(fig, title, "Стандартизированная разность средних (d)", "", height=360)


def show_feature_profile(project, predictions: pd.DataFrame) -> None:
    if not project.transactions.is_file():
        st.info(f"Нет {project.transactions.name}. Подготовьте данные: python -m src.data.cli split.")
        return
    try:
        profile, sizes = _profile(str(project.transactions), project.transactions.stat().st_mtime_ns, predictions)
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось сопоставить прогнозы с признаками: {exc}")
        return
    st.caption(f"Размеры групп: TP {sizes['TP']}, FN {sizes['FN']}, FP {sizes['FP']}, TN {sizes['TN']}. "
               f"При {sizes['FN']} FN и {sizes['FP']} FP различия описательные, а не статистически доказанные.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(smd_figure(profile, "smd_fn_vs_tp", "FN против TP: чем пропущенный fraud отличается от пойманного"),
                        key="fig_smd_fn")
    with right:
        st.plotly_chart(smd_figure(profile, "smd_fp_vs_tn", "FP против TN: чем ложные тревоги отличаются от обычных"),
                        key="fig_smd_fp")
    shown = profile[["feature", "mean_TP", "mean_FN", "mean_FP", "mean_TN", "smd_fn_vs_tp", "smd_fn_vs_tn", "smd_fp_vs_tn"]]
    st.dataframe(shown.round(3), hide_index=True)
    st.caption("d — разность средних в единицах общего стандартного отклонения. "
               "smd_fn_vs_tn около 0 значит, что пропущенный fraud по признаку похож на обычные транзакции.")


@st.cache_data(show_spinner="Выполняю SQL…")
def _error_sql(db_path: str, mtime: int, predictions: pd.DataFrame) -> dict:
    return errors.run_error_sql(db_path, predictions)


def show_sql(project, predictions: pd.DataFrame) -> None:
    if not project.sqlite.is_file():
        st.info("Нет fraud.db. Постройте базу: python -m src.data.cli build-db.")
        return
    try:
        results = _error_sql(str(project.sqlite), project.sqlite.stat().st_mtime_ns, predictions)
    except Exception as exc:  # ошибка SQL не должна ронять страницу
        st.error(f"SQL-запросы не выполнены: {exc}")
        return
    check = results.get("e04_label_check")
    if check is not None and not check.empty:
        row = check.iloc[0]
        if int(row["missing_in_db"]) or int(row["label_mismatch"]):
            st.error(f"Прогнозы не согласуются с базой: нет в базе {int(row['missing_in_db'])}, "
                     f"расхождений меток {int(row['label_mismatch'])}.")
        else:
            st.caption(f"Сверка с базой: все {int(row['n_predictions'])} прогнозов найдены, метки совпадают.")
    for name in ("e01_errors_by_amount_band", "e02_errors_by_hour", "e03_error_ranking"):
        info = errors.describe_error_query(name)
        st.markdown(f"**{info['title']}** — {info['question']}")
        st.dataframe(results[name], hide_index=True)
        st.caption(f"Конструкции SQL: {info['uses']}. Файл: src/evaluation/sql/{name}.sql. "
                   "База открыта только для чтения, прогнозы — во временной таблице.")


def show_importance(project) -> None:
    saved = importance.load_saved(project.evaluation_dir)
    if saved is None:
        st.info("Permutation importance ещё не посчитана: python -m src.evaluation.importance "
                "(несколько минут; результат сохраняется и показывается здесь).")
        return
    table, info = saved
    top = table.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance_mean"], y=top["feature"], orientation="h", marker={"color": plots.BLUE},
        error_x={"type": "data", "array": top["importance_std"], "color": plots.TEXT_2, "thickness": 1, "width": 0},
        hovertemplate="%{y}: падение AP %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(showlegend=False, bargap=0.45)
    plots._layout(fig, "Permutation importance на validation: падение AP при перемешивании признака",
                  "Падение average precision", "", height=480)
    st.plotly_chart(fig, key="fig_importance")
    st.caption(f"Запуск {info.get('run_id')}, {info.get('n_repeats')} повторов, seed {info.get('seed')}, "
               f"базовая AP {info.get('baseline_average_precision', float('nan')):.4f}. "
               "Линия — стандартное отклонение по повторам. Признаки V1–V28 — анонимные PCA-компоненты, "
               "поэтому важность говорит, на что опирается модель, но не даёт бизнес-объяснения.")


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

    table = predictions.data.assign(outcome=lambda t: outcomes(t))
    errors_tab, features_tab, sql_tab, importance_tab = st.tabs(
        ["Ошибки", "Признаки FN/FP", "SQL-срезы ошибок", "Permutation importance"])
    with errors_tab:
        show_error_list(table, metadata.data, predictions.path.name)
    with features_tab:
        show_feature_profile(project, predictions.data)
    with sql_tab:
        show_sql(project, predictions.data)
    with importance_tab:
        show_importance(project)
