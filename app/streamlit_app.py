"""Одна страница: сохранённые validation-результаты и прогноз обученного пайплайна."""

import streamlit as st

from app import ui_helpers
from src.ml.prediction import predict_transactions


MODEL_NAMES = {"dummy": "Dummy baseline", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
STRATEGY_NAMES = {"none": "Без балансировки", "class_weight": "Веса классов", "smote": "SMOTE", "undersampling": "Undersampling"}


def show_results():
    st.subheader("Выбранная модель — validation")
    st.info("Все показатели на этой странице относятся к validation. Итоговая оценка на test ещё не выполнена.")
    try:
        metadata = ui_helpers.load_saved_metadata()
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    metrics = metadata["validation_metrics"]
    st.write(f"{MODEL_NAMES.get(metadata.get('model_name'), metadata.get('model_name', 'Модель'))} · "
             f"{STRATEGY_NAMES.get(metadata.get('imbalance_strategy'), metadata.get('imbalance_strategy', ''))}")
    columns = st.columns(4)
    for column, name, label in zip(columns, ["precision", "recall", "pr_auc", "average_precision"],
                                   ["Precision fraud", "Recall fraud", "PR-AUC", "AP"]):
        column.metric(label, f"{metrics[name]:.4f}")
    st.caption(f"Сохранённый порог: {metadata['threshold']}. Минимальная precision: {metadata['min_precision']:.0%}.")
    st.caption("PR-AUC рассчитана трапециями; AP рассчитана отдельно через average precision.")
    if metrics["precision"] < metadata["min_precision"]:
        st.warning("На validation согласованный минимум precision не достигнут.")

    st.subheader("Сравнение экспериментов — validation")
    try:
        comparison = ui_helpers.load_comparison()
        st.caption(f"Сохранённых запусков: {len(comparison)}. Пустые значения означают отсутствие допустимого порога.")
        shown = comparison[["model_name", "imbalance_strategy", "tuning", "status", "precision", "recall", "pr_auc", "average_precision", "threshold"]].copy()
        shown["model_name"] = shown["model_name"].map(MODEL_NAMES).fillna(shown["model_name"])
        shown["imbalance_strategy"] = shown["imbalance_strategy"].map(STRATEGY_NAMES).fillna(shown["imbalance_strategy"])
        shown = shown.rename(columns={"model_name": "Модель", "imbalance_strategy": "Балансировка", "tuning": "Настройка", "status": "Статус", "precision": "Precision", "recall": "Recall", "pr_auc": "PR-AUC", "average_precision": "AP", "threshold": "Порог"})
        st.dataframe(shown, hide_index=True)
        st.caption("У константного baseline AP отражает долю fraud; трапециевидная PR-AUC включает конечную точку кривой и отличается от AP.")
    except (OSError, ValueError) as exc:
        st.warning(str(exc))

    figures = [
        ("validation_average_precision.png", "AP всех запусков на validation"),
        ("validation_precision_recall.png", "Precision–recall на validation"),
        ("validation_confusion_matrix.png", "Матрица ошибок выбранной модели на validation"),
    ]
    for filename, caption in figures:
        path = ui_helpers.ROOT_DIR / "reports" / "experiments" / "figures" / filename
        if path.is_file():
            st.image(str(path), caption=caption)


def show_prediction(frame):
    try:
        pipeline, metadata = ui_helpers.load_frozen_model()
        predictions = predict_transactions(frame, pipeline, metadata)
    except (OSError, ValueError) as exc:
        st.error(ui_helpers.prediction_error(exc))
        return
    st.dataframe(ui_helpers.prediction_table(predictions))
    st.caption("Fraud означает score ≥ сохранённого порога. Модель и порог при прогнозе не изменяются.")


def show_forecast():
    st.subheader("Готовый пример")
    st.caption("Примеры из validation подобраны по исходам TP, TN, FN и FP. Четыре строки не являются отдельной оценкой качества.")
    csv_template = ui_helpers.example_csv()
    try:
        examples = ui_helpers.load_demo_examples()
    except (OSError, ValueError) as exc:
        st.warning(f"Готовые примеры пока недоступны: {exc}")
    else:
        name = st.selectbox("Пример из validation", [example["name"] for example in examples], key="demo_example")
        selected = next(example for example in examples if example["name"] == name)
        csv_template = ui_helpers.example_csv(selected)
        actual = "Fraud" if selected["actual_class"] == 1 else "Обычная транзакция"
        st.write(f"Транзакция {selected['transaction_id']}. Фактический класс: {actual}.")
        if st.button("Рассчитать прогноз", key="predict_example"):
            show_prediction(ui_helpers.example_frame(selected))

    st.subheader("CSV с транзакциями")
    st.caption("Нужны 29 числовых колонок: V1–V28 и Amount. Используйте запятую как разделитель; Class и transaction_id не включайте.")
    st.download_button("Скачать CSV-шаблон", data=csv_template, file_name="transaction_example.csv",
                       mime="text/csv", key="download_csv_template")
    uploaded = st.file_uploader("Выберите CSV", type=["csv"], key="prediction_csv")
    if st.button("Рассчитать прогноз CSV", key="predict_csv", disabled=uploaded is None):
        try:
            frame = ui_helpers.read_prediction_csv(uploaded)
        except ValueError as exc:
            st.error(str(exc))
        else:
            show_prediction(frame)


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    st.title("Обнаружение мошеннических транзакций")
    results, forecast = st.tabs(["Результаты", "Прогноз"])
    with results:
        show_results()
    with forecast:
        show_forecast()


if __name__ == "__main__":
    main()
