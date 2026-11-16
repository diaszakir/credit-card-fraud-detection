"""Interactive demonstration using the frozen predictor and labelled validation examples."""
from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st

from app.views.common import paths
from src.config import FEATURE_COLUMNS
from src.evaluation.contract import check_input_schema


def load_examples(root, predictor) -> list[dict]:
    """Reject stale examples before presenting them as real validation transactions."""
    path = root / "reports" / "experiments" / "demo_examples.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    metadata = predictor.metadata
    if (document.get("source_split") != "validation"
            or document.get("run_id") != metadata.get("run_id")
            or document.get("artifact_sha256") != metadata.get("artifact_sha256")):
        raise ValueError("Примеры относятся к другой модели или выборке")
    examples = document.get("examples", [])
    if not examples:
        raise ValueError("Нет готовых примеров")
    frame = pd.DataFrame([example["features"] for example in examples])
    if not check_input_schema(frame, FEATURE_COLUMNS).ok:
        raise ValueError("Неверная схема готовых примеров")
    predicted = predictor.predict(frame)
    for example, row in zip(examples, predicted.to_dict("records")):
        if (example.get("actual_class") not in (0, 1)
                or example.get("expected_prediction") != row["y_pred"]
                or example.get("expected_score") != row["score"]):
            raise ValueError("Примеры не воспроизводятся текущей моделью")
    return examples


def feature_slider(name: str, value: float, prefix: str) -> float:
    # Bounds include the starting value and remain stable across reruns.
    low = float(min(-10, math.floor(value - 5)))
    high = float(max(10, math.ceil(value + 5)))
    return st.slider(name, min_value=low, max_value=high, value=float(value),
                     step=0.01, format="%.4f", key=f"{prefix}_{name}")


def render(status) -> None:
    st.subheader("Проверка транзакции")
    st.write("1. Выберите пример или свободный ввод. 2. Нажмите «Проверить». "
             "3. Сравните прогноз модели с фактическим результатом.")
    examples = []
    if status.mode == "frozen":
        try:
            examples = load_examples(paths().root, status.predictor)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.info(f"Готовые примеры недоступны: {exc}. Для их создания: python -m app.prepare.")
    free_mode = "Свободный ввод — своя транзакция"
    options = [e["name"] for e in examples] + [free_mode]
    labels = {
        (0, 0): "Пример обычной покупки",
        (1, 1): "Пример мошеннической операции",
        (0, 1): "Обычная покупка, на которой модель ошибается",
        (1, 0): "Мошенничество, которое модель пропускает",
    }
    def example_label(name):
        item = next((e for e in examples if e["name"] == name), None)
        return labels.get((item["actual_class"], item["expected_prediction"]), name) if item else name

    selected = st.selectbox("Выберите транзакцию", options, format_func=example_label, key="lab_example")
    example = next((e for e in examples if e["name"] == selected), None)
    base = {name: float(example["features"][name]) if example else 0.0 for name in FEATURE_COLUMNS}
    if example:
        st.write(f"Сумма покупки: **{base['Amount']:.2f}**")
        st.caption("Это реальная запись из validation. Модель получает сумму и сохранённый "
                   "обезличенный профиль транзакции. Вводить 28 признаков вручную не нужно.")
    else:
        st.info("Свободный режим: задайте сумму и V1–V28 самостоятельно. Начальные нули — "
                "синтетическая заготовка. Фактический класс неизвестен.")
    if st.button("Вернуть исходную транзакцию", key="lab_reset"):
        st.session_state["lab_revision"] = st.session_state.get("lab_revision", 0) + 1
    prefix = f"lab_{selected}_{st.session_state.get('lab_revision', 0)}"
    edited = base.copy()
    edited["Amount"] = st.number_input(
        "Введите сумму покупки", min_value=0.0, value=base["Amount"],
        step=1.0, format="%.2f", key=f"{prefix}_amount_input",
        help="Введите неотрицательную сумму с клавиатуры и нажмите Enter.")
    st.caption("Сумма учитывается вместе со всеми введёнными V-признаками. Фактический класс неизвестен."
               if example is None else "Сумма учитывается вместе с сохранённым профилем выбранной транзакции. "
               "Одна сумма не определяет fraud. При изменении суммы это гипотетический пример, "
               "для которого фактический класс неизвестен.")
    with st.expander("Введите признаки V1–V28" if example is None else "Технические признаки V1–V28 (необязательно)",
                     expanded=example is None):
        st.write("Это входные данные о транзакции после PCA-преобразования для обезличивания. "
                 "Их исходный смысл не опубликован. Это не гиперпараметры: гиперпараметры "
                 "задают устройство модели, например количество деревьев в Random Forest.")
        st.caption("Изменение признаков создаёт синтетический сценарий; их сочетание может не встречаться в реальных покупках.")
        columns = st.columns(3)
        for i, name in enumerate(n for n in FEATURE_COLUMNS if n != "Amount"):
            with columns[i % 3]:
                if example is None:
                    edited[name] = st.number_input(name, value=0.0, step=0.1, format="%.4f", key=f"{prefix}_{name}")
                else:
                    edited[name] = feature_slider(name, base[name], prefix)
    frame = pd.DataFrame([edited], columns=FEATURE_COLUMNS)
    with st.expander("Посмотреть все входные данные"):
        st.dataframe(frame, hide_index=True)
    st.download_button("Скачать эту транзакцию CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name="my_transaction.csv", mime="text/csv", key="lab_download")
    if status.predictor is None:
        st.info("Для предсказания восстановите модель по инструкции выше.")
        return
    if not st.button("Проверить транзакцию", type="primary", key="lab_predict"):
        st.caption("Нажмите «Проверить транзакцию», чтобы получить решение модели.")
        return
    try:
        result = status.predictor.predict(pd.DataFrame([base, edited], columns=FEATURE_COLUMNS))
    except ValueError as exc:
        st.error(f"Не удалось получить прогноз: {exc}")
        return
    original, current = result.iloc[0], result.iloc[1]
    if int(current["y_pred"]):
        st.warning("🚨 FRAUD — модель обнаружила подозрительную транзакцию")
    else:
        st.success("НЕ FRAUD — модель отнесла транзакцию к обычным")
    changed = any(edited[n] != base[n] for n in FEATURE_COLUMNS)
    if example and not changed:
        actual = "Мошенничество" if example["actual_class"] else "Обычная покупка"
        st.write(f"**Фактически в датасете: {actual}.**")
        if int(current["y_pred"]) == example["actual_class"]:
            st.success("Прогноз совпал с фактическим классом.")
        else:
            st.warning("Модель ошиблась на этом примере. Это показывает её ограничения.")
        st.caption("Примеры специально отобраны для демонстрации удачных решений и ошибок. "
                   "По этим четырём транзакциям нельзя судить об общем качестве модели.")
    elif changed:
        st.info("Вы изменили транзакцию. Её фактический класс неизвестен: сейчас показан только прогноз.")
    with st.expander("Как модель приняла решение? Оценка и порог"):
        st.write("Модель выдаёт оценку от 0 до 1. Если оценка достигает сохранённого порога, "
                 "транзакция считается подозрительной. Сам порог во время демонстрации не меняется.")
        cols = st.columns(3)
        cols[0].metric("Оценка fraud · score", f"{current['score']:.4f}",
                       delta=f"{current['score'] - original['score']:+.4f} к исходной")
        cols[1].metric("Сохранённый порог", f"{current['threshold']:.4f}")
        cols[2].metric("Изменено признаков", str(sum(edited[n] != base[n] for n in FEATURE_COLUMNS)))
        st.progress(float(current["score"]), text="Оценка модели от 0 до 1")
        st.caption("Score — оценка модели, не гарантия и не откалиброванная вероятность. "
                   "Правило: score ≥ порога → fraud. Порог и модель не изменяются.")
        comparison = result[["score", "threshold", "y_pred"]].copy()
        comparison.index = ["Исходная транзакция" if example else "Нулевая синтетическая заготовка", "Ваш сценарий"]
        comparison["Решение"] = comparison["y_pred"].map({0: "Не fraud", 1: "Fraud"})
        st.dataframe(comparison, hide_index=False)
        if example and any(edited[n] != base[n] for n in FEATURE_COLUMNS):
            st.caption("Фактическая метка исходного примера не переносится на изменённую транзакцию: её истинный класс неизвестен.")
