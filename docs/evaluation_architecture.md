# Evaluation & Application — архитектура (Адам)

Документ описывает модуль оценки `src/evaluation/` и Streamlit-приложение `app/fraud_app.py`.
Владельцы соседних модулей и их интерфейсы — в `docs/CONTRACTS.md`. Здесь ничего не меняется
в чужих папках: модуль Адама только **читает** результаты Жандарбека и Диаса.

## 1. Поток данных

```
Жандарбек (src/data, src/db)                  Диас (src/ml, models/)
  data/processed/transactions.parquet           models/fraud_pipeline.joblib  (локально, не в Git)
  data/processed/fraud.db                       models/fraud_pipeline.json    (порог, схема, метрики)
  reports/split_summary.json                    models/selection_lock.json    (SHA фиксации)
  reports/sql_results/*.csv                     reports/experiments/comparison.csv, runs/metrics.parquet
  reports/figures/*.png                         reports/experiments/cache/<run>/validation.parquet
            │                                              │
            └──────────────┬───────────────────────────────┘
                           ▼
          src/evaluation/sources.py      — единая точка чтения, ничего не пишет
          src/evaluation/contract.py     — контракт прогнозов, проверка схемы, Predictor
          (этап 2) metrics.py            — AP, PR-AUC, ROC-AUC, P/R/F1, confusion matrix
          (этап 3) curves.py, plots.py   — PR-кривые, P/R от порога, сравнение моделей
          (этап 4) errors.py, importance.py — FP/FN, permutation importance
          (этап 5) final_test.py         — однократная итоговая оценка test
                           ▼
          app/fraud_app.py + app/views/*  — Data Overview, Model Comparison,
                                            Error Analysis, Fraud Prediction
```

Правило слоёв: страницы `app/views/*` не открывают файлы сами и не вызывают `joblib`.
Они получают данные только через `sources.py` и прогнозы только через `Predictor`.
Поэтому любую страницу можно разрабатывать и тестировать на временной папке с
синтетическими файлами, пока итоговой модели нет на компьютере.

## 2. Контракт прогнозов

### Вход

| Колонка | Обязательна | Правило |
|---|---|---|
| `V1`…`V28`, `Amount` | да | ровно 29 признаков из `metadata["feature_names"]`, конечные числа |
| `transaction_id` | нет | целые уникальные ≥ 0; становится индексом результата |
| `Class` | нет | 0/1; не подаётся в модель, используется только для сравнения с прогнозом |

Любая другая колонка — ошибка схемы. `check_input_schema` собирает **все** проблемы сразу
(отсутствующие, лишние, нечисловые колонки, пропуски/inf), чтобы пользователь увидел полный
список. Затем строгая проверка Диаса `src.ml.prediction.validate_features` приводит порядок и
типы к сохранённой схеме.

### Выход `Predictor.predict(X)`

| Колонка | Тип | Правило |
|---|---|---|
| `score` | float64 | вероятность fraud в [0, 1], столбец `classes_ == 1` |
| `threshold` | float64 | одинаковый во всех строках, из метаданных модели |
| `y_pred` | int8 | ровно `score >= threshold` |

Индекс и порядок строк совпадают со входом. `validate_prediction_frame` проверяет это после
каждого вызова, поэтому ошибка в любой реализации сразу становится видна.

### Реализации `Predictor`

| Класс | Когда | provenance |
|---|---|---|
| `FrozenPipelinePredictor` | есть `models/fraud_pipeline.joblib`, SHA совпадает с `selection_lock.json` | `real` |
| `StubPredictor` | только при явном `FRAUD_APP_STUB=1` или в тестах | `stub` |

`StubPredictor` нужен только для разработки интерфейса. Его `score` — детерминированный хеш
значений строки, а не модель; порог 0.5 условный. Приложение показывает красную плашку
«Заглушка», а будущий модуль метрик отказывается сохранять результаты с `provenance != "real"`.
Автоматической подмены реальной модели заглушкой нет: при отсутствии `.joblib` приложение
показывает команду восстановления из README.

## 3. Контракт таблицы прогнозов (для Error Analysis)

Читается `reports/experiments/predictions.parquet` (экспорт `src.ml.comparison`) или кэш
выбранного запуска `metadata["source_paths"]["validation_scores"]`:
`run_id`, `transaction_id`, `split`, `y_true`, `score`, `y_pred`.
Проверки (`validate_predictions_table`): уникальные `transaction_id` внутри `run_id`, метки 0/1,
score в [0, 1], и если известен порог — `y_pred == (score >= threshold)`.

## 4. Правила оценки, которые соблюдает модуль

1. Порог не подбирается заново: используется сохранённый `threshold` Диаса.
2. Test читается один раз, в `final_test.py`; результат пишется в отдельный файл и не меняет
   `models/fraud_pipeline.json`.
3. PR-AUC — трапеции `auc(recall, precision)`, AP — `average_precision_score`, как у Диаса
   (`src.ml.metrics.compute_metrics` переиспользуется, а не копируется).
4. Отсутствующий порог (Dummy) — это «нет допустимого порога», а не 0 или NaN.
5. В validation 55 fraud, в test 52 (`reports/split_summary.json`): у precision/recall будет
   указана неопределённость (этап 2: доверительный интервал Уилсона).

## 5. План этапов

| Этап | Результат | Зависит от |
|---|---|---|
| 1 | архитектура, контракт, `sources.py`, каркас 4 разделов, тесты | — |
| 2 | `metrics.py`: AP, PR-AUC, ROC-AUC, P/R/F1, CM, интервалы | кэш validation Диаса |
| 3 | `curves.py`, `plots.py`: PR-кривые, P/R от порога, сравнение 17 запусков | этап 2 |
| 4 | `errors.py`: FP/FN, SQL JOIN с `transactions`; `importance.py`: permutation importance | этап 2, `fraud.db` |
| 5 | `final_test.py`: однократная оценка test, отчёт | замороженная модель |
| 6 | README, `requirements.txt` через `uv pip compile`, проверка чистого клона | всё выше |

## 6. Что нужно согласовать с командой

* **Диас:** Dockerfile/compose запускают `app/streamlit_app.py`. После приёмки нового приложения —
  переключить их на `app/fraud_app.py` (это его файлы). Старое демо до этого не удаляется.
* **Жандарбек:** живые SQL-запросы читаются через `src.db.queries.run_query`; если `fraud.db`
  нет, страница показывает сохранённые CSV из `reports/sql_results/`.
