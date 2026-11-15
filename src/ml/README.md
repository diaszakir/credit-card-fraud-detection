# ML часть Диаса

Модуль сравнивает baseline, LogisticRegression и RandomForest на хронологическом разбиении 70/15/15. Цель — максимальный validation recall при **precision ≥ 0.85**. Test оставлен Адаму для итоговой оценки замороженной модели.

## Окружение

Команды — из корня репозитория. Используйте точные версии из `requirements.txt`.

Linux и macOS:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

В PowerShell замените `./.venv/bin/python` на `.\.venv\Scripts\python.exe`.

## Получить итоговую модель после клонирования

Данные, бинарные модели и построчные прогнозы не входят в Git. Метаданные выбранной модели содержат точный рецепт её восстановления. После установки окружения:

```bash
./.venv/bin/python -m src.data.cli download
./.venv/bin/python -m src.data.cli optimize
./.venv/bin/python -m src.data.cli split
./.venv/bin/python -m src.ml.data_checks --review-recurring
./.venv/bin/python -m src.ml.reproduce
./.venv/bin/python -m src.ml.handoff verify
```

`reproduce` обучает только выбранную конфигурацию на train и проверяет идентичность validation scores сохранённому рецепту. CV и выбор порога повторно не выполняются. При уже имеющемся исправном артефакте повторного обучения нет.

## Порядок запуска

Этот раздел нужен для повторения исследования до итоговой оценки test. Если JSON результатов уже пришли из Git, а локальные joblib/cache отсутствуют, добавьте `--force` к отдельной команде `run`. В том числе для CV можно выполнить `run --model ... --strategy ... --tune --force`; команда `tune` переиспользует уже полные локальные результаты. Для получения только готовой модели достаточно предыдущего раздела.

```bash
./.venv/bin/python -m src.data.cli run-all
./.venv/bin/python -m src.ml.data_checks --review-recurring
./.venv/bin/python -m src.ml.cli run --model dummy
./.venv/bin/python -m src.ml.cli run --model logistic_regression
./.venv/bin/python -m src.ml.cli run --model random_forest
./.venv/bin/python -m src.ml.cli run --model logistic_regression --strategy class_weight
./.venv/bin/python -m src.ml.cli run --model logistic_regression --strategy smote
./.venv/bin/python -m src.ml.cli run --model logistic_regression --strategy undersampling
```

Три стратегии повторяются для `--model random_forest`. `--strategy none` — вариант без балансировки. Dummy используется как baseline без пересэмплирования.

```bash
./.venv/bin/python -m src.ml.cli tune
./.venv/bin/python -m src.ml.cli compare
./.venv/bin/python -m src.ml.plots
./.venv/bin/python -m src.ml.handoff freeze
./.venv/bin/python -m src.ml.handoff verify
./.venv/bin/python -m pytest -q
```

`tune` проверяет обе модели и четыре стратегии. Для одного варианта добавьте `--tune` к `run`. Запуски последовательные, один поток.

## Что делает код

- `data_checks.py`: схема, контрольные суммы, manifest и временные фолды.
- `pipelines.py`: предобработка, модель и отдельная стратегия балансировки.
- `metrics.py`: AP, PR-AUC и выбор порога.
- `prediction.py`: одна функция `predict_transactions` для оценки и приложения.
- `experiment.py`, `cli.py`: обучение, команды, время и память.
- `comparison.py`: выбор допустимого запуска и экспорт Parquet/CSV.
- `plots.py`: графики по проверенным сохранённым validation scores.
- `artifacts.py`: сохранение и проверка доверенного joblib.
- `handoff.py`: фиксация модели/порога и проверка после загрузки.
- `reproduce.py`: восстановление выбранной конфигурации из рецепта.
- `progress.py`, `progress.json`: точка восстановления этапов.

## Правила экспериментов

Настройки заранее записаны в `experiment_config.json`. Три CV-фолда используют расширяющееся окно внутри train и четыре равных временных интервала. Одинаковый `Time` остаётся в одной части. Масштабирование и пересэмплирование обучаются только внутри обучающей части фолда; критерий CV — средняя **AP**.

**PR-AUC** вычисляется трапециями через `auc(recall, precision)`, **AP** — отдельно через `average_precision_score`. У константного baseline AP равна доле fraud; трапециевидная PR-AUC включает искусственную конечную точку кривой и отличается от AP.

Порог выбирается на полном validation. Ничьи: recall, precision, затем больший порог. Класс определяется через `score >= threshold`; столбец fraud находится по `classes_ == 1`. Недостижимое ограничение записывается как `no_feasible_threshold`, без подстановки 0.5.

Точные исходные копии удалены до `float32`, исходные ID сохранены. Повторяющиеся признаки легитимных транзакций с разным временем проверяются отдельно и сохраняются в принятой временной постановке. Дополнительные `novel_validation_metrics` описывают строки с признаками, отсутствовавшими в train; они используют уже выбранный порог и не влияют на выбор модели.

## Артефакты и восстановление

Каждый запуск сохраняет весь Pipeline в `models/runs/`, JSON в `reports/experiments/results/` и validation scores в локальном cache. Метаданные содержат версии данных, разбиения, кода и библиотек, параметры, seed и команду. `freeze` создаёт `models/fraud_pipeline.joblib` и соседний JSON с происхождением исходных файлов (`source_paths`); `verify` проверяет передачу. Загружайте только доверенные joblib.

После сбоя прочитайте `progress.json` и повторите сохранённую команду: завершённый запуск переиспользуется после проверки контрольных сумм. `--force` явно запускает обучение заново. После выбора порога пайплайн не переобучается на train+validation. Итоговые test-метрики и Streamlit добавляет Адам.

После `compare` файлы результатов можно загрузить в уже созданную SQLite базу:

```python
from src.db.engine import get_engine
from src.db.experiments import load_experiments

engine = get_engine()
print(load_experiments(engine))
engine.dispose()
```

В экспорт входят только validation-прогнозы. У Dummy без допустимого порога сохраняются AP и PR-AUC, но нет выдуманных решений класса. База и построчные прогнозы остаются локальными файлами.

## Источники и помощь AI

Источник данных — [ULB Credit Card Fraud Detection на Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Проверенные метаданные провайдера, обозначение лицензии и SHA-256 загруженного CSV сохранены в `reports/experiments/data_source.json`. Исходные транзакции не распространяются через Git.

Определения метрик: [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) и [precision–recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html); обучение с пересэмплированием — [imblearn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html).

ML-код, проверки и документация подготовлены с помощью Codex под руководством Диаса. Перед сдачей Диас должен проверить свой вклад, уметь объяснить функции и оформить раскрытие AI-помощи по правилам курса.
