# Credit Card Fraud Detection

Учебный проект по Programming for AI: обнаружение мошеннических транзакций.

Подготовка данных, SQLite и SQL описаны в [руководстве по данным](docs/data_engineering.md). Интерфейсы модулей — в [контрактах проекта](docs/CONTRACTS.md).

В проекте сравниваются baseline, LogisticRegression и RandomForest на хронологическом разбиении 70/15/15. Цель — максимальный validation recall при **precision ≥ 0.85**. Test используется для итоговой оценки замороженной модели.

## Docker: запуск для защиты

Нужен работающий Docker Desktop с Linux containers или Docker Engine с плагином Compose. Python на компьютере устанавливать не требуется. В Windows при работе из WSL включите свою систему в Docker Desktop → Settings → Resources → WSL Integration. [Инструкция Docker](https://docs.docker.com/desktop/features/wsl/).

Из корня проекта выполните:

```bash
docker compose up --build -d
```

Откройте **http://localhost:8501**. При первом запуске сервис `prepare` скачает исходный CSV размером около 150 MB, подготовит данные и восстановит выбранную модель. Полная CV и повторный выбор порога не выполняются. После успешной подготовки автоматически запускается сервис `app` со Streamlit. Сделайте первый запуск заранее, пока есть интернет.

Данные, модель и подготовленные примеры сохраняются в трёх Docker volumes. Они сохраняются после остановки контейнеров. При повторном запуске подготовка проверяет сохранённые файлы и использует готовую модель.

Проверить состояние и журнал подготовки:

```bash
docker compose ps --all
docker compose logs --tail=60 prepare
docker compose logs --tail=60 app
```

У `prepare` ожидается `Exited (0)`, у `app` — `healthy`.

В интерфейсе:

- **Результаты** — сравнение 17 запусков, validation-метрики выбранной модели и графики.
- **Прогноз** — четыре примера из validation по типам исхода и загрузка своего CSV. CSV должен содержать только `V1`–`V28` и `Amount`; шаблон можно скачать в приложении. Прогноз показывает оценку fraud, сохранённый порог и итоговый класс.

После подготовки запуск на защите не требует интернета:

```bash
docker compose up -d --no-build --pull never
```

Остановить контейнеры, сохранив данные и модель:

```bash
docker compose down
```

Запустить тесты в том же образе:

```bash
docker compose run --rm cli
```

Для демонстрации SQLite после подготовки данных:

```bash
docker compose run --rm cli -m src.data.cli build-db
docker compose run --rm cli -m src.data.cli query
docker compose run --rm cli -m src.data.cli query q01_split_overview
```

Образ использует Python 3.12.16 и `linux/amd64`, чтобы участники запускали одинаковое окружение. На ARM-компьютерах используется эмуляция Docker, поэтому первая подготовка может идти дольше. Исходные транзакции и готовый joblib не включены в образ: они создаются в volumes. Docker-сценарий предназначен для восстановления выбранной модели и демонстрации; команды полного исследования приведены ниже для локального окружения.

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

## Структура ML-кода

- `src/ml/data_checks.py`: схема, контрольные суммы, manifest и временные фолды.
- `src/ml/pipelines.py`: предобработка, модель и отдельная стратегия балансировки.
- `src/ml/metrics.py`: AP, PR-AUC и выбор порога.
- `src/ml/prediction.py`: одна функция `predict_transactions` для оценки и приложения.
- `src/ml/experiment.py`, `src/ml/cli.py`: обучение, команды, время и память.
- `src/ml/comparison.py`: выбор допустимого запуска и экспорт Parquet/CSV.
- `src/ml/plots.py`: графики по проверенным сохранённым validation scores.
- `src/ml/artifacts.py`: сохранение и проверка доверенного joblib.
- `src/ml/handoff.py`: фиксация модели/порога и проверка после загрузки.
- `src/ml/reproduce.py`: восстановление выбранной конфигурации из рецепта.
- `src/ml/progress.py`, `src/ml/progress.json`: точка восстановления этапов.
- `app/prepare.py`: первая подготовка и проверка готового комплекта для демонстрации.
- `app/streamlit_app.py`, `app/ui_helpers.py`: экран результатов и прогноз через сохранённый пайплайн.

## Правила экспериментов

Настройки заранее записаны в `src/ml/experiment_config.json`. Три CV-фолда используют расширяющееся окно внутри train и четыре равных временных интервала. Одинаковый `Time` остаётся в одной части. Масштабирование и пересэмплирование обучаются только внутри обучающей части фолда; критерий CV — средняя **AP**.

**PR-AUC** вычисляется трапециями через `auc(recall, precision)`, **AP** — отдельно через `average_precision_score`. У константного baseline AP равна доле fraud; трапециевидная PR-AUC включает искусственную конечную точку кривой и отличается от AP.

Порог выбирается на полном validation. Ничьи: recall, precision, затем больший порог. Класс определяется через `score >= threshold`; столбец fraud находится по `classes_ == 1`. Недостижимое ограничение записывается как `no_feasible_threshold`, без подстановки 0.5.

Точные исходные копии удалены до `float32`, исходные ID сохранены. Повторяющиеся признаки легитимных транзакций с разным временем проверяются отдельно и сохраняются в принятой временной постановке. Дополнительные `novel_validation_metrics` описывают строки с признаками, отсутствовавшими в train; они используют уже выбранный порог и не влияют на выбор модели.

## Артефакты и восстановление

Каждый запуск сохраняет весь Pipeline в `models/runs/`, JSON в `reports/experiments/results/` и validation scores в локальном cache. Метаданные содержат версии данных, разбиения, кода и библиотек, параметры, seed и команду. `freeze` создаёт `models/fraud_pipeline.joblib` и соседний JSON с происхождением исходных файлов (`source_paths`); `verify` проверяет передачу. Загружайте только доверенные joblib.

После сбоя прочитайте `src/ml/progress.json` и повторите сохранённую команду: завершённый запуск переиспользуется после проверки контрольных сумм. `--force` явно запускает обучение заново. После выбора порога пайплайн не переобучается на train+validation. Streamlit использует этот же пайплайн и функцию предсказания; итоговая оценка на test выполняется отдельно.

После `compare` файлы результатов можно загрузить в уже созданную SQLite базу:

```python
from src.db.engine import get_engine
from src.db.experiments import load_experiments

engine = get_engine()
print(load_experiments(engine))
engine.dispose()
```

В экспорт входят только validation-прогнозы. У Dummy без допустимого порога сохраняются AP и PR-AUC, но нет выдуманных решений класса. База и построчные прогнозы остаются локальными файлами.

## Источники

Источник данных — [ULB Credit Card Fraud Detection на Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Проверенные метаданные провайдера, обозначение лицензии и SHA-256 загруженного CSV сохранены в `reports/experiments/data_source.json`. Исходные транзакции не распространяются через Git.

Определения метрик: [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) и [precision–recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html); обучение с пересэмплированием — [imblearn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html).
