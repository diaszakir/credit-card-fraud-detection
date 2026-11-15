# Credit Card Fraud Detection

Final Project №32 по дисциплине Programming for Data Analysis and Database: воспроизводимая
Python-система обнаружения мошеннических транзакций на датасете
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284 807 транзакций, 492 fraud; после удаления точных дубликатов — 283 726 и 473).

В проекте: аудит и EDA данных, SQLite-база с аналитическими SQL-запросами, сравнение baseline,
Logistic Regression и Random Forest с четырьмя стратегиями балансировки, единый модуль оценки,
анализ ошибок и Streamlit-приложение для получения предсказаний.

| Участник | Роль | Папки |
|---|---|---|
| Жандарбек | Data Engineering & SQL: загрузка, аудит, EDA, Parquet, разбиение, SQLite, ER-диаграмма, 13 SQL-запросов | `src/data/`, `src/db/`, `reports/figures/`, `reports/sql_results/` |
| Диас | Machine Learning: Pipeline, baseline, модели, балансировка, CV, выбор порога, сохранение модели | `src/ml/`, `models/` |
| Адам | Evaluation & Application: модуль оценки, кривые, анализ ошибок, permutation importance, итоговая оценка test, Streamlit, README | `src/evaluation/`, `app/`, `README.md`, `requirements.txt` |

Интерфейсы между модулями — в [контрактах проекта](docs/CONTRACTS.md), подготовка данных и SQL —
в [руководстве по данным](docs/data_engineering.md), модуль оценки и приложение —
в [архитектуре оценки](docs/evaluation_architecture.md).

## Быстрый старт

Нужны Git, Python 3.12 и [uv](https://docs.astral.sh/uv/). Команды выполняются из корня репозитория.
Сохранённая модель проверяет версию Python и библиотек, поэтому используйте именно Python 3.12
и версии из `requirements.txt`.

```bash
uv venv -p 3.12 .venv                 # uv сам скачает Python 3.12, если его нет
uv pip install -r requirements.txt    # ставит в .venv текущей папки
```

Если команда `uv` не найдена, но пакет установлен, замените `uv` на `python -m uv`.
Без uv: `python3.12 -m venv .venv`, затем `python -m pip install -r requirements.txt` внутри окружения.

Дальше в командах `PY` — это Python из окружения:

| Система | `PY` |
|---|---|
| Linux, macOS | `./.venv/bin/python` |
| Windows, Git Bash | `.venv/Scripts/python.exe` |
| Windows, PowerShell | `.\.venv\Scripts\python.exe` |

Данные, бинарная модель и построчные прогнозы не хранятся в Git. Получить их (нужен интернет,
скачивается CSV около 150 МБ):

```bash
PY -m src.data.cli download                    # скачать CSV и проверить число строк и fraud
PY -m src.data.cli optimize                    # удалить точные дубликаты, компактные типы, Parquet
PY -m src.data.cli split                       # хронологическое разбиение 70/15/15
PY -m src.data.cli build-db                    # SQLite fraud.db
PY -m src.ml.data_checks --review-recurring    # проверки данных перед моделью
PY -m src.ml.reproduce                         # обучить выбранную конфигурацию на train
PY -m src.ml.handoff verify                    # проверить модель, порог и контрольные суммы
```

`reproduce` обучает только выбранную конфигурацию и проверяет, что validation-оценки совпадают
с сохранённым рецептом. CV и выбор порога повторно не выполняются.

Проверка и запуск:

```bash
PY -m src.evaluation.check --strict            # окружение, файлы, модель, четыре раздела приложения
PY -m streamlit run app/fraud_app.py           # приложение: http://localhost:8501
```

## Приложение

`app/fraud_app.py` — четыре раздела в боковой панели:

- **Data Overview** — разбиение данных, EDA-графики и SQL-аналитика Жандарбека: сохранённые
  результаты запросов или живой запрос к `fraud.db`.
- **Model Comparison** — метрики выбранной модели с доверительными интервалами, итоговая оценка
  на test, PR-кривая, precision/recall/F1 от порога, ROC, таблица и графики всех 17 запусков
  Диаса, время обучения.
- **Error Analysis** — полные метрики и матрица ошибок на validation, списки FN и FP, сравнение
  признаков ошибок с правильными ответами, SQL-срезы ошибок, permutation importance.
- **Fraud Prediction** — загрузка CSV или ввод одной транзакции. Проверяется схема
  (V1–V28 и Amount; по желанию `transaction_id` и `Class`), затем сохранённый Pipeline выдаёт
  `score`, сохранённый порог и класс. Модель и порог при прогнозе не меняются.

Приложение открывается и без данных или модели: каждый раздел объясняет, какого файла нет и
какой командой его создать. Проверка схемы CSV работает без модели. Для разработки интерфейса
без модели есть режим `FRAUD_APP_STUB=1`: его прогнозы — не результат модели, и экран помечает
это красной плашкой.

## Оценка моделей

Все команды читают сохранённые прогнозы замороженной модели; порог нигде не подбирается заново.

```bash
PY -m src.evaluation.metrics                   # метрики validation, сверка с результатами ML
PY -m src.evaluation.plots                     # PNG для отчёта: PR, порог, ROC, рабочие точки
PY -m src.evaluation.importance                # permutation importance на validation
PY -m src.evaluation.final_test                # проверки перед итоговой оценкой, test не читается
PY -m src.evaluation.final_test --confirm      # однократная итоговая оценка на test
PY -m src.evaluation.final_test --show         # показать сохранённый результат test
```

Результаты пишутся в `reports/experiments/evaluation/`. Сохраняются только результаты реальной
модели (`provenance = "real"`); результаты тестовых фикстур и заглушки записать нельзя.

Что считает модуль (`src/evaluation/metrics.py`): AP, PR-AUC (трапеции), ROC-AUC, precision,
recall, F1, specificity, FPR, матрицу ошибок и 95% доверительные интервалы Уилсона для precision
и recall. Общие метрики вычисляет функция `src.ml.metrics.compute_metrics`, поэтому числа
оценки и ML совпадают; команда `metrics` останавливается с ошибкой, если это не так.
Неопределённая метрика (например, precision без положительных прогнозов) остаётся пустой, а не 0.

## Результаты

Выбранная модель: Random Forest с весами классов (`random_forest_class_weight_cv_9db7932bff3b`),
порог 0.3452 выбран на validation как максимальный recall при precision ≥ 0.85.

| Метрика | Validation | Test |
|---|---|---|
| Транзакций / fraud | 42 560 / 55 | 42 557 / 52 |
| Average precision | 0.8722 | 0.7779 |
| PR-AUC | 0.8721 | 0.7781 |
| ROC-AUC | 0.9777 | 0.9337 |
| Precision | 0.9184 | 0.7800 |
| Precision, 95% ДИ | 0.808–0.968 | 0.648–0.872 |
| Recall | 0.8182 | 0.7500 |
| Recall, 95% ДИ | 0.697–0.898 | 0.618–0.848 |
| F1 | 0.8654 | 0.7647 |
| TP / FP / FN / TN | 45 / 4 / 10 / 42 501 | 39 / 11 / 13 / 42 494 |

Источники чисел: `reports/experiments/evaluation/validation_selected.json` и `test_final.json`.
Test оценён один раз, после заморозки модели; порог не менялся, модель не переобучалась.

Как читать результат:

- **На test требование precision ≥ 0.85 не выполнено**: 0.78 при 11 ложных тревогах на 50
  положительных прогнозов. Верхняя граница интервала (0.872) выше 0.85, поэтому по 52 fraud
  нельзя утверждать, что истинная precision ниже требования, но и подтвердить его нельзя.
- Качество на test ниже, чем на validation, по всем метрикам. Возможные причины, которые проект
  не разделяет: модель и порог выбраны на одном и том же validation среди 17 запусков, что даёт
  оптимистичную оценку; разбиение хронологическое, test — более поздний период; в каждой части
  около 50 fraud, поэтому интервалы широкие и пересекаются.
- Порог после test не пересматривался: это сделало бы оценку test не независимой. Для
  следующей версии честный путь — отдельная часть данных для выбора порога и запас по precision.
- ROC-AUC высок на обеих частях, но при доле fraud около 0.13% он мало говорит о качестве
  положительных прогнозов; основная метрика проекта — average precision.

Permutation importance на validation (падение AP при перемешивании признака, 5 повторов):
V4 0.046, V14 0.032, V12 0.016, V17 0.009, V11 0.007. Признаки V1–V28 — анонимные
PCA-компоненты, поэтому важность показывает, на что опирается модель, но не даёт
содержательного объяснения. Расчёт занял 32 с на ноутбуке автора.

## Docker: запуск демо-экрана

Docker-сценарий Диаса запускает сокращённый экран `app/streamlit_app.py` (результаты validation и
прогноз). Полное приложение с четырьмя разделами запускается локально командой из раздела
«Быстрый старт».

Нужен работающий Docker Desktop с Linux containers или Docker Engine с плагином Compose. В Windows
при работе из WSL включите свою систему в Docker Desktop → Settings → Resources → WSL Integration.
[Инструкция Docker](https://docs.docker.com/desktop/features/wsl/).

```bash
docker compose up --build -d
```

Откройте **http://localhost:8501**. При первом запуске сервис `prepare` скачает исходный CSV,
подготовит данные и восстановит выбранную модель; затем запускается сервис `app`. Сделайте первый
запуск заранее, пока есть интернет. Данные, модель и примеры сохраняются в Docker volumes.

```bash
docker compose ps --all                        # prepare: Exited (0), app: healthy
docker compose logs --tail=60 prepare
docker compose up -d --no-build --pull never   # повторный запуск без интернета
docker compose down                            # остановить, сохранив данные и модель
docker compose run --rm cli                    # тесты в том же образе
docker compose run --rm cli -m src.data.cli build-db
docker compose run --rm cli -m src.data.cli query q01_split_overview
```

Образ использует Python 3.12.16 и `linux/amd64`. На ARM-компьютерах работает эмуляция, первая
подготовка идёт дольше.

## Повторить исследование целиком

Этот раздел нужен, чтобы заново получить все 17 запусков. Для работы приложения он не требуется.
Если JSON результатов уже пришли из Git, а локальные joblib и кэш отсутствуют, добавьте `--force`
к отдельной команде `run`.

```bash
PY -m src.data.cli run-all
PY -m src.ml.data_checks --review-recurring
PY -m src.ml.cli run --model dummy
PY -m src.ml.cli run --model logistic_regression
PY -m src.ml.cli run --model random_forest
PY -m src.ml.cli run --model logistic_regression --strategy class_weight
PY -m src.ml.cli run --model logistic_regression --strategy smote
PY -m src.ml.cli run --model logistic_regression --strategy undersampling
```

Три стратегии повторяются для `--model random_forest`. `--strategy none` — вариант без
балансировки. Dummy используется как baseline без пересэмплирования.

```bash
PY -m src.ml.cli tune
PY -m src.ml.cli compare
PY -m src.ml.plots
PY -m src.ml.handoff freeze
PY -m src.ml.handoff verify
```

`tune` проверяет обе модели и четыре стратегии. Запуски последовательные, один поток. После
`compare` результаты можно загрузить в SQLite:

```python
from src.db.engine import get_engine
from src.db.experiments import load_experiments

engine = get_engine()
print(load_experiments(engine))
engine.dispose()
```

## Правила экспериментов

Настройки заранее записаны в `src/ml/experiment_config.json`.

- **Разбиение.** Хронологическое 70/15/15 после удаления точных исходных дубликатов; исходные
  идентификаторы транзакций сохранены. Одинаковый `Time` не разделяется между частями.
- **Без утечек.** Масштабирование и пересэмплирование обучаются только на обучающей части;
  validation и test сохраняют естественный баланс классов.
- **CV.** Три фолда с расширяющимся окном внутри train; критерий — средняя AP.
- **Метрики.** PR-AUC вычисляется трапециями через `auc(recall, precision)`, AP — отдельно через
  `average_precision_score`. У константного baseline AP равна доле fraud.
- **Порог.** Выбирается только на полном validation: максимальный recall при precision ≥ 0.85;
  ничьи — по precision, затем больший порог. Класс определяется как `score >= threshold`.
  Недостижимое ограничение записывается как `no_feasible_threshold`, без подстановки 0.5.
- **Test.** Используется один раз, для итоговой оценки замороженной модели.
- **Повторяющиеся признаки.** Легитимные транзакции с одинаковыми признаками и разным временем
  сохраняются; дополнительные `novel_validation_metrics` описывают строки, не встречавшиеся
  в train, и не влияют на выбор модели.

## Артефакты и восстановление

Каждый запуск сохраняет весь Pipeline в `models/runs/`, JSON в `reports/experiments/results/` и
validation-оценки в локальном кэше. `freeze` создаёт `models/fraud_pipeline.joblib` и соседний
JSON со схемой признаков, порогом, версиями библиотек и контрольными суммами; `verify` проверяет
передачу. Загружайте только доверенные joblib.

После сбоя прочитайте `src/ml/progress.json` и повторите сохранённую команду: завершённый запуск
переиспользуется после проверки контрольных сумм. `--force` явно запускает обучение заново.

## Структура

```
src/config.py           общие константы: пути, названия колонок, разбиение
src/data/               загрузка, аудит, типы, разбиение, Parquet, EDA, CLI
src/db/                 схема SQLite, загрузчик, ER-диаграмма, SQL-запросы
src/ml/                 пайплайны, эксперименты, метрики, выбор порога, артефакты
src/evaluation/
  contract.py           контракт прогнозов: проверка схемы входа, Predictor, проверка выхода
  sources.py            единая точка чтения файлов команды (только чтение)
  metrics.py            метрики, интервалы, сверка с результатами ML
  curves.py, plots.py   PR, ROC, зависимость от порога, сравнение запусков
  errors.py, sql/       исходы TP/FP/FN/TN, признаки ошибок, SQL-срезы
  importance.py         permutation importance
  final_test.py         однократная итоговая оценка test
  check.py              проверка запуска
app/fraud_app.py        приложение; разделы в app/views/
app/streamlit_app.py    демо-экран для Docker
models/                 метаданные выбранной модели (joblib создаётся локально)
reports/                аудит, EDA, результаты SQL, эксперименты, оценка
docs/                   контракты, ER-диаграмма, описания модулей
tests/                  data_engineering, ml, demo, evaluation
```

## Тесты и зависимости

```bash
PY -m pytest -q
```

Тесты не скачивают реальные данные: они используют небольшие синтетические таблицы во временной
папке и файлы, сохранённые в Git.

`requirements.txt` собран uv из списка прямых зависимостей `requirements.in`:

```bash
uv pip compile requirements.in -o requirements.txt --python-version 3.12 --universal
```

Версии numpy, scipy, scikit-learn, imbalanced-learn и joblib менять нельзя без переобучения
модели: сохранённый Pipeline проверяет их при загрузке.

## Если что-то не работает

| Сообщение | Что сделать |
|---|---|
| `artifact requires Python 3.12…` или несовпадение версии пакета | Пересоздать окружение: `uv venv -p 3.12 .venv`, `uv pip install -r requirements.txt` |
| «Модель недоступна: не найден файл модели .joblib» | `PY -m src.ml.reproduce`, затем `PY -m src.ml.handoff verify` |
| «Нет fraud.db» | `PY -m src.data.cli build-db` |
| «Построчных validation-прогнозов нет» | `PY -m src.ml.reproduce` |
| «Схема входных данных не прошла проверку» | В CSV должны быть ровно V1–V28 и Amount, числа без пропусков; шаблон скачивается в разделе Fraud Prediction |
| `SHA-256 mismatch` или «не совпадают с selection_lock.json» | Файлы модели изменены после заморозки: `git checkout -- models/`, удалить `models/fraud_pipeline.joblib`, затем `PY -m src.ml.reproduce` |
| Любая другая проблема | `PY -m src.evaluation.check` покажет, какой проверке не хватает файла |

## Источники

Данные — [ULB Credit Card Fraud Detection на Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Метаданные провайдера, обозначение лицензии и SHA-256 загруженного CSV сохранены в
`reports/experiments/data_source.json`. Исходные транзакции не распространяются через Git.

Определения метрик: [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[precision–recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html),
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html);
обучение с пересэмплированием — [imblearn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html).
