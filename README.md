# Credit Card Fraud Detection

Final Project №32 по дисциплине Programming for AI: воспроизводимая
Python-система обнаружения мошеннических транзакций на датасете
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284 807 транзакций, 492 fraud; после удаления точных дубликатов — 283 726 и 473).

В проекте: аудит и EDA данных, SQLite-база с аналитическими SQL-запросами, сравнение baseline,
Logistic Regression и Random Forest с четырьмя стратегиями балансировки, единый модуль оценки,
анализ ошибок и Streamlit-приложение для получения предсказаний.

Интерфейсы между модулями — в [контрактах проекта](docs/CONTRACTS.md), подготовка данных и SQL —
в [руководстве по данным](docs/data_engineering.md), модуль оценки и приложение —
в [архитектуре оценки](docs/evaluation_architecture.md).

## Быстрый старт

Нужны Git и Python 3.12. Команды выполняются из корня репозитория.
Сохранённая модель проверяет версию Python и библиотек: устанавливайте зависимости из
`requirements.txt` через pip.

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

Если данные и модель уже подготовлены, сразу запустите полное приложение:

```bash
./.venv/bin/python -m streamlit run app/fraud_app.py
```

В PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/fraud_app.py
```

Откройте **http://localhost:8501**. Для остановки нажмите `Ctrl+C` в терминале сервера.
На чистом клоне сначала выполните подготовку ниже. Её также можно запустить одной командой
(потребуется интернет для исходных данных):

```bash
./.venv/bin/python -m app.prepare --full-app
```

В PowerShell: `.\.venv\Scripts\python.exe -m app.prepare --full-app`.
Команда проверяет замороженную модель, готовит примеры, SQLite, EDA и SQL-результаты.

После изменения кода Docker нужно пересобрать: `docker compose up --build -d`.
Оба способа используют порт 8501: перед переходом с Docker на локальный запуск выполните
`docker compose down`; перед переходом на Docker остановите локальный Streamlit через `Ctrl+C`.

В дальнейших командах `PY` — обозначение пути к Python: замените его на значение из таблицы.
Это не отдельная установленная команда.

| Система | `PY` |
|---|---|
| Linux, macOS | `./.venv/bin/python` |
| Windows, Git Bash | `.venv/Scripts/python.exe` |
| Windows, PowerShell | `.\.venv\Scripts\python.exe` |

Данные, бинарная модель и построчные прогнозы не хранятся в Git. Получить их (нужен интернет,
скачивается CSV около 150 МБ):

```bash
PY -m src.data.cli download                    # скачать CSV и проверить число строк и fraud
PY -m src.data.cli audit                       # аудит исходных данных
PY -m src.data.cli optimize                    # удалить точные дубликаты, компактные типы, Parquet
PY -m src.data.cli split                       # хронологическое разбиение 70/15/15
PY -m src.data.cli eda                         # EDA на train
PY -m src.data.cli build-db                    # SQLite fraud.db
PY -m src.data.cli sql                         # выполнить SQL и сохранить результаты
PY -m src.data.cli query q01_split_overview     # пример отдельного запроса
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

## Docker: одинаковый запуск у всей команды

На каждом компьютере нужен Git и работающий Docker Desktop (Linux containers) либо Docker Engine
с плагином Compose. Python, pip, исходный CSV и joblib на компьютере устанавливать/копировать не нужно.
Команды одинаковы в PowerShell, macOS и Linux:

```bash
git clone https://github.com/diaszakir/credit-card-fraud-detection.git
cd credit-card-fraud-detection
docker compose up --build -d
```

Откройте **http://localhost:8501** на своём компьютере. Первый запуск занимает несколько минут:
`prepare` скачивает CSV около 150 МБ, проверяет SHA-256, готовит данные, восстанавливает выбранную
модель на train, создаёт SQLite/EDA/SQL и проверяет комплект. До успешной проверки сервер не стартует.
Порог не подбирается заново, test при подготовке не читается. Прогресс и готовность:

```bash
docker compose logs -f prepare
```

`Ctrl+C` завершает просмотр логов, подготовка продолжается. Затем:

```bash
docker compose ps --all
```

Ожидается `prepare: Exited (0)`, `app: Up (healthy)`. Полный интерфейс и ручной ввод одинаковы
с локальным запуском `app/fraud_app.py`. В Windows при использовании WSL включите свою систему
в Docker Desktop → Settings → Resources → WSL Integration.

Чтобы у всех была одинаковая версия приложения, используйте один коммит (`git rev-parse HEAD`).
После получения обновлений из GitHub:

```bash
git pull --ff-only
docker compose up --build -d
```

Локальные правки Git может попросить сохранить отдельно; не сбрасывайте их ради обновления.
Сборка фиксирует базовый Python 3.12.16 по digest, архитектуру `linux/amd64`, версии зависимостей,
число вычислительных потоков, locale и часовой пояс. На ARM используется эмуляция, поэтому подготовка
может идти дольше. Перед стартом проверяется обученное состояние восстановленной модели и сохранённый
порог: разные байты сериализации joblib не считаются другой моделью при совпадении состояния и рецепта.

Данные и модель хранятся в named volumes, а исходники приложения берутся из нового образа.
Базовые команды:

```bash
docker compose logs --tail=60 app
docker compose exec -T app python -m src.evaluation.check --strict
docker compose down                          # остановка с сохранением данных
docker compose up -d --no-build --pull never # повторный запуск подготовленного комплекта без интернета
docker compose run --rm cli                   # тесты в том же образе
```

Не используйте `docker compose down -v` для обычного обновления: этот флаг удаляет данные и модель.
Для первого запуска нужен интернет. Исходные транзакции и бинарные модели в Git и build context не входят.
Локальный Streamlit нужно остановить перед Docker, если оба используют порт 8501.

Если порт занят, можно выбрать другой порт без изменения файлов проекта.

PowerShell:

```powershell
$env:FRAUD_APP_PORT="8502"
docker compose up --build -d
```

Linux/macOS:

```bash
FRAUD_APP_PORT=8502 docker compose up --build -d
```

Тогда адрес — **http://localhost:8502**. Альтернатива для всех ОС — файл `.env` в корне проекта
со строкой `FRAUD_APP_PORT=8502` (локальный файл не коммитится).

Если подготовка завершилась с ошибкой, посмотрите `docker compose logs --tail=100 prepare`.
При недоступной загрузке проверьте интернет и повторите `docker compose up --build -d`.
При несовпадении контрольных сумм подготовка остановится: не отключайте эту проверку и не заменяйте
файлы из разных версий проекта. Обратитесь к команде с текстом ошибки.

## Приложение

`app/fraud_app.py` — четыре раздела в боковой панели:

- **Data Overview** — разбиение данных, EDA-графики и SQL-аналитика: сохранённые
  результаты запросов или живой запрос к `fraud.db`.
- **Model Comparison** — метрики выбранной модели с доверительными интервалами, итоговая оценка
  на test, PR-кривая, precision/recall/F1 от порога, ROC, таблица и графики всех 17 экспериментов, время обучения.
- **Error Analysis** — полные метрики и матрица ошибок на validation, списки FN и FP, сравнение
  признаков ошибок с правильными ответами, SQL-срезы ошибок, permutation importance.
- **Fraud Prediction** — стартовый экран для защиты: примеры TP/TN/FN/FP из validation,
  пятый вариант «Свободный ввод» с числовыми полями всех признаков, поле суммы
  и скрытые ползунки PCA-признаков для готовых примеров, сравнение исходной транзакции
  с вашим сценарием и скачивание входных данных. Также доступны загрузка CSV и табличный ввод. Проверяется схема
  (V1–V28 и Amount; по желанию `transaction_id` и `Class`), затем сохранённый Pipeline выдаёт
  `score`, сохранённый порог и класс. Модель и порог при прогнозе не меняются.

Приложение открывается и без данных или модели: каждый раздел объясняет, какого файла нет и
какой командой его создать. Проверка схемы CSV работает без модели. Для разработки интерфейса
без модели есть режим `FRAUD_APP_STUB=1`: его прогнозы — не результат модели, и экран помечает
это красной плашкой.

## Демонстрация на защите

Пошаговая инструкция по предсказаниям и объяснение V1–V28: [руководство для демонстрации](docs/DEMO_GUIDE.md).

1. Запустите `PY -m app.prepare`, чтобы проверить комплект модели и создать четыре готовых примера.
2. Откройте приложение: стартовый раздел — **Fraud Prediction**, вкладка «Интерактивная демонстрация».
3. Выберите «Пример обычной покупки» или «Пример мошеннической операции» и нажмите «Проверить транзакцию».
4. Покажите решение модели, фактический класс и сообщение о совпадении или ошибке.
5. Введите другую сумму в поле на основном экране или раскройте технические признаки. Основная демонстрация не требует ручного ввода V1–V28.

Особенность приложения — понятная проверка реальных validation-примеров с демонстрацией
правильных решений и ошибок. Дополнительно доступно сравнение «до / после» с фиксированным порогом.
Это особенность реализации проекта, а не заявление о мировой новизне метода.
V1–V28 — анонимные PCA-компоненты; изменённые строки являются гипотетическими сценариями.
Score не следует интерпретировать как откалиброванную вероятность мошенничества.

## Оценка моделей

Все команды читают сохранённые прогнозы замороженной модели; порог нигде не подбирается заново.

```bash
PY -m src.evaluation.metrics                   # метрики validation, сверка с результатами ML
PY -m src.evaluation.plots                     # PNG для отчёта: PR, порог, ROC, рабочие точки
PY -m src.evaluation.importance                # permutation importance на validation
PY -m src.evaluation.final_test                # проверки перед итоговой оценкой, test не читается
PY -m src.evaluation.final_test --confirm      # итоговая оценка с уже зафиксированной моделью
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
Первичная оценка test выполнена после заморозки модели. Позднее результат явно перепроверен
на текущем замороженном пакете из-за отличающихся хешей сериализации: все опубликованные
метрики совпали, модель и порог не изменялись. Исходный результат, его fingerprint и дата
сохранены в `provenance_verification.original_result` внутри `test_final.json`.
Проверка не была новым поиском модели или порога.

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
содержательного объяснения. Сохранённое время расчёта — 32 с; на другом оборудовании оно может отличаться.

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
- **Test.** Первичная оценка проводится после заморозки модели. Повторная проверка того же
  решения допустима для контроля воспроизводимости; test не используется для подбора модели или порога.
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
  final_test.py         итоговая оценка test и проверка её происхождения
  check.py              проверка запуска
app/fraud_app.py        приложение; разделы в app/views/
app/streamlit_app.py    прежний сокращённый демо-экран
models/                 метаданные выбранной модели (joblib создаётся локально)
reports/                аудит, EDA, результаты SQL, эксперименты, оценка
docs/                   контракты, ER-диаграмма, описания модулей
tests/                  data_engineering, ml, demo, evaluation
```

## Тесты и зависимости

```bash
PY -m pytest -q
```

Тесты не скачивают реальные данные: большинство использует синтетические таблицы во временной
папке и файлы из Git. При наличии локального датасета дополнительно выполняются проверки реальных данных.

`requirements.in` перечисляет прямые зависимости; `requirements.txt` фиксирует также
транзитивные зависимости и платформенные маркеры. Для установки используйте
`PY -m pip install -r requirements.txt`, для проверки — `PY -m pip check`.

Версии numpy, scipy, scikit-learn, imbalanced-learn и joblib менять нельзя без переобучения
модели: сохранённый Pipeline проверяет их при загрузке.

## Если что-то не работает

| Сообщение | Что сделать |
|---|---|
| `artifact requires Python 3.12…` или несовпадение версии пакета | Пересоздать окружение: создать `.venv` средствами Python 3.12 и установить зависимости через pip (команды выше) |
| «Модель недоступна: не найден файл модели .joblib» | `PY -m src.ml.reproduce`, затем `PY -m src.ml.handoff verify` |
| «Нет fraud.db» | `PY -m src.data.cli build-db` |
| «Построчных validation-прогнозов нет» | `PY -m src.ml.reproduce` |
| «Схема входных данных не прошла проверку» | В CSV должны быть ровно V1–V28 и Amount, числа без пропусков; шаблон скачивается в разделе Fraud Prediction |
| `SHA-256 mismatch` или «не совпадают с selection_lock.json» | Файлы модели изменены после заморозки: проверить локальные изменения метаданных и `selection_lock.json`, восстановить согласованный рецепт из Git; сохранить копию старого joblib вне проекта перед `PY -m src.ml.reproduce` |
| Любая другая проблема | `PY -m src.evaluation.check` покажет, какой проверке не хватает файла |

## Источники

Данные — [ULB Credit Card Fraud Detection на Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Метаданные провайдера, обозначение лицензии и SHA-256 загруженного CSV сохранены в
`reports/experiments/data_source.json`. Исходные транзакции не распространяются через Git.

Определения метрик: [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[precision–recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html),
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html);
обучение с пересэмплированием — [imblearn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html).
