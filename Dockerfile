FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HOME=/home/app \
    MPLCONFIGDIR=/home/app/.cache/matplotlib

WORKDIR /app

# Scientific wheels используют библиотеки C++ и OpenMP.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --user-group --uid 10001 app \
    && python -m venv /app/.venv

# Изменения приложения не требуют повторной установки тех же зависимостей.
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,id=credit-card-fraud-pip,target=/tmp/pip-cache \
    python -m pip install --cache-dir /tmp/pip-cache --timeout 120 --retries 10 --only-binary=:all: -r requirements.txt \
    && python -m pip check \
    && python -c "import numpy, sklearn, imblearn, streamlit"

COPY --chown=app:app src/ ./src/
COPY --chown=app:app app/ ./app/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app docs/ ./docs/
COPY --chown=app:app README.md ./README.md
COPY --chown=app:app models/fraud_pipeline.json models/selection_lock.json ./models/
COPY --chown=app:app reports/data_audit.json reports/split_summary.json ./reports/
COPY --chown=app:app reports/experiments/data_checks.json \
    reports/experiments/data_source.json \
    reports/experiments/comparison.csv \
    reports/experiments/summary.json \
    reports/experiments/handoff_verification.json \
    reports/experiments/synthetic_prediction_check.json ./reports/experiments/
COPY --chown=app:app reports/experiments/figures/*.png ./reports/experiments/figures/
COPY --chown=app:app reports/experiments/evaluation/ /opt/fraud/evaluation/

# Права этих каталогов наследуются новыми named volumes при первом запуске.
RUN mkdir -p /app/data/raw /app/data/interim /app/data/processed \
    && chown app:app /app \
    && chown -R app:app /app/data /app/models /app/reports

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["python", "-m", "streamlit", "run", "app/fraud_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
