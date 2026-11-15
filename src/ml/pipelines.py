"""Простые пайплайны: одна модель и одна стратегия дисбаланса за запуск."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("dummy", "logistic_regression", "random_forest")
STRATEGIES = ("none", "class_weight", "smote", "undersampling")
SAMPLING_RATIO = 0.1
SMOTE_NEIGHBORS = 5


def make_pipeline(model_name, strategy="none", seed=42, params=None):
    """Создать Pipeline; params содержит ключи вида ``clf__C`` для настройки модели.

    Imputer, scaler и sampler обучаются только при fit. При predict / predict_proba
    sampler не меняет число строк. Стратегии не объединяются между собой.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if model_name == "dummy" and strategy != "none":
        raise ValueError("DummyClassifier is a baseline with strategy='none'")

    class_weight = "balanced" if strategy == "class_weight" else None
    if model_name == "dummy":
        model = DummyClassifier(strategy="prior", random_state=seed)
    elif model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=seed,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )

    sampler = "passthrough"
    if strategy == "smote":
        sampler = SMOTE(
            sampling_strategy=SAMPLING_RATIO,
            k_neighbors=SMOTE_NEIGHBORS,
            random_state=seed,
        )
    elif strategy == "undersampling":
        sampler = RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=seed)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("sampler", sampler),
        ("clf", model),
    ])
    if params:
        # Балансировка задаётся отдельной стратегией, а не скрытым параметром модели.
        if any(not name.startswith("clf__") or name == "clf__class_weight" for name in params):
            raise ValueError("params must contain clf__ parameters; class_weight is set by strategy")
        pipeline.set_params(**params)
    return pipeline


def parameter_grid(model_name):
    """Небольшие сетки для CV только внутри train, одинаковые для всех стратегий."""
    if model_name == "dummy":
        return {}
    if model_name == "logistic_regression":
        return {"clf__C": [0.1, 1.0, 10.0]}
    if model_name == "random_forest":
        return {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 3]}
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
