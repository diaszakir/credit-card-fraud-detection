"""Exploratory data analysis: tables, static figures (Matplotlib) and interactive ones (Plotly).

Leakage policy for EDA
----------------------
Anything that relates *features to the target* (correlations, which features separate the
classes, Amount by class) is computed on the **train split only** - the same rule the models
follow, so EDA cannot leak validation/test information into feature choices.
Descriptive facts about the whole file (class balance, time profile, drift between splits) use
all rows because they describe the data, not a model.

Colours follow the validated reference palette: blue = legitimate, orange = fraud (categorical
slots 1 and 2, checked for CVD separation); blue <-> red diverging for signed correlations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; works without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.config import (  # noqa: E402
    AMOUNT_COLUMN,
    EDA_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    ID_COLUMN,
    RAW_COLUMNS,
    RANDOM_SEED,
    ROOT_DIR,
    SECONDS_PER_HOUR,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    TARGET_COLUMN,
    TIME_COLUMN,
    V_COLUMNS,
)
from src.data.audit import iqr_outlier_table  # noqa: E402
from src.data.io_utils import write_json  # noqa: E402
from src.data.splitting import split_summary  # noqa: E402

logger = logging.getLogger(__name__)

# --- design tokens (reference palette, light surface) ----------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
LEGIT = "#2a78d6"  # categorical slot 1
FRAUD = "#eb6834"  # categorical slot 2
NEG, NEUTRAL, POS = "#2a78d6", "#f0efec", "#e34948"  # diverging pair, grey midpoint
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [NEG, NEUTRAL, POS])
CLASS_COLORS = {0: LEGIT, 1: FRAUD}
CLASS_NAMES = {0: "Legitimate", 1: "Fraud"}


def apply_style() -> None:
    """Quiet defaults: hairline grid, no top/right spines, text in ink tokens."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "figure.dpi": 110,
        }
    )


def _relative(path: Path) -> str:
    """Path relative to the repository root when possible (portable in JSON reports)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure %s", path)
    return path


# ------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """describe() + skewness + missing count for the raw columns."""
    table = df[RAW_COLUMNS].astype("float64").describe().T
    table["skew"] = df[RAW_COLUMNS].astype("float64").skew()
    table["missing"] = df[RAW_COLUMNS].isna().sum()
    return table.rename_axis("column").reset_index()


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions and frauds per hour since the first transaction (0 .. 47)."""
    hour = (df[TIME_COLUMN].astype("int64") // SECONDS_PER_HOUR).rename("hour_index")
    grouped = df.groupby(hour)[TARGET_COLUMN].agg(n_transactions="size", n_fraud="sum")
    full = pd.RangeIndex(int(hour.min()), int(hour.max()) + 1, name="hour_index")
    grouped = grouped.reindex(full, fill_value=0).reset_index()
    grouped["fraud_rate_percent"] = 100.0 * grouped["n_fraud"] / grouped["n_transactions"].replace(0, np.nan)
    return grouped


def feature_target_correlation(train: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of every feature with the target, sorted by |r| (train only)."""
    corr = train[FEATURE_COLUMNS].astype("float64").corrwith(train[TARGET_COLUMN].astype("float64"))
    table = corr.rename("correlation").rename_axis("feature").reset_index()
    table["abs_correlation"] = table["correlation"].abs()
    return table.sort_values("abs_correlation", ascending=False, ignore_index=True)


def v_feature_dependence(train: pd.DataFrame) -> dict:
    """How correlated are V1..V28 with each other (off-diagonal |r|)?"""
    corr = train[V_COLUMNS].astype("float64").corr().to_numpy()
    off = np.abs(corr[~np.eye(len(corr), dtype=bool)])
    return {"mean_abs_corr": float(off.mean()), "max_abs_corr": float(off.max()), "n_pairs": int(off.size // 2)}


# ------------------------------------------------------------------------------------------
# Static figures
# ------------------------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, path: Path) -> Path:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.grid(axis="y", visible=False)
    for row, (cls, n) in enumerate(counts.items()):
        ax.barh(row, n, height=0.5, color=CLASS_COLORS[cls])
        ax.text(
            n + total * 0.01, row, f"{n:,}  ({100 * n / total:.4f} %)", va="center", color=INK, fontsize=10
        )
    ax.set_yticks(range(len(counts)), [CLASS_NAMES[c] for c in counts.index])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.32)
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x / 1000)}K" if x else "0")
    ax.set_xlabel("transactions")
    ax.set_title(f"Class balance: {int(counts.get(1, 0)):,} frauds in {total:,} transactions")
    return _save(fig, path)


def plot_amount_by_class(train: pd.DataFrame, path: Path) -> Path:
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    log_amount = np.log10(train[AMOUNT_COLUMN].astype("float64") + 1.0)
    bins = np.linspace(0, float(log_amount.max()), 40)
    for cls in (0, 1):
        values = log_amount[train[TARGET_COLUMN] == cls]
        ax_hist.hist(values, bins=bins, density=True, histtype="step", linewidth=2, color=CLASS_COLORS[cls])
        ax_hist.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.10, color=CLASS_COLORS[cls])
    ax_hist.set_xlabel("log10(1 + Amount)")
    ax_hist.set_ylabel("density (each class sums to 1)")
    ax_hist.set_title("Amount distribution by class (train)")
    ax_hist.legend(
        handles=[
            Line2D([0], [0], color=CLASS_COLORS[c], lw=2,
                   label=f"{CLASS_NAMES[c]} (n={int((train[TARGET_COLUMN] == c).sum()):,})")
            for c in (0, 1)
        ],
        loc="upper right",
    )

    groups = [train.loc[train[TARGET_COLUMN] == c, AMOUNT_COLUMN].astype("float64") for c in (0, 1)]
    box = ax_box.boxplot(
        groups,
        tick_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        patch_artist=True,
        widths=0.45,
        whis=1.5,
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_SECONDARY, "markeredgecolor": "none", "alpha": 0.25},
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": INK_SECONDARY},
        capprops={"color": INK_SECONDARY},
    )
    for patch, cls in zip(box["boxes"], (0, 1)):
        patch.set(facecolor=CLASS_COLORS[cls], alpha=0.35, edgecolor=CLASS_COLORS[cls], linewidth=1.5)
    ax_box.set_yscale("symlog", linthresh=10)
    ax_box.set_ylabel("Amount (symlog scale)")
    ax_box.set_title("Boxplot, IQR rule (train)")
    ax_box.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_time_profile(df: pd.DataFrame, profile: pd.DataFrame, path: Path) -> Path:
    """Two stacked panels with a shared x axis (no dual y axis: counts and rates differ in scale)."""
    fig, (ax_n, ax_rate) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    hours = profile["hour_index"].to_numpy()
    ax_n.bar(hours + 0.5, profile["n_transactions"], width=0.7, color=LEGIT, alpha=0.85)
    ax_n.set_ylabel("transactions per hour")
    ax_n.set_title("Transactions and fraud rate by hour since the first transaction")

    ax_rate.plot(hours + 0.5, profile["fraud_rate_percent"], color=FRAUD, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_rate.set_ylabel("fraud rate, % of the hour")
    ax_rate.set_xlabel("hours since the first transaction (absolute clock time is not given in the data)")
    ax_rate.set_ylim(bottom=0)

    # split boundaries (hairlines + labels), taken from the data
    summary = split_summary(df)
    edges = [(name, summary[name]["time_min"] / SECONDS_PER_HOUR, summary[name]["time_max"] / SECONDS_PER_HOUR)
             for name in SPLIT_NAMES]
    for name, start, end in edges:
        ymax = ax_n.get_ylim()[1]
        ax_n.text((start + end) / 2, ymax * 0.97, name, ha="center", va="top", color=INK_SECONDARY, fontsize=9)
        if name != SPLIT_TRAIN:
            for ax in (ax_n, ax_rate):
                ax.axvline(start, color=INK_SECONDARY, linewidth=1, alpha=0.6)
    ax_n.set_xlim(0, hours.max() + 1)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_target_correlation(table: pd.DataFrame, path: Path, top: int = 15) -> Path:
    top_table = table.head(top).iloc[::-1]  # largest at the top of a horizontal bar chart
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.grid(axis="y", visible=False)
    colors = [POS if r > 0 else NEG for r in top_table["correlation"]]
    ax.barh(top_table["feature"], top_table["correlation"], height=0.6, color=colors)
    for y, value in enumerate(top_table["correlation"]):
        ax.text(value + (0.006 if value > 0 else -0.006), y, f"{value:+.3f}", va="center",
                ha="left" if value > 0 else "right", color=INK_SECONDARY, fontsize=9)
    low, high = top_table["correlation"].min(), top_table["correlation"].max()
    ax.set_xlim(min(low * 1.25, -0.02), max(high * 1.25, 0.02))
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_xlabel("Pearson correlation with Class")
    ax.set_title(f"Top {top} features by |correlation| with fraud (train)")
    ax.legend(
        handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor="none", markersize=9, label=l)
                 for c, l in ((POS, "positive"), (NEG, "negative"))],
        loc="lower right",
    )
    return _save(fig, path)


def plot_correlation_heatmap(train: pd.DataFrame, dependence: dict, path: Path) -> Path:
    cols = [*V_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]
    corr = train[cols].astype("float64").corr()
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    image = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")
    cbar.outline.set_visible(False)
    ax.set_title(
        f"Correlation matrix (train): mean |r| between V features = {dependence['mean_abs_corr']:.3f}"
    )
    return _save(fig, path)


def plot_top_feature_boxplots(train: pd.DataFrame, features: list[str], path: Path) -> Path:
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.1 * n_rows))
    for ax, feature in zip(np.ravel(axes), features):
        groups = [train.loc[train[TARGET_COLUMN] == c, feature].astype("float64") for c in (0, 1)]
        box = ax.boxplot(
            groups, tick_labels=["Legit", "Fraud"], patch_artist=True, widths=0.5, whis=1.5,
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": INK_SECONDARY, "markeredgecolor": "none", "alpha": 0.2},
            medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY},
        )
        for patch, cls in zip(box["boxes"], (0, 1)):
            patch.set(facecolor=CLASS_COLORS[cls], alpha=0.35, edgecolor=CLASS_COLORS[cls], linewidth=1.5)
        ax.set_title(feature, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in np.ravel(axes)[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Most discriminative features by class, IQR boxplots (train)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_split_drift(summary: dict, path: Path) -> Path:
    names = list(SPLIT_NAMES)
    rates = [summary[n]["fraud_rate_percent"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(axis="x", visible=False)
    ax.bar(names, rates, width=0.5, color=FRAUD)
    for x, name in enumerate(names):
        info = summary[name]
        ax.text(x, info["fraud_rate_percent"] + max(rates) * 0.02,
                f"{info['fraud_rate_percent']:.3f} %\n{info['n_fraud']} / {info['n_rows']:,}",
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylim(0, max(rates) * 1.3)
    ax.set_ylabel("fraud rate, % of transactions")
    ax.set_title("Fraud rate by chronological split (frauds / rows)")
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Interactive figures (Plotly)
# ------------------------------------------------------------------------------------------
def _plotly_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15, "color": INK}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_SECONDARY, "size": 12},
        legend={"orientation": "h", "y": 1.08, "x": 0.01},
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)


def interactive_time_profile(profile: pd.DataFrame, path: Path) -> Path:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Transactions per hour", "Fraud rate per hour (%)"))
    fig.add_bar(x=profile["hour_index"], y=profile["n_transactions"], name="Transactions",
                marker_color=LEGIT, customdata=profile[["n_fraud"]],
                hovertemplate="%{y:,} transactions<br>%{customdata[0]} frauds<extra></extra>", row=1, col=1)
    fig.add_scatter(x=profile["hour_index"], y=profile["fraud_rate_percent"], name="Fraud rate (%)",
                    mode="lines+markers", line={"color": FRAUD, "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                    customdata=profile[["n_fraud", "n_transactions"]],
                    hovertemplate="%{y:.3f} %  (%{customdata[0]} of %{customdata[1]:,})<extra></extra>", row=2, col=1)
    fig.update_xaxes(title_text="hours since the first transaction", row=2, col=1)
    fig.update_yaxes(rangemode="tozero")
    _plotly_layout(fig, "Interactive: transactions and fraud rate by hour")
    fig.update_layout(height=620)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def interactive_feature_scatter(train: pd.DataFrame, x_feature: str, y_feature: str, path: Path,
                                n_legit_sample: int = 4000) -> Path:
    """Scatter of two features: ALL train frauds + a random sample of legitimate rows (seeded)."""
    legit = train[train[TARGET_COLUMN] == 0].sample(n=min(n_legit_sample, int((train[TARGET_COLUMN] == 0).sum())),
                                                    random_state=RANDOM_SEED)
    fraud = train[train[TARGET_COLUMN] == 1]
    fig = go.Figure()
    for cls, part in ((0, legit), (1, fraud)):
        fig.add_scattergl(
            x=part[x_feature], y=part[y_feature], mode="markers",
            name=f"{CLASS_NAMES[cls]} ({'sample of ' if cls == 0 else 'all '}{len(part):,})",
            marker={"color": CLASS_COLORS[cls], "size": 7 if cls else 5, "opacity": 0.85 if cls else 0.35,
                    "line": {"color": SURFACE, "width": 1 if cls else 0}},
            customdata=part[[ID_COLUMN, AMOUNT_COLUMN]],
            hovertemplate=f"{x_feature}=%{{x:.2f}}<br>{y_feature}=%{{y:.2f}}<br>id %{{customdata[0]}}, amount %{{customdata[1]:.2f}}<extra></extra>",
        )
    _plotly_layout(fig, f"Interactive: {x_feature} vs {y_feature} (train; legitimate = random sample)")
    fig.update_layout(hovermode="closest", height=560, xaxis_title=x_feature, yaxis_title=y_feature)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


# ------------------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR, tables_dir: Path = EDA_DIR) -> dict:
    """Produce all EDA tables and figures from the processed frame; return a summary dict.

    ``df`` is the processed frame (with ``transaction_id`` and ``split``).
    """
    apply_style()
    figures_dir, tables_dir = Path(figures_dir), Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    train = df[df[SPLIT_COLUMN] == SPLIT_TRAIN]

    # tables -------------------------------------------------------------------------------
    stats = descriptive_stats(df)
    outliers = iqr_outlier_table(df[RAW_COLUMNS])
    target_corr = feature_target_correlation(train)
    profile = hourly_profile(df)
    dependence = v_feature_dependence(train)
    summary = split_summary(df)
    for name, table in (
        ("descriptive_stats", stats),
        ("outliers_iqr", outliers),
        ("feature_target_correlation_train", target_corr),
        ("hourly_profile", profile),
    ):
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    # figures ------------------------------------------------------------------------------
    top_features = target_corr["feature"].head(6).tolist()
    files = {
        "class_balance": plot_class_balance(df, figures_dir / "eda_01_class_balance.png"),
        "amount_by_class": plot_amount_by_class(train, figures_dir / "eda_02_amount_by_class.png"),
        "time_profile": plot_time_profile(df, profile, figures_dir / "eda_03_time_profile.png"),
        "feature_target_correlation": plot_feature_target_correlation(target_corr, figures_dir / "eda_04_feature_target_correlation.png"),
        "correlation_heatmap": plot_correlation_heatmap(train, dependence, figures_dir / "eda_05_correlation_heatmap.png"),
        "top_feature_boxplots": plot_top_feature_boxplots(train, top_features, figures_dir / "eda_06_top_feature_boxplots.png"),
        "split_drift": plot_split_drift(summary, figures_dir / "eda_07_split_drift.png"),
        "interactive_time_profile": interactive_time_profile(profile, figures_dir / "eda_08_interactive_time_profile.html"),
        "interactive_feature_scatter": interactive_feature_scatter(train, top_features[0], top_features[1], figures_dir / "eda_09_interactive_feature_scatter.html"),
    }

    peak = profile.loc[profile["fraud_rate_percent"].idxmax()]
    result = {
        "policy": "feature-target statistics use the train split only",
        "top_features_by_abs_correlation_train": target_corr.head(6)[["feature", "correlation"]].to_dict(orient="records"),
        "v_feature_dependence_train": dependence,
        "hour_with_highest_fraud_rate": {
            "hour_index": int(peak["hour_index"]),
            "fraud_rate_percent": float(peak["fraud_rate_percent"]),
            "n_transactions": int(peak["n_transactions"]),
            "n_fraud": int(peak["n_fraud"]),
        },
        "hours_with_fewer_than_100_transactions": int((profile["n_transactions"] < 100).sum()),
        "split_summary": summary,
        "files": {k: _relative(v) for k, v in files.items()},
    }
    write_json(result, tables_dir / "eda_summary.json")
    return result
