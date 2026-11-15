"""Графики оценки: Plotly для приложения, Matplotlib (PNG) для отчёта и презентации.

Командная строка (сохраняет PNG по реальным validation-прогнозам выбранной модели):
    python -m src.evaluation.plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Палитра: три категориальных слота (проверены на различимость при дальтонизме) + нейтральные.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, GRID, TEXT, TEXT_2, SURFACE = "#a3a29c", "#e6e5e0", "#0b0b0b", "#52514e", "#fcfcfb"
MODEL_COLORS = {"logistic_regression": BLUE, "random_forest": ORANGE}


def _translucent(hex_color: str, alpha: float = 0.35) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
MODEL_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "dummy": "Dummy"}
STRATEGY_NAMES = {"none": "без балансировки", "class_weight": "веса классов", "smote": "SMOTE",
                  "undersampling": "undersampling"}


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": TEXT}, "x": 0, "xanchor": "left"},
        height=height, margin={"l": 60, "r": 30, "t": 50, "b": 50},
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font={"color": TEXT_2, "size": 12},
        hovermode="closest", legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    axis = {"showgrid": True, "gridcolor": GRID, "gridwidth": 1, "zeroline": False,
            "linecolor": GRID, "ticks": "outside", "tickcolor": GRID}
    fig.update_xaxes(title=x_title, **axis)
    fig.update_yaxes(title=y_title, **axis)
    return fig


def _reference_h(fig, y: float, label: str) -> None:
    fig.add_hline(y=y, line={"color": MUTED, "width": 1},
                  annotation={"text": label, "font": {"color": TEXT_2, "size": 11}}, annotation_position="bottom left")


def pr_figure(curve: pd.DataFrame, point: dict, min_precision: float, prevalence: float, ap: float | None = None) -> go.Figure:
    fig = go.Figure()
    name = "PR-кривая" + (f" (AP {ap:.3f})" if ap is not None else "")
    fig.add_trace(go.Scatter(
        x=curve["recall"], y=curve["precision"], mode="lines", name=name, line={"color": BLUE, "width": 2},
        customdata=curve["threshold"],
        hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<br>порог %{customdata:.4f}<extra></extra>",
    ))
    _add_point(fig, point["recall"], point["precision"], point["threshold"], BLUE)
    _reference_h(fig, min_precision, f"требование precision ≥ {min_precision:.2f}")
    _reference_h(fig, prevalence, f"доля fraud {prevalence:.4f} (случайный классификатор)")
    fig.update_xaxes(range=[0, 1.02])
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, "Precision–Recall, validation", "Recall", "Precision")


def _add_point(fig, x, y, threshold, color) -> None:
    if x is None or y is None:
        return
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", name=f"Сохранённый порог {threshold:.4f}",
        marker={"size": 11, "color": color, "line": {"color": SURFACE, "width": 2}},
        text=[f"P {y:.3f} · R {x:.3f}"], textposition="bottom left", textfont={"color": TEXT, "size": 12},
        hovertemplate=f"порог {threshold:.4f}<br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>",
    ))


def threshold_figure(curve: pd.DataFrame, threshold: float, min_precision: float) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (("precision", "Precision", BLUE), ("recall", "Recall", ORANGE), ("f1", "F1", AQUA)):
        fig.add_trace(go.Scatter(
            x=curve["threshold"], y=curve[column], mode="lines", name=label, line={"color": color, "width": 2},
            hovertemplate=f"порог %{{x:.4f}}<br>{label} %{{y:.3f}}<extra></extra>",
        ))
        last = curve.iloc[-1]
        fig.add_annotation(x=last["threshold"], y=last[column], text=label, showarrow=False, xanchor="left",
                           xshift=6, font={"color": TEXT, "size": 12})
    fig.add_vline(x=threshold, line={"color": TEXT_2, "width": 1},
                  annotation={"text": f"сохранённый порог {threshold:.4f}", "font": {"color": TEXT_2, "size": 11}},
                  annotation_position="top right")
    _reference_h(fig, min_precision, f"precision ≥ {min_precision:.2f}")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(range=[0, 1.08])
    return _layout(fig, "Precision, recall и F1 в зависимости от порога, validation", "Порог score", "Значение")


def roc_figure(roc: pd.DataFrame, point: dict, roc_auc: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="случайный классификатор",
                             line={"color": MUTED, "width": 1}, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=roc["fpr"], y=roc["tpr"], mode="lines", line={"color": BLUE, "width": 2},
        name="ROC" + (f" (AUC {roc_auc:.3f})" if roc_auc is not None else ""),
        hovertemplate="FPR %{x:.4f}<br>TPR %{y:.3f}<extra></extra>",
    ))
    if point.get("recall") is not None:
        fig.add_trace(go.Scatter(x=[point["fpr"]], y=[point["recall"]], mode="markers", name="сохранённый порог",
                                 marker={"size": 11, "color": BLUE, "line": {"color": SURFACE, "width": 2}},
                                 hovertemplate="FPR %{x:.5f}<br>TPR %{y:.3f}<extra></extra>"))
    return _layout(fig, "ROC, validation", "False positive rate", "True positive rate (recall)")


def operating_points_figure(points: pd.DataFrame, selected_run_id: str | None, min_precision: float) -> go.Figure:
    """Рабочие точки всех запусков с 95% интервалами Уилсона по обеим осям."""
    fig = go.Figure()
    for model, group in points.groupby("model_name", sort=False):
        labels = [f"{MODEL_NAMES.get(model, model)} · {STRATEGY_NAMES.get(s, s)} · {t}"
                  for s, t in zip(group["imbalance_strategy"], group["tuning"])]
        fig.add_trace(go.Scatter(
            x=group["recall"], y=group["precision"], mode="markers", name=MODEL_NAMES.get(model, model),
            marker={"size": 10, "color": MODEL_COLORS.get(model, MUTED), "line": {"color": SURFACE, "width": 2},
                    "symbol": "circle" if model == "random_forest" else "diamond"},
            error_x={"type": "data", "symmetric": False, "array": group["recall_high"] - group["recall"],
                     "arrayminus": group["recall"] - group["recall_low"],
                     "color": _translucent(MODEL_COLORS.get(model, MUTED)), "thickness": 1, "width": 0},
            error_y={"type": "data", "symmetric": False, "array": group["precision_high"] - group["precision"],
                     "arrayminus": group["precision"] - group["precision_low"],
                     "color": _translucent(MODEL_COLORS.get(model, MUTED)), "thickness": 1, "width": 0},
            customdata=list(zip(labels, group["f1"], group["average_precision"])),
            hovertemplate="%{customdata[0]}<br>precision %{y:.3f}<br>recall %{x:.3f}"
                          "<br>F1 %{customdata[1]:.3f} · AP %{customdata[2]:.3f}<extra></extra>",
        ))
    selected = points[points["run_id"] == selected_run_id]
    if not selected.empty:
        row = selected.iloc[0]
        fig.add_annotation(x=row["recall"], y=row["precision"], text="выбранная модель", showarrow=True,
                           arrowcolor=TEXT_2, ax=40, ay=-40, font={"color": TEXT, "size": 12})
    _reference_h(fig, min_precision, f"precision ≥ {min_precision:.2f}")
    return _layout(fig, "Рабочие точки запусков на validation (линии — 95% ДИ)", "Recall", "Precision", height=460)


def ap_figure(comparison: pd.DataFrame, selected_run_id: str | None) -> go.Figure:
    table = comparison.sort_values("average_precision")
    labels = [f"{MODEL_NAMES.get(m, m)} · {STRATEGY_NAMES.get(s, s)} · {t}"
              for m, s, t in zip(table["model_name"], table["imbalance_strategy"], table["tuning"])]
    colors = [BLUE if run == selected_run_id else MUTED for run in table["run_id"]]
    fig = go.Figure(go.Bar(
        x=table["average_precision"], y=labels, orientation="h", marker={"color": colors},
        text=[f"{v:.3f}" for v in table["average_precision"]], textposition="outside",
        textfont={"color": TEXT, "size": 11}, hovertemplate="%{y}<br>AP %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(bargap=0.45, showlegend=False)
    fig.update_xaxes(range=[0, 1.08])
    return _layout(fig, "Average precision всех запусков, validation (синий — выбранная модель)",
                   "Average precision", "", height=max(320, 26 * len(table) + 110))


# --------------------------------------------------------------------------- #
# PNG для отчёта (Matplotlib, без интерактивности)
# --------------------------------------------------------------------------- #
def _mpl_axes(title, x_title, y_title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel(x_title, color=TEXT_2)
    ax.set_ylabel(y_title, color=TEXT_2)
    ax.tick_params(colors=TEXT_2)
    return fig, ax


def save_report_figures(root=None, allow_non_real: bool = False) -> list[Path]:
    """Построить PNG по реальным validation-прогнозам выбранной модели и comparison.csv.

    allow_non_real нужен только тестам на синтетической фикстуре.
    """
    import matplotlib.pyplot as plt

    from src.evaluation import curves, sources
    from src.evaluation.metrics import evaluate_selected_validation

    paths = sources.project_paths(root)
    result = evaluate_selected_validation(root)  # сверка с Диасом внутри; без неё графики не строятся
    metadata = sources.load_model_metadata(paths).data
    if metadata.get("provenance") != "real" and not allow_non_real:
        raise ValueError("графики для отчёта строятся только по модели с provenance='real'")
    table = sources.load_validation_predictions(paths, metadata).data
    y, score = table["y_true"].to_numpy(), table["score"].to_numpy()
    threshold, min_precision = metadata["threshold"], metadata["min_precision"]
    point = curves.operating_point(y, score, threshold)
    out = paths.evaluation_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    pr = curves.pr_curve(y, score)
    fig, ax = _mpl_axes("Precision–Recall, validation", "Recall", "Precision")
    ax.plot(pr["recall"], pr["precision"], color=BLUE, linewidth=2, label=f"AP {result['average_precision']:.3f}")
    ax.axhline(min_precision, color=MUTED, linewidth=1)
    ax.text(0.01, min_precision - 0.05, f"precision ≥ {min_precision:.2f}", color=TEXT_2, fontsize=9)
    ax.scatter([point["recall"]], [point["precision"]], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"порог {threshold:.4f}\nP {point['precision']:.3f} · R {point['recall']:.3f}",
                (point["recall"], point["precision"]), xytext=(-110, -40), textcoords="offset points",
                color=TEXT, fontsize=9, arrowprops={"arrowstyle": "-", "color": TEXT_2})
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    written.append(_save(fig, out / "pr_curve_validation.png", plt))

    thr = curves.threshold_curve(y, score)
    fig, ax = _mpl_axes("Precision, recall и F1 от порога, validation", "Порог score", "Значение")
    for column, label, color in (("precision", "Precision", BLUE), ("recall", "Recall", ORANGE), ("f1", "F1", AQUA)):
        ax.plot(thr["threshold"], thr[column], color=color, linewidth=2, label=label)
    ax.axvline(threshold, color=TEXT_2, linewidth=1)
    ax.text(threshold + 0.01, 1.07, f"сохранённый порог {threshold:.4f}", color=TEXT_2, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    written.append(_save(fig, out / "threshold_curve_validation.png", plt))

    roc = curves.roc_points(y, score)
    fig, ax = _mpl_axes("ROC, validation", "False positive rate", "True positive rate")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax.plot(roc["fpr"], roc["tpr"], color=BLUE, linewidth=2, label=f"ROC-AUC {result['roc_auc']:.3f}")
    ax.legend(frameon=False, loc="lower right")
    written.append(_save(fig, out / "roc_curve_validation.png", plt))

    comparison = sources.load_comparison(paths)
    if comparison.available:
        points = curves.run_operating_points(comparison.data)
        fig, ax = _mpl_axes("Рабочие точки запусков, validation (линии — 95% ДИ)", "Recall", "Precision")
        for model, group in points.groupby("model_name", sort=False):
            ax.errorbar(group["recall"], group["precision"],
                        xerr=[group["recall"] - group["recall_low"], group["recall_high"] - group["recall"]],
                        yerr=[group["precision"] - group["precision_low"], group["precision_high"] - group["precision"]],
                        fmt="o" if model == "random_forest" else "D", color=MODEL_COLORS.get(model, MUTED),
                        ecolor=_translucent_rgb(MODEL_COLORS.get(model, MUTED)), elinewidth=1, markersize=6,
                        markeredgecolor=SURFACE, label=MODEL_NAMES.get(model, model))
        chosen = points[points["run_id"] == metadata.get("run_id")]
        if not chosen.empty:
            row = chosen.iloc[0]
            ax.scatter([row["recall"]], [row["precision"]], s=160, facecolor="none", edgecolor=TEXT, linewidth=1.5, zorder=4)
            ax.annotate("выбранная модель", (row["recall"], row["precision"]), xytext=(-120, 30),
                        textcoords="offset points", color=TEXT, fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": TEXT_2})
        ax.axhline(min_precision, color=MUTED, linewidth=1)
        ax.legend(frameon=False, loc="lower left")
        written.append(_save(fig, out / "operating_points_validation.png", plt))
    return written


def _translucent_rgb(hex_color: str, alpha: float = 0.35):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (alpha,)


def _save(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    argparse.ArgumentParser(description="PNG-графики оценки выбранной модели на validation").parse_args(argv)
    for path in save_report_figures():
        print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    from src.evaluation.cli_utils import run_cli

    raise SystemExit(run_cli(main))
