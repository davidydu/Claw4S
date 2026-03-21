"""Publication-quality matplotlib figures for scaling law analysis.

Generates 5 PNG files from the results dict produced by run_full_analysis().
All figures use seaborn-v0_8-whitegrid style (falls back to default if unavailable).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import CEREBRAS_GPT, get_family_data, get_training_tokens
from src.scaling_models import (
    kaplan_loss,
    chinchilla_loss,
    corrected_loss,
)

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"
_COLORS = {
    "kaplan": "#1f77b4",      # blue
    "chinchilla": "#ff7f0e",  # orange
    "corrected": "#2ca02c",   # green
    "data": "#333333",        # dark grey for data points
    "train": "#2ca02c",       # green for training points
    "actual": "#1f77b4",      # blue for actual test values
    "predicted": "#d62728",   # red for predicted values
}


def _use_style() -> None:
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("default")


def _savefig(path: str) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Helper: dense N range for smooth curves
# ---------------------------------------------------------------------------

def _dense_n(n_data: np.ndarray, n_points: int = 200) -> np.ndarray:
    return np.logspace(np.log10(n_data.min()), np.log10(n_data.max()), n_points)


# ---------------------------------------------------------------------------
# Figure 1: Loss scaling (log-log)
# ---------------------------------------------------------------------------

def _plot_loss_scaling(results: dict, output_dir: str) -> None:
    """Log-log plot of Pile test loss vs model parameters.

    Plots the three scaling formulations (Kaplan, Chinchilla, corrected)
    as smooth curves over data points from Cerebras-GPT.
    """
    _use_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Actual data points
    n_data, y_data = get_family_data(CEREBRAS_GPT, "pile_test_loss")
    d_data = get_training_tokens(CEREBRAS_GPT)

    ax.scatter(
        n_data, y_data,
        color=_COLORS["data"], zorder=5, s=60,
        label="Cerebras-GPT (data)", marker="o",
    )

    loss_scaling = results.get("loss_scaling", {})
    n_dense = _dense_n(n_data)
    # Dense training tokens scaled proportionally (D ≈ 20N for Cerebras-GPT)
    d_ratio = d_data / n_data  # ~20 for all sizes
    d_dense = n_dense * float(np.mean(d_ratio))

    formulation_funcs = {
        "kaplan": lambda n_arr, params: kaplan_loss(
            n_arr, params["a"], params["alpha"], params["l_inf"]
        ),
        "chinchilla": lambda n_arr, params: chinchilla_loss(
            n_arr, d_dense,
            params["a"], params["alpha"], params["b"], params["beta"], params["l_inf"],
        ),
        "corrected": lambda n_arr, params: corrected_loss(
            n_arr, params["a"], params["alpha"], params["c"], params["gamma"], params["l_inf"]
        ),
    }

    for form_name, color in [
        ("kaplan", _COLORS["kaplan"]),
        ("chinchilla", _COLORS["chinchilla"]),
        ("corrected", _COLORS["corrected"]),
    ]:
        form_result = loss_scaling.get(form_name, {})
        params = form_result.get("params", {})
        adj_r2 = form_result.get("adj_r_squared", float("nan"))

        if params and not any(
            isinstance(v, float) and (np.isnan(v) or np.isinf(v))
            for v in params.values()
        ):
            try:
                y_curve = formulation_funcs[form_name](n_dense, params)
                r2_str = f"{adj_r2:.4f}" if not np.isnan(adj_r2) else "N/A"
                ax.plot(
                    n_dense, y_curve, color=color, linewidth=2,
                    label=f"{form_name.capitalize()} (adj-R²={r2_str})",
                )
            except Exception as exc:
                print(f"  WARNING: Could not plot curve: {type(exc).__name__}: {exc}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (N)", fontsize=12)
    ax.set_ylabel("Pile Test Loss", fontsize=12)
    ax.set_title("Loss Scaling: Cerebras-GPT", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    _savefig(os.path.join(output_dir, "loss_scaling.png"))


# ---------------------------------------------------------------------------
# Figure 2: Task scaling (2×4 subplot grid)
# ---------------------------------------------------------------------------

def _plot_task_scaling(results: dict, output_dir: str) -> None:
    """2×4 subplot grid — one subplot per benchmark task.

    Each subplot shows data points and the bounded-power-law fitted curve
    on a log-linear scale.
    """
    _use_style()
    task_scaling = results.get("task_scaling", {})
    tasks = list(task_scaling.keys())

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, task in enumerate(tasks[:8]):
        ax = axes_flat[idx]
        task_result = task_scaling[task]

        # Actual data
        n_data, y_data = get_family_data(CEREBRAS_GPT, task)
        ax.scatter(n_data, y_data, color=_COLORS["data"], zorder=5, s=50, marker="o")

        # Bounded power-law fitted curve
        bpl = task_result.get("bounded_power_law", {})
        adj_r2 = bpl.get("adj_r_squared", float("nan"))
        params = bpl.get("params", {})
        converged = bpl.get("converged", False)

        if converged and params:
            a = params.get("a", float("nan"))
            alpha = params.get("alpha", float("nan"))
            if not (np.isnan(a) or np.isnan(alpha)):
                n_dense = _dense_n(n_data)
                y_curve = 1.0 - a * np.power(n_dense, -alpha)
                ax.plot(n_dense, y_curve, color=_COLORS["kaplan"], linewidth=2)

        r2_str = f"{adj_r2:.3f}" if not np.isnan(adj_r2) else "N/A"
        task_label = task.replace("_acc", "").replace("_", " ").title()
        ax.set_title(f"{task_label}\n(adj-R²={r2_str})", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (N)", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Task Scaling: Cerebras-GPT", fontsize=14, y=1.01)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, "task_scaling.png"))


# ---------------------------------------------------------------------------
# Figure 3: Residuals (side-by-side)
# ---------------------------------------------------------------------------

def _plot_residuals(results: dict, output_dir: str) -> None:
    """Side-by-side residual plots.

    Left: residuals of the best loss scaling formulation vs model size.
    Right: average task scaling residuals vs model size.
    """
    _use_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    n_data, y_loss = get_family_data(CEREBRAS_GPT, "pile_test_loss")

    # --- Left: Loss residuals (best formulation by adj-R²) ---
    loss_scaling = results.get("loss_scaling", {})
    best_form = None
    best_r2 = -np.inf
    for form_name, form_result in loss_scaling.items():
        r2 = form_result.get("adj_r_squared", float("nan"))
        if not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_form = form_name

    if best_form is not None:
        form_result = loss_scaling[best_form]
        params = form_result.get("params", {})
        d_data = get_training_tokens(CEREBRAS_GPT)
        d_ratio = float(np.mean(d_data / n_data))

        try:
            if best_form == "kaplan":
                y_pred = kaplan_loss(n_data, params["a"], params["alpha"], params["l_inf"])
            elif best_form == "chinchilla":
                y_pred = chinchilla_loss(
                    n_data, d_data,
                    params["a"], params["alpha"], params["b"], params["beta"], params["l_inf"],
                )
            elif best_form == "corrected":
                y_pred = corrected_loss(
                    n_data, params["a"], params["alpha"], params["c"], params["gamma"], params["l_inf"],
                )
            else:
                y_pred = np.full_like(y_loss, np.nan)
            residuals_loss = y_loss - y_pred
        except Exception as exc:
            print(f"  WARNING: Could not plot curve: {type(exc).__name__}: {exc}")
            residuals_loss = np.full_like(y_loss, np.nan)
    else:
        residuals_loss = np.full_like(y_loss, np.nan)
        best_form = "N/A"

    ax_left.scatter(n_data, residuals_loss, color=_COLORS["kaplan"], s=80, zorder=5)
    ax_left.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_left.set_xscale("log")
    ax_left.set_xlabel("Model Parameters (N)", fontsize=11)
    ax_left.set_ylabel("Residual (Actual − Predicted)", fontsize=11)
    ax_left.set_title(f"Loss Residuals\n(best formulation: {best_form})", fontsize=12)
    ax_left.grid(True, which="both", alpha=0.3)

    # --- Right: Average task scaling residuals ---
    task_scaling = results.get("task_scaling", {})
    all_residuals = []
    for task, task_result in task_scaling.items():
        n_t, acc = get_family_data(CEREBRAS_GPT, task)
        bpl = task_result.get("bounded_power_law", {})
        params = bpl.get("params", {})
        converged = bpl.get("converged", False)
        if converged and params:
            a = params.get("a", float("nan"))
            alpha = params.get("alpha", float("nan"))
            if not (np.isnan(a) or np.isnan(alpha)):
                y_pred_t = 1.0 - a * np.power(n_t, -alpha)
                all_residuals.append(acc - y_pred_t)

    if all_residuals:
        avg_residuals = np.mean(all_residuals, axis=0)
        n_task, _ = get_family_data(CEREBRAS_GPT, list(task_scaling.keys())[0])
    else:
        avg_residuals = np.full(len(n_data), np.nan)
        n_task = n_data

    ax_right.scatter(n_task, avg_residuals, color=_COLORS["corrected"], s=80, zorder=5)
    ax_right.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Model Parameters (N)", fontsize=11)
    ax_right.set_ylabel("Avg Residual (Actual − Predicted)", fontsize=11)
    ax_right.set_title("Task Scaling Residuals\n(avg across all benchmarks)", fontsize=12)
    ax_right.grid(True, which="both", alpha=0.3)

    fig.suptitle("Residual Analysis: Loss vs Task Scaling", fontsize=14)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, "residuals.png"))


# ---------------------------------------------------------------------------
# Figure 4: Model selection (grouped bar chart)
# ---------------------------------------------------------------------------

def _plot_model_selection(results: dict, output_dir: str) -> None:
    """Grouped bar chart comparing AIC and BIC across the 3 loss formulations."""
    _use_style()
    loss_scaling = results.get("loss_scaling", {})

    formulations = ["kaplan", "chinchilla", "corrected"]
    aic_vals = [loss_scaling.get(f, {}).get("aic", float("nan")) for f in formulations]
    bic_vals = [loss_scaling.get(f, {}).get("bic", float("nan")) for f in formulations]

    x = np.arange(len(formulations))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_aic = ax.bar(
        x - bar_width / 2, aic_vals, bar_width,
        label="AIC", color=_COLORS["kaplan"], alpha=0.85, edgecolor="white",
    )
    bars_bic = ax.bar(
        x + bar_width / 2, bic_vals, bar_width,
        label="BIC", color=_COLORS["chinchilla"], alpha=0.85, edgecolor="white",
    )

    # Annotate bars with values
    for bar in bars_aic:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.05,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9,
            )
    for bar in bars_bic:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.05,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in formulations], fontsize=11)
    ax.set_xlabel("Formulation", fontsize=12)
    ax.set_ylabel("Criterion Value (lower is better)", fontsize=12)
    ax.set_title("Model Selection: AIC vs BIC", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    _savefig(os.path.join(output_dir, "model_selection.png"))


# ---------------------------------------------------------------------------
# Figure 5: Extrapolation risk
# ---------------------------------------------------------------------------

def _plot_extrapolation(results: dict, output_dir: str) -> None:
    """Prediction vs actual for large models (extrapolation from small models).

    Left: loss extrapolation.
    Right: task accuracy extrapolation (average across benchmarks).
    """
    _use_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    extrapolation = results.get("extrapolation", {})
    n_data, y_loss = get_family_data(CEREBRAS_GPT, "pile_test_loss")

    # We replicate the same train/test split used in run_extrapolation_risk (n_train=4)
    n_train_split = 4
    n_tr = n_data[:n_train_split]
    y_tr = y_loss[:n_train_split]
    n_te = n_data[n_train_split:]
    y_te = y_loss[n_train_split:]

    # --- Left: Loss extrapolation ---
    loss_preds = extrapolation.get("loss_predictions", {})
    loss_mape = extrapolation.get("loss_mape", float("nan"))

    ax_left.scatter(
        n_tr, y_tr,
        color=_COLORS["train"], s=80, marker="o", zorder=5,
        label="Training points",
    )

    pred_n = []
    pred_actual = []
    pred_hat = []
    for key, val in loss_preds.items():
        pred_n.append(float(key))
        pred_actual.append(val["actual"])
        pred_hat.append(val["predicted"])

    if pred_n:
        pred_n = np.array(pred_n)
        pred_actual = np.array(pred_actual)
        pred_hat = np.array(pred_hat)

        ax_left.scatter(
            pred_n, pred_actual,
            color=_COLORS["actual"], s=80, marker="o", zorder=6,
            label="Test (actual)",
        )
        ax_left.scatter(
            pred_n, pred_hat,
            color=_COLORS["predicted"], s=80, marker="X", zorder=6,
            label="Test (predicted)",
        )
        # Connect actual-predicted pairs with thin lines
        for nx, act, hat in zip(pred_n, pred_actual, pred_hat):
            ax_left.plot(
                [nx, nx], [act, hat],
                color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
            )

    mape_str = f"{loss_mape:.1f}%" if not np.isnan(loss_mape) else "N/A"
    ax_left.set_xscale("log")
    ax_left.set_xlabel("Model Parameters (N)", fontsize=11)
    ax_left.set_ylabel("Pile Test Loss", fontsize=11)
    ax_left.set_title(f"Loss Extrapolation\n(MAPE={mape_str})", fontsize=12)
    ax_left.legend(fontsize=10)
    ax_left.grid(True, which="both", alpha=0.3)

    # --- Right: Task accuracy extrapolation (average across benchmarks) ---
    task_preds = extrapolation.get("task_predictions", {})
    task_mape_avg = extrapolation.get("task_mape_avg", float("nan"))

    # Collect average actual and predicted across benchmarks per model size
    avg_actual: dict[str, list[float]] = {}
    avg_hat: dict[str, list[float]] = {}
    for bench, bench_preds in task_preds.items():
        for key, val in bench_preds.items():
            avg_actual.setdefault(key, []).append(val["actual"])
            avg_hat.setdefault(key, []).append(val["predicted"])

    # Also collect training points (use first available benchmark)
    first_bench = next(iter(task_preds.keys())) if task_preds else None
    if first_bench:
        n_t, acc_t = get_family_data(CEREBRAS_GPT, first_bench)
        n_tr_t = n_t[:n_train_split]
        acc_tr_t = acc_t[:n_train_split]
        ax_right.scatter(
            n_tr_t, acc_tr_t,
            color=_COLORS["train"], s=80, marker="o", zorder=5,
            label="Training points",
        )

    if avg_actual:
        te_n = np.array([float(k) for k in sorted(avg_actual.keys(), key=float)])
        te_act = np.array([float(np.mean(avg_actual[k])) for k in sorted(avg_actual.keys(), key=float)])
        te_hat = np.array([float(np.mean(avg_hat[k])) for k in sorted(avg_hat.keys(), key=float)])

        ax_right.scatter(
            te_n, te_act,
            color=_COLORS["actual"], s=80, marker="o", zorder=6,
            label="Test (actual avg)",
        )
        ax_right.scatter(
            te_n, te_hat,
            color=_COLORS["predicted"], s=80, marker="X", zorder=6,
            label="Test (predicted avg)",
        )
        for nx, act, hat in zip(te_n, te_act, te_hat):
            ax_right.plot(
                [nx, nx], [act, hat],
                color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
            )

    task_mape_str = f"{task_mape_avg:.1f}%" if not np.isnan(task_mape_avg) else "N/A"
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Model Parameters (N)", fontsize=11)
    ax_right.set_ylabel("Accuracy (avg across benchmarks)", fontsize=11)
    ax_right.set_title(f"Task Accuracy Extrapolation\n(Avg MAPE={task_mape_str})", fontsize=12)
    ax_right.legend(fontsize=10)
    ax_right.grid(True, which="both", alpha=0.3)

    fig.suptitle("Extrapolation Risk", fontsize=14)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, "extrapolation.png"))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def generate_all_plots(results: dict, output_dir: str = "results/figures") -> None:
    """Generate all 5 publication-quality figures from analysis results.

    Creates the following PNG files in output_dir:
      - loss_scaling.png    : log-log plot of loss vs params (3 formulations)
      - task_scaling.png    : 2x4 grid of per-benchmark scaling curves
      - residuals.png       : loss vs task residual comparison
      - model_selection.png : grouped AIC/BIC bar chart
      - extrapolation.png   : train-on-small, predict-large comparison

    Args:
        results: dict returned by run_full_analysis().
        output_dir: directory where PNG files are written (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[plots] Writing figures to '{output_dir}/'")

    print("[plots] 1/5 loss_scaling.png ...")
    _plot_loss_scaling(results, output_dir)

    print("[plots] 2/5 task_scaling.png ...")
    _plot_task_scaling(results, output_dir)

    print("[plots] 3/5 residuals.png ...")
    _plot_residuals(results, output_dir)

    print("[plots] 4/5 model_selection.png ...")
    _plot_model_selection(results, output_dir)

    print("[plots] 5/5 extrapolation.png ...")
    _plot_extrapolation(results, output_dir)

    print("[plots] Done — 5 figures saved.")
