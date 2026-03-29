"""Main analysis pipeline: train models, fit curves, produce results."""

from datetime import datetime, timezone
import json
import math
import os
import platform
import sys
import time

import matplotlib
import numpy as np
import scipy
import torch

from src.trainer import SEED, train_run
from src.curve_fitting import fit_all_forms
from src.tasks import TASK_REGISTRY

TASKS = list(TASK_REGISTRY.keys())
HIDDEN_SIZES = [32, 64, 128]
N_EPOCHS = 1500
SKIP_EPOCHS = 10  # skip initial transient for fitting


def _analysis_config(
    tasks: list[str],
    hidden_sizes: list[int],
    n_epochs: int,
    skip_epochs: int,
    seed: int,
) -> dict:
    return {
        "tasks": list(tasks),
        "hidden_sizes": list(hidden_sizes),
        "n_epochs": int(n_epochs),
        "skip_epochs": int(skip_epochs),
        "seed": int(seed),
    }


def _build_provenance(seed: int) -> dict:
    """Capture environment metadata needed to reproduce the run."""
    return {
        "seed": int(seed),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_checkpoint(
    checkpoint_path: str,
    runs: list[dict],
    config: dict,
    elapsed_seconds: float,
) -> None:
    """Persist partial results so interrupted runs can resume."""
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "metadata": {
            **config,
            "elapsed_seconds": round(float(elapsed_seconds), 1),
        },
        "runs": runs,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_checkpoint(checkpoint_path: str, expected_config: dict) -> tuple[list[dict], float]:
    """Load checkpoint only when config matches the requested analysis."""
    if not os.path.isfile(checkpoint_path):
        return [], 0.0

    try:
        with open(checkpoint_path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], 0.0

    metadata = payload.get("metadata", {})
    compare_keys = ["tasks", "hidden_sizes", "n_epochs", "skip_epochs", "seed"]
    for key in compare_keys:
        if metadata.get(key) != expected_config.get(key):
            print(
                f"  Checkpoint config mismatch for {key}; starting fresh.",
                flush=True,
            )
            return [], 0.0

    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return [], 0.0

    elapsed = float(metadata.get("elapsed_seconds", 0.0))
    return runs, elapsed


def _support_level(delta_aic: float | None) -> str:
    if delta_aic is None or not math.isfinite(delta_aic):
        return "undetermined"
    if delta_aic >= 10.0:
        return "strong"
    if delta_aic >= 4.0:
        return "moderate"
    return "weak"


def _summarize_fit_support(fits: list[dict]) -> dict:
    """Summarize confidence in the best model using ΔAIC (best vs second)."""
    converged = [fit for fit in fits if fit.get("converged")]
    converged.sort(key=lambda fit: fit.get("aic", float("inf")))

    if not converged:
        return {
            "best_form": "none",
            "second_form": None,
            "delta_aic": None,
            "support_level": "undetermined",
            "n_converged": 0,
        }

    best = converged[0]
    second = converged[1] if len(converged) > 1 else None
    delta_aic = None
    second_form = None
    if second is not None:
        delta_aic = float(second["aic"] - best["aic"])
        second_form = second["form"]

    return {
        "best_form": best["form"],
        "second_form": second_form,
        "delta_aic": None if delta_aic is None else round(delta_aic, 6),
        "support_level": _support_level(delta_aic),
        "n_converged": len(converged),
    }


def run_analysis(
    tasks: list[str] | None = None,
    hidden_sizes: list[int] | None = None,
    n_epochs: int = N_EPOCHS,
    skip_epochs: int = SKIP_EPOCHS,
    seed: int = SEED,
    resume: bool = True,
    checkpoint_path: str = os.path.join("results", "checkpoint.json"),
) -> dict:
    """Run the full analysis: 12 training runs + curve fitting.

    Returns a dict with:
        - runs: list of per-run results
        - universality: summary of best-fit forms and exponent distributions
        - metadata: timing and configuration info
    """
    task_list = list(tasks) if tasks is not None else list(TASKS)
    size_list = list(hidden_sizes) if hidden_sizes is not None else list(HIDDEN_SIZES)
    config = _analysis_config(task_list, size_list, n_epochs, skip_epochs, seed)

    runs: list[dict] = []
    resume_elapsed = 0.0
    if resume:
        runs, resume_elapsed = _load_checkpoint(checkpoint_path, config)

    total = len(task_list) * len(size_list)
    if runs:
        print(
            f"  Resuming from checkpoint: {len(runs)}/{total} runs complete.",
            flush=True,
        )

    # Backfill support metadata for old checkpoints.
    for run in runs:
        if "fit_support" not in run:
            run["fit_support"] = _summarize_fit_support(run.get("fits", []))
        run.setdefault("seed", seed)

    completed = {(r.get("task"), r.get("hidden_size")) for r in runs}
    start_time = time.time()
    idx = 0

    for task_name in task_list:
        for hidden_size in size_list:
            idx += 1
            if (task_name, hidden_size) in completed:
                print(
                    f"[{idx}/{total}] Skipping {task_name} (hidden={hidden_size}) "
                    f"from checkpoint.",
                    flush=True,
                )
                continue

            print(
                f"[{idx}/{total}] Training {task_name} (hidden={hidden_size})...",
                flush=True,
            )
            run_result = train_run(
                task_name=task_name,
                hidden_size=hidden_size,
                n_epochs=n_epochs,
                seed=seed,
            )

            print(
                f"  Final loss: {run_result['final_loss']:.6f} "
                f"({run_result['n_params']} params)",
                flush=True,
            )

            # Fit curves
            fits = fit_all_forms(
                run_result["epochs"],
                run_result["losses"],
                skip_epochs=skip_epochs,
            )
            fit_support = _summarize_fit_support(fits)

            best_fit = fits[0] if fits else None
            best_form = best_fit["form"] if best_fit else "none"
            best_aic = best_fit["aic"] if best_fit else float("inf")

            print(
                f"  Best fit: {best_form} (AIC={best_aic:.1f}, "
                f"support={fit_support['support_level']})",
                flush=True,
            )

            run_entry = {
                "task": task_name,
                "hidden_size": hidden_size,
                "seed": seed,
                "n_params": run_result["n_params"],
                "final_loss": run_result["final_loss"],
                "epochs": run_result["epochs"],
                "losses": run_result["losses"],
                "fits": fits,
                "best_form": best_form,
                "fit_support": fit_support,
            }
            runs.append(run_entry)

            if resume:
                elapsed = resume_elapsed + (time.time() - start_time)
                _save_checkpoint(checkpoint_path, runs, config, elapsed)

    elapsed = resume_elapsed + (time.time() - start_time)

    task_order = {task: i for i, task in enumerate(task_list)}
    size_order = {size: i for i, size in enumerate(size_list)}
    runs.sort(
        key=lambda run: (
            task_order.get(run.get("task"), 10**9),
            size_order.get(run.get("hidden_size"), 10**9),
        )
    )

    # Universality analysis
    universality = _analyze_universality(runs)

    result = {
        "runs": runs,
        "universality": universality,
        "metadata": {
            "tasks": task_list,
            "hidden_sizes": size_list,
            "n_epochs": n_epochs,
            "skip_epochs": skip_epochs,
            "seed": seed,
            "total_runs": total,
            "elapsed_seconds": round(elapsed, 1),
            "provenance": _build_provenance(seed),
        },
    }

    return result


def _analyze_universality(runs: list[dict]) -> dict:
    """Analyze universality of best-fit forms and exponent distributions."""
    # Count best forms
    form_counts: dict[str, int] = {}
    support_counts = {"strong": 0, "moderate": 0, "weak": 0, "undetermined": 0}
    support_by_task: dict[str, dict[str, int]] = {}
    run_support: list[dict] = []

    for run in runs:
        support = run.get("fit_support") or _summarize_fit_support(run.get("fits", []))
        form = run.get("best_form", support["best_form"])
        form_counts[form] = form_counts.get(form, 0) + 1

        level = support.get("support_level", "undetermined")
        if level not in support_counts:
            level = "undetermined"
        support_counts[level] += 1

        task = run["task"]
        if task not in support_by_task:
            support_by_task[task] = {
                "strong": 0,
                "moderate": 0,
                "weak": 0,
                "undetermined": 0,
            }
        support_by_task[task][level] += 1
        run_support.append(
            {
                "task": task,
                "hidden_size": run.get("hidden_size"),
                "best_form": form,
                "second_form": support.get("second_form"),
                "delta_aic": support.get("delta_aic"),
                "support_level": level,
            }
        )

    # Group exponents by task and by form
    exponents_by_task: dict[str, dict[str, list[float]]] = {}
    exponents_by_form: dict[str, dict[str, list[float]]] = {}

    for run in runs:
        task = run["task"]
        for fit in run["fits"]:
            if not fit["converged"]:
                continue
            form = fit["form"]
            params = fit["params"]

            if task not in exponents_by_task:
                exponents_by_task[task] = {}
            if form not in exponents_by_task[task]:
                exponents_by_task[task][form] = []

            if form not in exponents_by_form:
                exponents_by_form[form] = {}

            # Extract key exponent for each form
            key_param = _get_key_exponent(form, params)
            if key_param is not None:
                exponents_by_task[task][form].append(key_param)

                param_name = _get_key_exponent_name(form)
                if param_name not in exponents_by_form[form]:
                    exponents_by_form[form][param_name] = []
                exponents_by_form[form][param_name].append(key_param)

    # Best form per task: select the form with lowest total AIC across all sizes
    best_form_by_task: dict[str, str] = {}
    for task in set(r["task"] for r in runs):
        task_runs = [r for r in runs if r["task"] == task]
        form_aic_totals: dict[str, float] = {}
        for r in task_runs:
            for fit in r["fits"]:
                if not fit["converged"]:
                    continue
                form = fit["form"]
                form_aic_totals[form] = form_aic_totals.get(form, 0.0) + fit["aic"]
        if form_aic_totals:
            best_form_by_task[task] = min(
                form_aic_totals,
                key=lambda form: form_aic_totals[form],
            )

    # Determine majority form
    if form_counts:
        majority_form = max(form_counts, key=lambda k: form_counts[k])
        majority_fraction = form_counts[majority_form] / sum(form_counts.values())
    else:
        majority_form = "none"
        majority_fraction = 0.0

    return {
        "form_counts": form_counts,
        "majority_form": majority_form,
        "majority_fraction": round(majority_fraction, 3),
        "best_form_by_task": best_form_by_task,
        "support_counts": support_counts,
        "support_by_task": support_by_task,
        "run_support": run_support,
        "exponents_by_task": _serialize_exponents(exponents_by_task),
        "exponents_by_form": _serialize_exponents_flat(exponents_by_form),
    }


def _get_key_exponent(form: str, params: dict) -> float | None:
    """Extract the key shape exponent from fitted params."""
    if form == "power_law":
        return params.get("beta")
    elif form == "exponential":
        return params.get("lambda")
    elif form == "stretched_exp":
        return params.get("gamma")
    elif form == "log_power":
        return params.get("beta")
    return None


def _get_key_exponent_name(form: str) -> str:
    if form == "power_law":
        return "beta"
    elif form == "exponential":
        return "lambda"
    elif form == "stretched_exp":
        return "gamma"
    elif form == "log_power":
        return "beta"
    return "unknown"


def _serialize_exponents(d: dict) -> dict:
    """Make exponent dicts JSON-serializable."""
    out = {}
    for task, forms in d.items():
        out[task] = {}
        for form, vals in forms.items():
            out[task][form] = [round(v, 6) for v in vals]
    return out


def _serialize_exponents_flat(d: dict) -> dict:
    out = {}
    for form, params in d.items():
        out[form] = {}
        for pname, vals in params.items():
            out[form][pname] = [round(v, 6) for v in vals]
    return out


def save_results(results: dict, output_dir: str = "results") -> None:
    """Save results to JSON, stripping epoch-level data for compactness."""
    os.makedirs(output_dir, exist_ok=True)

    # Save compact version (without per-epoch data) for validation
    compact = {
        "universality": results["universality"],
        "metadata": results["metadata"],
        "runs_summary": [],
    }
    for run in results["runs"]:
        summary = {
            "task": run["task"],
            "hidden_size": run["hidden_size"],
            "seed": run.get("seed"),
            "n_params": run["n_params"],
            "final_loss": run["final_loss"],
            "best_form": run["best_form"],
            "fit_support": run.get("fit_support"),
            "fits": run["fits"],
        }
        compact["runs_summary"].append(summary)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(compact, f, indent=2)
    print(f"  Saved results/results.json")

    # Save full data (with epochs/losses) for plotting
    full_path = os.path.join(output_dir, "full_curves.json")
    full_data = []
    for run in results["runs"]:
        full_data.append({
            "task": run["task"],
            "hidden_size": run["hidden_size"],
            "epochs": run["epochs"],
            "losses": run["losses"],
            "fits": run["fits"],
        })
    with open(full_path, "w") as f:
        json.dump(full_data, f)
    print(f"  Saved results/full_curves.json")
