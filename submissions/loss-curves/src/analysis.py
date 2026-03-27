"""Main analysis pipeline: train models, fit curves, produce results."""

import json
import os
import time

from src.trainer import train_run
from src.curve_fitting import fit_all_forms
from src.tasks import TASK_REGISTRY

TASKS = list(TASK_REGISTRY.keys())
HIDDEN_SIZES = [32, 64, 128]
N_EPOCHS = 1500
SKIP_EPOCHS = 10  # skip initial transient for fitting


def run_analysis() -> dict:
    """Run the full analysis: 12 training runs + curve fitting.

    Returns a dict with:
        - runs: list of per-run results
        - universality: summary of best-fit forms and exponent distributions
        - metadata: timing and configuration info
    """
    start_time = time.time()
    runs = []
    total = len(TASKS) * len(HIDDEN_SIZES)
    idx = 0

    for task_name in TASKS:
        for hidden_size in HIDDEN_SIZES:
            idx += 1
            print(
                f"[{idx}/{total}] Training {task_name} (hidden={hidden_size})...",
                flush=True,
            )
            run_result = train_run(
                task_name=task_name,
                hidden_size=hidden_size,
                n_epochs=N_EPOCHS,
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
                skip_epochs=SKIP_EPOCHS,
            )

            best_fit = fits[0] if fits else None
            best_form = best_fit["form"] if best_fit else "none"
            best_aic = best_fit["aic"] if best_fit else float("inf")

            print(
                f"  Best fit: {best_form} (AIC={best_aic:.1f})",
                flush=True,
            )

            run_entry = {
                "task": task_name,
                "hidden_size": hidden_size,
                "n_params": run_result["n_params"],
                "final_loss": run_result["final_loss"],
                "epochs": run_result["epochs"],
                "losses": run_result["losses"],
                "fits": fits,
                "best_form": best_form,
            }
            runs.append(run_entry)

    elapsed = time.time() - start_time

    # Universality analysis
    universality = _analyze_universality(runs)

    result = {
        "runs": runs,
        "universality": universality,
        "metadata": {
            "tasks": TASKS,
            "hidden_sizes": HIDDEN_SIZES,
            "n_epochs": N_EPOCHS,
            "skip_epochs": SKIP_EPOCHS,
            "total_runs": total,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    return result


def _analyze_universality(runs: list[dict]) -> dict:
    """Analyze universality of best-fit forms and exponent distributions."""
    # Count best forms
    form_counts: dict[str, int] = {}
    for run in runs:
        form = run["best_form"]
        form_counts[form] = form_counts.get(form, 0) + 1

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
        # Count which form wins (lowest AIC) across sizes for this task
        form_wins: dict[str, int] = {}
        for r in task_runs:
            if r["best_form"]:
                form_wins[r["best_form"]] = form_wins.get(r["best_form"], 0) + 1
        if form_wins:
            best_form_by_task[task] = max(form_wins, key=lambda k: form_wins[k])

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
            "n_params": run["n_params"],
            "final_loss": run["final_loss"],
            "best_form": run["best_form"],
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
