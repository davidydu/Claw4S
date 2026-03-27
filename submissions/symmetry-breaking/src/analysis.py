"""Analysis and summary statistics for symmetry-breaking experiments."""

import json
import os
from typing import Dict, List, Any

import numpy as np


def compute_breaking_epoch(
    epochs: List[int],
    symmetry_values: List[float],
    threshold: float = 0.5,
) -> int:
    """Find the first epoch where symmetry drops below a threshold.

    Args:
        epochs: List of epoch numbers at which symmetry was measured.
        symmetry_values: Corresponding symmetry metric values.
        threshold: Symmetry value below which we consider symmetry "broken".

    Returns:
        The epoch at which symmetry first dropped below threshold,
        or -1 if it never did.
    """
    for epoch, sym in zip(epochs, symmetry_values):
        if sym < threshold:
            return epoch
    return -1


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics across all experiment runs.

    Args:
        results: List of per-run result dictionaries from trainer.

    Returns:
        Summary dictionary with key findings.
    """
    summary: Dict[str, Any] = {
        "num_runs": len(results),
        "hidden_dims": sorted(set(r["hidden_dim"] for r in results)),
        "epsilons": sorted(set(r["epsilon"] for r in results)),
        "runs": [],
    }

    for r in results:
        breaking_epoch = compute_breaking_epoch(
            r["epochs_logged"], r["symmetry_values"], threshold=0.5
        )
        run_summary = {
            "hidden_dim": r["hidden_dim"],
            "epsilon": r["epsilon"],
            "initial_symmetry": r["initial_symmetry"],
            "final_symmetry": r["final_symmetry"],
            "symmetry_drop": r["initial_symmetry"] - r["final_symmetry"],
            "breaking_epoch_0.5": breaking_epoch,
            "final_test_acc": r["final_test_acc"],
            "final_train_acc": r["final_train_acc"],
        }
        summary["runs"].append(run_summary)

    # Key finding: does epsilon=0 break symmetry?
    zero_eps_runs = [r for r in summary["runs"] if r["epsilon"] == 0.0]
    if zero_eps_runs:
        summary["zero_eps_final_symmetry_mean"] = float(
            np.mean([r["final_symmetry"] for r in zero_eps_runs])
        )
        summary["zero_eps_test_acc_mean"] = float(
            np.mean([r["final_test_acc"] for r in zero_eps_runs])
        )

    # Key finding: does epsilon>0 break symmetry?
    nonzero_eps_runs = [r for r in summary["runs"] if r["epsilon"] > 0.0]
    if nonzero_eps_runs:
        summary["nonzero_eps_final_symmetry_mean"] = float(
            np.mean([r["final_symmetry"] for r in nonzero_eps_runs])
        )
        summary["nonzero_eps_test_acc_mean"] = float(
            np.mean([r["final_test_acc"] for r in nonzero_eps_runs])
        )

    # Breaking epoch statistics for runs with epsilon >= 1e-4
    substantial_eps_runs = [
        r for r in summary["runs"]
        if r["epsilon"] >= 1e-4 and r["breaking_epoch_0.5"] > 0
    ]
    if substantial_eps_runs:
        breaking_epochs = [r["breaking_epoch_0.5"] for r in substantial_eps_runs]
        summary["median_breaking_epoch"] = float(np.median(breaking_epochs))
        summary["mean_breaking_epoch"] = float(np.mean(breaking_epochs))
        summary["std_breaking_epoch"] = float(np.std(breaking_epochs))

    return summary


def generate_report(
    results: List[Dict[str, Any]], summary: Dict[str, Any]
) -> str:
    """Generate a human-readable report of the experiment results.

    Args:
        results: List of per-run result dicts.
        summary: Summary statistics from summarize_results().

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# Symmetry Breaking in Neural Network Training: Results Report",
        "",
        f"Total runs: {summary['num_runs']}",
        f"Hidden dimensions: {summary['hidden_dims']}",
        f"Epsilon values: {summary['epsilons']}",
        "",
        "## Per-Run Results",
        "",
        "| Hidden Dim | Epsilon | Init Sym | Final Sym | Drop | Breaking Epoch | Test Acc |",
        "|-----------|---------|----------|-----------|------|----------------|----------|",
    ]

    for r in summary["runs"]:
        be = r["breaking_epoch_0.5"]
        be_str = str(be) if be > 0 else "never"
        lines.append(
            f"| {r['hidden_dim']:>9} "
            f"| {r['epsilon']:.1e} "
            f"| {r['initial_symmetry']:.4f}   "
            f"| {r['final_symmetry']:.4f}    "
            f"| {r['symmetry_drop']:.4f} "
            f"| {be_str:>14} "
            f"| {r['final_test_acc']:.4f}   |"
        )

    lines.extend(["", "## Key Findings", ""])

    if "zero_eps_final_symmetry_mean" in summary:
        lines.append(
            f"- **Epsilon=0 (pure symmetric init):** mean final symmetry = "
            f"{summary['zero_eps_final_symmetry_mean']:.4f}, "
            f"mean test accuracy = {summary['zero_eps_test_acc_mean']:.4f}"
        )

    if "nonzero_eps_final_symmetry_mean" in summary:
        lines.append(
            f"- **Epsilon>0 (perturbed init):** mean final symmetry = "
            f"{summary['nonzero_eps_final_symmetry_mean']:.4f}, "
            f"mean test accuracy = {summary['nonzero_eps_test_acc_mean']:.4f}"
        )

    if "median_breaking_epoch" in summary:
        lines.append(
            f"- **Breaking speed (epsilon >= 1e-4):** median breaking epoch = "
            f"{summary['median_breaking_epoch']:.0f}, "
            f"mean = {summary['mean_breaking_epoch']:.0f} "
            f"(std = {summary['std_breaking_epoch']:.0f})"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Symmetry breaking is essential for neural networks to learn diverse",
        "representations. With perfectly symmetric initialization (epsilon=0),",
        "SGD batch noise alone may or may not suffice to break the symmetry,",
        "depending on the interaction of gradients with the symmetric weight",
        "structure. Even a tiny perturbation (epsilon=1e-6) can seed the",
        "divergence of neuron weight vectors, which SGD then amplifies.",
        "",
        "Larger epsilon values lead to faster and more complete symmetry",
        "breaking, and networks with broken symmetry learn more effectively",
        "on the modular addition task, which requires diverse feature",
        "representations across neurons.",
    ])

    return "\n".join(lines) + "\n"


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    report: str,
    output_dir: str = "results",
) -> None:
    """Save results, summary, and report to disk.

    Args:
        results: List of per-run result dicts.
        summary: Summary statistics.
        report: Markdown report string.
        output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report to {report_path}")
