"""Generate summary report from experiment results."""

import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np


def generate_report(results: dict) -> str:
    """Generate a markdown report summarizing depth-vs-width findings.

    Args:
        results: Full results dict from run_all_experiments.

    Returns:
        Markdown-formatted report string.
    """
    metadata = results["metadata"]
    experiments = [r for r in results["results"] if not r.get("skipped")]

    lines = []
    lines.append("# Depth vs Width Tradeoff: Experiment Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- Seed: {metadata['seed']}")
    lines.append(f"- Parameter budgets: {metadata['param_budgets']}")
    lines.append(f"- Depths tested: {metadata['depths']}")
    if metadata.get("n_bits"):
        lines.append(f"- Sparse parity: n_bits={metadata['n_bits']}, "
                      f"k_relevant={metadata['k_relevant']}")
    lines.append(f"- PyTorch version: {metadata['torch_version']}")
    lines.append(f"- Total experiments: {metadata['num_experiments']}")
    task_hparams = metadata.get("task_hparams", {})
    if task_hparams:
        lines.append("- Task hyperparameters:")
        for task, hp in task_hparams.items():
            lines.append(f"  - {task}: lr={hp.get('lr')}, "
                         f"wd={hp.get('weight_decay')}, "
                         f"epochs={hp.get('max_epochs')}")
    lines.append("")

    # Group by task
    tasks = sorted(set(r["task_name"] for r in experiments))

    for task in tasks:
        task_results = [r for r in experiments if r["task_name"] == task]
        metric_name = task_results[0]["metric_name"]
        metric_label = (
            "Accuracy" if metric_name == "accuracy" else "R-squared"
        )

        lines.append(f"## Task: {task}")
        lines.append("")

        # Results table
        budgets = sorted(set(r["param_budget"] for r in task_results))
        depths = sorted(set(r["num_hidden_layers"] for r in task_results))

        lines.append(f"### Test {metric_label} by Depth and Budget")
        lines.append("")
        header = "| Depth | " + " | ".join(
            f"{b // 1000}K params" for b in budgets
        ) + " |"
        sep = "|-------|" + "|".join(
            "-----------" for _ in budgets
        ) + "|"
        lines.append(header)
        lines.append(sep)

        for d in depths:
            row = f"| {d:5d} |"
            for b in budgets:
                match = [
                    r for r in task_results
                    if r["num_hidden_layers"] == d and r["param_budget"] == b
                ]
                if match:
                    val = match[0]["best_test_metric"]
                    row += f" {val:9.4f} |"
                else:
                    row += "       N/A |"
            lines.append(row)

        lines.append("")

        # Convergence speed table — use task-specific threshold from metadata
        task_hp = task_hparams.get(task, {})
        conv_thresh = task_hp.get("convergence_threshold", 0.90)
        lines.append(f"### Convergence Speed (Epochs to {metric_label} "
                      f">= {conv_thresh:.2f})")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        for d in depths:
            row = f"| {d:5d} |"
            for b in budgets:
                match = [
                    r for r in task_results
                    if r["num_hidden_layers"] == d and r["param_budget"] == b
                ]
                if match:
                    ce = match[0].get("convergence_epoch")
                    if ce is not None:
                        row += f" {ce:9d} |"
                    else:
                        row += "     never |"
                else:
                    row += "       N/A |"
            lines.append(row)

        lines.append("")

        # Width and actual params table
        lines.append("### Architecture Details (Width / Actual Params)")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        for d in depths:
            row = f"| {d:5d} |"
            for b in budgets:
                match = [
                    r for r in task_results
                    if r["num_hidden_layers"] == d and r["param_budget"] == b
                ]
                if match:
                    w = match[0]["hidden_width"]
                    p = match[0]["actual_params"]
                    row += f" {w}w/{p}p |"
                else:
                    row += "       N/A |"
            lines.append(row)

        lines.append("")

        # Best depth per budget
        lines.append("### Best Depth per Budget")
        lines.append("")
        for b in budgets:
            budget_results = [
                r for r in task_results if r["param_budget"] == b
            ]
            if budget_results:
                tied_best = _select_tied_results(budget_results)
                if len(tied_best) == 1:
                    best = tied_best[0]
                    lines.append(
                        f"- **{b // 1000}K params**: depth={best['num_hidden_layers']}"
                        f" (width={best['hidden_width']}, "
                        f"{metric_label}={best['best_test_metric']:.4f})"
                    )
                else:
                    tied_depths = _format_depths(tied_best)
                    line = (
                        f"- **{b // 1000}K params**: tie among depths="
                        f"{tied_depths} ({metric_label}="
                        f"{tied_best[0]['best_test_metric']:.4f})"
                    )
                    fastest_tied = _fastest_converged(tied_best)
                    if fastest_tied is not None:
                        line += (
                            f"; fastest among tied: depth="
                            f"{fastest_tied['num_hidden_layers']} "
                            f"(epoch {fastest_tied['convergence_epoch']})"
                        )
                    lines.append(line)
        lines.append("")

    # Cross-task analysis
    lines.append("## Cross-Task Analysis")
    lines.append("")

    for budget in sorted(set(r["param_budget"] for r in experiments)):
        lines.append(f"### Budget: {budget // 1000}K params")
        lines.append("")
        for task in tasks:
            match = [
                r for r in experiments
                if r["param_budget"] == budget and r["task_name"] == task
            ]
            if match:
                best = _select_tied_results(match)
                worst = _select_tied_results(match, best=False)
                lines.append(
                    f"- **{task}**: "
                    f"{_format_extreme_summary(best, label='Best')}, "
                    f"{_format_extreme_summary(worst, label='Worst')}"
                )
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Analyze results by task
    finding_num = 1
    for task in tasks:
        task_results = [r for r in experiments if r["task_name"] == task]
        if not task_results:
            continue
        metric_name = task_results[0]["metric_name"]
        metric_label = (
            "Accuracy" if metric_name == "accuracy" else "R-squared"
        )
        tied_best = _select_tied_results(task_results)
        depth_advantage = _compute_depth_advantage(task_results)

        if len(tied_best) == 1:
            best = tied_best[0]
            lines.append(
                f"{finding_num}. **{task}**: Best overall config is "
                f"depth={best['num_hidden_layers']}, "
                f"budget={best['param_budget'] // 1000}K "
                f"({metric_label}={best['best_test_metric']:.4f})"
            )
        else:
            lines.append(
                f"{finding_num}. **{task}**: Best overall {metric_label}="
                f"{tied_best[0]['best_test_metric']:.4f}, achieved by "
                f"{len(tied_best)} configurations across depths "
                f"{_format_depths(tied_best)}."
            )

        if depth_advantage > 0:
            lines.append(
                f"   - By final {metric_name}, deeper models outperform "
                f"shallow/moderate ones on average "
                f"(avg deep-minus-shallow: +{depth_advantage:.4f})"
            )
        elif depth_advantage < 0:
            lines.append(
                f"   - By final {metric_name}, shallow/moderate models "
                f"outperform deeper ones on average "
                f"(avg deep-minus-shallow: {depth_advantage:.4f})"
            )
        else:
            lines.append(
                f"   - By final {metric_name}, shallow/moderate and deeper "
                f"models are tied on average."
            )

        # Convergence speed analysis
        converged = [
            r for r in task_results if r.get("convergence_epoch") is not None
        ]
        if converged:
            fastest = min(converged, key=lambda r: r["convergence_epoch"])
            slowest = max(converged, key=lambda r: r["convergence_epoch"])
            speedup = slowest["convergence_epoch"] / max(
                fastest["convergence_epoch"], 1
            )
            lines.append(
                f"   - Fastest convergence: depth={fastest['num_hidden_layers']}"
                f" at {fastest['param_budget'] // 1000}K "
                f"(epoch {fastest['convergence_epoch']}). "
                f"Slowest converged: depth={slowest['num_hidden_layers']}"
                f" at {slowest['param_budget'] // 1000}K "
                f"(epoch {slowest['convergence_epoch']}). "
                f"Speedup: {speedup:.1f}x"
            )
        finding_num += 1

    # Cross-cutting findings (data-driven from experiments list)
    best_by_config: dict[tuple, list[dict]] = {}
    for task in tasks:
        task_results = [r for r in experiments if r["task_name"] == task]
        for budget in sorted(set(r["param_budget"] for r in task_results)):
            config_results = [
                r for r in task_results if r["param_budget"] == budget
            ]
            best_by_config[(task, budget)] = _select_tied_results(
                config_results
            )

    appearance_counts: dict[int, int] = defaultdict(int)
    for tied_rows in best_by_config.values():
        for depth in {r["num_hidden_layers"] for r in tied_rows}:
            appearance_counts[depth] += 1

    if appearance_counts:
        max_appearances = max(appearance_counts.values())
        leading_depths = sorted(
            depth
            for depth, count in appearance_counts.items()
            if count == max_appearances
        )
        if len(leading_depths) == 1:
            lines.append(
                f"{finding_num}. **Depth {leading_depths[0]} appears in the "
                f"best-performing set for {max_appearances}/"
                f"{len(best_by_config)} configs**, suggesting it is a robust "
                f"default choice."
            )
        else:
            lines.append(
                f"{finding_num}. **Depths {_format_depth_values(leading_depths)} "
                f"appear in the best-performing set for {max_appearances}/"
                f"{len(best_by_config)} configs**, suggesting each is a robust "
                f"default choice."
            )
    finding_num += 1

    # Check if deepest depth fails at smallest budget
    smallest_budget = min(r["param_budget"] for r in experiments)
    deepest = max(r["num_hidden_layers"] for r in experiments)
    deep_small = [r for r in experiments if r["num_hidden_layers"] == deepest and r["param_budget"] == smallest_budget]
    if deep_small:
        avg_metric = sum(r.get("best_test_metric", 0) for r in deep_small) / len(deep_small)
        if avg_metric < 0.8:
            lines.append(
                f"{finding_num}. **Depth {deepest} underperforms at {smallest_budget//1000}K params** "
                f"(avg metric={avg_metric:.3f}), likely due to very narrow layers "
                f"causing optimization instability without skip connections."
            )
            finding_num += 1
    lines.append("")

    return "\n".join(lines)


def _compute_depth_advantage(results: list) -> float:
    """Compute average performance advantage of deep (4,8) over shallow (1,2).

    Positive means deep is better.
    """
    budgets = set(r["param_budget"] for r in results)
    advantages = []
    for b in budgets:
        b_results = [r for r in results if r["param_budget"] == b]
        shallow = [
            r["best_test_metric"] for r in b_results
            if r["num_hidden_layers"] <= 2
        ]
        deep = [
            r["best_test_metric"] for r in b_results
            if r["num_hidden_layers"] >= 4
        ]
        if shallow and deep:
            advantages.append(np.mean(deep) - np.mean(shallow))
    return float(np.mean(advantages)) if advantages else 0.0


def _select_tied_results(
    results: list[dict],
    best: bool = True,
    atol: float = 1e-9,
) -> list[dict]:
    """Return all rows tied for the best or worst metric value."""
    if not results:
        return []

    metric_values = [r["best_test_metric"] for r in results]
    extreme_value = max(metric_values) if best else min(metric_values)
    tied = [
        r for r in results
        if abs(r["best_test_metric"] - extreme_value) <= atol
    ]
    return sorted(tied, key=lambda r: (r["num_hidden_layers"], r["param_budget"]))


def _format_depths(results: list[dict]) -> str:
    """Format the sorted unique depths appearing in result rows."""
    return _format_depth_values(sorted({r["num_hidden_layers"] for r in results}))


def _format_depth_values(depths: list[int]) -> str:
    """Format depth integers for human-readable report text."""
    return ", ".join(str(depth) for depth in depths)


def _fastest_converged(results: list[dict]) -> Optional[dict]:
    """Return the fastest converged row, if any."""
    converged = [r for r in results if r.get("convergence_epoch") is not None]
    if not converged:
        return None
    return min(
        converged,
        key=lambda r: (r["convergence_epoch"], r["num_hidden_layers"]),
    )


def _format_extreme_summary(results: list[dict], label: str) -> str:
    """Format a best/worst summary, preserving metric ties."""
    metric_name = results[0]["metric_name"]
    metric_value = results[0]["best_test_metric"]
    if len(results) == 1:
        return (
            f"{label} depth={results[0]['num_hidden_layers']} "
            f"({metric_name}={metric_value:.4f})"
        )
    return (
        f"{label} depths={_format_depths(results)} "
        f"(tie at {metric_name}={metric_value:.4f})"
    )


def save_report(results: dict, output_dir: str = "results") -> str:
    """Generate and save the report.

    Args:
        results: Full results dict.
        output_dir: Output directory.

    Returns:
        Path to saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = generate_report(results)
    path = os.path.join(output_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Report saved to {path}")
    return path
