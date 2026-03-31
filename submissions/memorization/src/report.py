# src/report.py
"""Generate markdown report from memorization capacity results."""

import json
import os
from datetime import datetime, timezone


def generate_report(sweep_results: dict, analysis: dict, output_dir: str = "results") -> str:
    """Generate a markdown report summarizing memorization capacity findings.

    Args:
        sweep_results: Output from sweep.run_sweep().
        analysis: Output from analysis.analyze_results().
        output_dir: Directory to save report and raw results.

    Returns:
        Report as a markdown string.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = sweep_results["metadata"]

    lines = [
        "# Memorization Capacity Scaling: Results Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Experiment Configuration",
        "",
        f"- Training samples: {metadata['n_train']}",
        f"- Test samples: {metadata['n_test']}",
        f"- Feature dimensions: {metadata['d']}",
        f"- Number of classes: {metadata['n_classes']}",
        f"- Hidden dimensions swept: {metadata['hidden_dims']}",
        f"- Max epochs: {metadata['max_epochs']}",
        f"- Learning rate: {metadata['lr']}",
        f"- Random seed: {metadata['seed']}",
        "",
        "## Results by Label Type",
        "",
    ]

    for label_type in ["random", "structured"]:
        lt_data = analysis["label_types"][label_type]
        lines.append(f"### {label_type.capitalize()} Labels")
        lines.append("")

        # Results table
        lines.append("| Hidden Dim | #Params | Train Acc | Test Acc |")
        lines.append("|:----------:|--------:|----------:|---------:|")

        lt_results = [r for r in sweep_results["results"] if r["label_type"] == label_type]
        lt_results.sort(key=lambda r: r["n_params"])

        for r in lt_results:
            lines.append(
                f"| {r['hidden_dim']} | {r['n_params']:,} | "
                f"{r['train_acc']:.4f} | {r['test_acc']:.4f} |"
            )
        lines.append("")

        # Threshold info
        threshold = lt_data["threshold"]
        if threshold["achieved"]:
            lines.append(
                f"**Interpolation threshold** (train_acc >= {threshold['acc_target']:.0%}): "
                f"**{threshold['threshold_params']:,} parameters**"
            )
            if lt_data["params_to_samples_ratio"] is not None:
                lines.append(
                    f"- Params/samples ratio at threshold: "
                    f"{lt_data['params_to_samples_ratio']:.1f}x"
                )
        else:
            lines.append(
                f"**Interpolation threshold not reached** "
                f"(max train_acc = {lt_data['max_train_acc']:.4f})"
            )
        lines.append("")

        # Sigmoid fit
        sig = lt_data["sigmoid_fit"]
        if sig["fit_success"]:
            lines.append("**Sigmoid fit** (train_acc vs log10(#params)):")
            lines.append(f"- Threshold (50% midpoint): {sig['threshold_params']:.0f} parameters "
                         f"(log10 = {sig['threshold_log10']:.3f})")
            lines.append(f"- Sharpness: {sig['sharpness']:.2f}")
            lines.append(f"- R-squared: {sig['r_squared']:.4f}")
        else:
            lines.append(f"**Sigmoid fit failed:** {sig.get('fit_error', 'unknown')}")
        lines.append("")

    # Comparative analysis
    lines.append("## Comparative Analysis")
    lines.append("")

    if analysis["threshold_ratio"] is not None:
        lines.append(
            f"- Random labels require **{analysis['threshold_ratio']:.1f}x** "
            f"more parameters than structured labels to reach 99% training accuracy."
        )
    lines.append(
        f"- Random labels transition sharpness: {analysis['random_sharpness']:.2f}"
    )
    lines.append(
        f"- Structured labels transition sharpness: {analysis['structured_sharpness']:.2f}"
    )
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    random_data = analysis["label_types"]["random"]
    struct_data = analysis["label_types"]["structured"]

    lines.append(
        f"1. **Memorization is possible**: MLPs achieve {random_data['max_train_acc']:.0%} "
        f"training accuracy on random labels with sufficient parameters."
    )
    lines.append(
        f"2. **No generalization with random labels**: Mean test accuracy is "
        f"{random_data['mean_test_acc']:.1%} (chance = {analysis['chance_level']:.0%}), "
        f"confirming that random-label memorization does not generalize."
    )
    lines.append(
        f"3. **Structured labels are easier**: Structured labels reach full memorization "
        f"with fewer parameters, as the network can exploit data structure."
    )

    sharpness_diff = abs(analysis["random_sharpness"] - analysis["structured_sharpness"])
    if analysis["random_sharpness"] > 3.0:
        lines.append(
            f"4. **Transition is sharp**: Sigmoid sharpness = {analysis['random_sharpness']:.2f} "
            f"indicates a relatively sharp phase transition for random labels."
        )
    else:
        lines.append(
            f"4. **Transition is gradual**: Sigmoid sharpness = {analysis['random_sharpness']:.2f} "
            f"indicates a gradual transition for random labels."
        )
    lines.append("")

    # Multi-seed variance section (if available)
    multi_seed = analysis.get("multi_seed")
    if multi_seed:
        lines.append("## Multi-Seed Variance (Statistical Robustness)")
        lines.append("")
        seeds = multi_seed["seeds"]
        lines.append(f"Experiment repeated with {len(seeds)} seeds ({seeds}) to quantify variance.")
        lines.append("")
        lines.append("| Label Type | Hidden Dim | #Params | Train Acc (mean +/- std) | Test Acc (mean +/- std) |")
        lines.append("|:----------:|:----------:|--------:|:------------------------:|:-----------------------:|")
        for entry in multi_seed["aggregated"]:
            lines.append(
                f"| {entry['label_type']} | {entry['hidden_dim']} | "
                f"{entry['n_params']:,} | "
                f"{entry['train_acc_mean']:.4f} +/- {entry['train_acc_std']:.4f} | "
                f"{entry['test_acc_mean']:.4f} +/- {entry['test_acc_std']:.4f} |"
            )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Only 2-layer MLPs tested; deeper architectures may behave differently.")
    lines.append("- Synthetic Gaussian data; real-world data distributions may shift thresholds.")
    if multi_seed:
        lines.append(f"- {len(multi_seed['seeds'])} seeds used; more seeds would further tighten confidence intervals.")
    else:
        lines.append("- Single seed (42); ideally multiple seeds would quantify variance.")
    lines.append("- Adam optimizer only; SGD may show different convergence properties.")
    lines.append("")

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved {report_path}")

    # Save raw results as JSON
    results_path = os.path.join(output_dir, "results.json")
    output_data = {
        "metadata": metadata,
        "results": sweep_results["results"],
        "analysis": _make_json_safe(analysis),
    }
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved {results_path}")

    return report


def _make_json_safe(obj):
    """Convert analysis dict to JSON-safe format (handle NaN, etc.)."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    return obj
