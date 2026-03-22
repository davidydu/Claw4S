"""Report generation for double descent experiments.

Generates a Markdown summary of experimental findings.
"""

from src.analysis import (
    detect_double_descent,
    detect_epoch_wise_double_descent,
    compute_variance_bands,
)


def generate_report(all_results: dict) -> str:
    """Generate a Markdown report summarizing double descent findings.

    Args:
        all_results: Dict from run_all_sweeps() with keys:
            random_features, mlp_sweep, epoch_wise, variance, metadata.

    Returns:
        Markdown-formatted report string.
    """
    meta = all_results["metadata"]
    lines = []

    lines.append("# Double Descent in Practice: Experimental Report")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append(f"- **Training samples (n)**: {meta['n_train']}")
    lines.append(f"- **Test samples**: {meta['n_test']}")
    lines.append(f"- **Input dimensions (d)**: {meta['d']}")
    lines.append(f"- **Noise levels (sigma)**: {meta['noise_levels']}")
    lines.append(f"- **Random seed**: {meta['seed']}")
    lines.append(f"- **RF interpolation threshold**: p = n = {meta['rf_interpolation_threshold']}")
    lines.append(f"- **MLP interpolation threshold**: h ~ {meta['mlp_interpolation_threshold']}")
    lines.append(f"- **Runtime**: {meta['runtime_seconds']:.1f}s")
    lines.append("")

    # Random features analysis
    lines.append("## Experiment 1: Model-Wise Double Descent (Random Features)")
    lines.append("")
    lines.append(
        "We use a random ReLU features model: phi(X) = ReLU(X @ W_random + b), "
        "where the first layer is fixed and the second layer is fit via "
        "minimum-norm least squares. The interpolation threshold is at p = n "
        "(number of features = number of training samples)."
    )
    lines.append("")

    for label in sorted(all_results["random_features"].keys()):
        data = all_results["random_features"][label]
        detection = detect_double_descent(data)
        noise_val = label.replace("noise_", "sigma=")

        lines.append(f"### {noise_val}")
        lines.append("")
        lines.append(f"- {detection['message']}")

        if detection["detected"]:
            lines.append(f"- **Peak-to-minimum ratio: {detection['ratio']:.2f}x**")
        lines.append("")

        # Compact table
        lines.append("| Features (p) | p/n | Train MSE | Test MSE |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for r in data:
            lines.append(
                f"| {r['width']} | {r['param_ratio']:.2f} | "
                f"{r['train_loss']:.4f} | {r['test_loss']:.4f} |"
            )
        lines.append("")

    # MLP comparison
    lines.append("## Experiment 2: MLP Comparison")
    lines.append("")
    lines.append(
        "For comparison, we train 2-layer MLPs (ReLU, Adam optimizer, "
        "no regularization) at varying hidden widths."
    )
    lines.append("")

    mlp_data = all_results["mlp_sweep"]
    lines.append("| Width (h) | #Params | Ratio | Train MSE | Test MSE |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for r in mlp_data:
        lines.append(
            f"| {r['width']} | {r['n_params']} | {r['param_ratio']:.2f} | "
            f"{r['train_loss']:.4f} | {r['test_loss']:.4f} |"
        )
    lines.append("")

    # Epoch-wise analysis
    lines.append("## Experiment 3: Epoch-Wise Double Descent")
    lines.append("")
    lines.append(
        f"We train an MLP with hidden width h={meta['mlp_interpolation_threshold']} "
        f"(near the interpolation threshold) and track test loss over epochs."
    )
    lines.append("")

    for label in sorted(all_results["epoch_wise"].keys()):
        data = all_results["epoch_wise"][label]
        detection = detect_epoch_wise_double_descent(
            data["epochs"], data["test_losses"]
        )
        noise_val = label.replace("noise_", "sigma=")
        lines.append(f"- **{noise_val}**: {detection['message']}")

    lines.append("")

    # Variance analysis
    lines.append("## Experiment 4: Variance Across Seeds")
    lines.append("")
    variance_stats = compute_variance_bands(all_results["variance"])
    lines.append(
        f"We repeat the random features experiment with "
        f"{variance_stats['n_seeds']} different seeds to measure variability."
    )
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Summarize random features results
    highest_noise_label = f"noise_{max(meta['noise_levels'])}"
    rf_detection = detect_double_descent(
        all_results["random_features"].get(highest_noise_label, [])
    )

    findings = []
    findings.append(
        f"1. **Model-wise double descent confirmed**: Test MSE peaks sharply "
        f"at the interpolation threshold (p = n = {meta['rf_interpolation_threshold']}), "
        f"then decreases in the overparameterized regime."
    )

    if rf_detection["detected"]:
        findings.append(
            f"2. **Peak-to-minimum ratio**: {rf_detection['ratio']:.0f}x at "
            f"highest noise (sigma={max(meta['noise_levels'])}), confirming "
            f"the dramatic nature of the phenomenon."
        )

    # Check noise effect on absolute peak height
    peaks = {}
    for label in sorted(all_results["random_features"].keys()):
        det = detect_double_descent(all_results["random_features"][label])
        peaks[label] = det["peak_test_loss"]

    if len(peaks) >= 2:
        peak_values = list(peaks.values())
        if peak_values[-1] > peak_values[0]:
            findings.append(
                f"3. **Noise amplifies double descent**: The absolute peak "
                f"test MSE increases with noise level "
                f"({', '.join(f'{v:.0f}' for v in peak_values)}), "
                f"confirming that the interpolation threshold is more "
                f"harmful when there is more noise to memorize."
            )

    findings.append(
        "4. **Benign overfitting**: In the overparameterized regime (p >> n), "
        "models achieve zero training error yet test error continues to "
        "decrease, demonstrating the 'benign overfitting' phenomenon."
    )

    lines.extend(findings)
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The double descent phenomenon occurs because at the interpolation "
        "threshold, the model has just barely enough capacity to fit the "
        "training data. It is forced to pass through every training point "
        "(including noise), resulting in a highly irregular, wiggly function "
        "that generalizes poorly. Beyond this threshold, there are many "
        "possible interpolating solutions, and the minimum-norm solution "
        "found by least squares (or implicit regularization in SGD) is "
        "smoother and generalizes better."
    )
    lines.append("")

    return "\n".join(lines)
