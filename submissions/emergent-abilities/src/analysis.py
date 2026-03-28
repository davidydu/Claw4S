"""Core analyses for testing the emergence-as-mirage hypothesis.

Implements four analyses:
1. Metric comparison: discontinuous vs. continuous metrics on the same data
2. Nonlinearity detection: sigmoid vs. linear fits, with MSI computation
3. Synthetic demonstration: linear per-token improvement -> apparent emergence
4. MMLU scaling analysis: smooth scaling across model families
"""

import numpy as np

from src.data import (
    get_bigbench_tasks,
    get_bigbench_data,
    get_bigbench_task_info,
    get_mmlu_data,
    get_model_families,
    MMLU_DATA,
)
from src.metrics import (
    exact_match_from_token_accuracy,
    partial_credit_from_token_accuracy,
    token_edit_distance,
    sigmoid_fit,
    linear_fit,
    compute_aic,
    compute_bic,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def infer_per_token_accuracy(exact_match: float, n_tokens: int) -> float:
    """Infer per-token accuracy from exact-match accuracy.

    Under the token-independence assumption:
        exact_match = p^n  =>  p = exact_match^(1/n)

    This is the inverse of exact_match_from_token_accuracy.

    Args:
        exact_match: Exact-match accuracy in [0, 1].
        n_tokens: Number of tokens in the answer.

    Returns:
        Inferred per-token accuracy in [0, 1].
    """
    if exact_match <= 0.0:
        return 0.0
    if exact_match >= 1.0:
        return 1.0
    return exact_match ** (1.0 / n_tokens)


# ── Analysis 1: Metric Comparison ────────────────────────────────────────────

def compute_metric_comparison(task_name: str) -> dict:
    """Compare discontinuous and continuous metrics for a BIG-Bench task.

    For each model in the task, compute:
    - exact_match: the published accuracy (discontinuous)
    - partial_credit: inferred per-token accuracy (continuous)
    - token_edit_distance: expected edit distance (continuous, lower=better)

    Args:
        task_name: Name of the BIG-Bench task.

    Returns:
        Dict with 'task', 'n_tokens', 'entries' (list of per-model results).
    """
    task_info = get_bigbench_task_info(task_name)
    data = get_bigbench_data(task_name)
    n_tokens = task_info["n_tokens"]

    entries = []
    for d in data:
        p = infer_per_token_accuracy(d["accuracy"], n_tokens)
        entries.append({
            "model": d["model"],
            "family": d["family"],
            "params_b": d["params_b"],
            "exact_match": d["accuracy"],
            "partial_credit": partial_credit_from_token_accuracy(p, n_tokens),
            "token_edit_distance": token_edit_distance(p, n_tokens),
            "inferred_per_token_acc": p,
        })

    return {
        "task": task_name,
        "n_tokens": n_tokens,
        "metric_type": task_info["metric_type"],
        "entries": entries,
    }


# ── Analysis 2: Nonlinearity Detection ──────────────────────────────────────

def compute_nonlinearity_scores() -> dict[str, dict]:
    """Compute nonlinearity scores for all BIG-Bench tasks.

    For each task, fit both linear and sigmoid models to:
    - Discontinuous metric (exact match) vs. log(params)
    - Continuous metric (partial credit) vs. log(params)

    Then compute the Metric Sensitivity Index (MSI):
        MSI = (sigmoid_R2 - linear_R2)_discontinuous / (sigmoid_R2 - linear_R2)_continuous

    High MSI means the nonlinearity is mostly a metric artifact.
    Low MSI (near 1) means genuine nonlinearity.

    Returns:
        Dict mapping task_name -> score dict.
    """
    scores = {}

    for task_name in get_bigbench_tasks():
        comparison = compute_metric_comparison(task_name)
        entries = comparison["entries"]

        if len(entries) < 3:
            continue

        log_params = np.array([np.log10(e["params_b"]) for e in entries])
        exact_matches = np.array([e["exact_match"] for e in entries])
        partial_credits = np.array([e["partial_credit"] for e in entries])

        # Fit models to discontinuous metric
        _, lin_r2_disc, lin_res_disc = linear_fit(log_params, exact_matches)
        _, sig_r2_disc, sig_res_disc = sigmoid_fit(log_params, exact_matches)

        # Fit models to continuous metric
        _, lin_r2_cont, lin_res_cont = linear_fit(log_params, partial_credits)
        _, sig_r2_cont, sig_res_cont = sigmoid_fit(log_params, partial_credits)

        # Compute MSI
        disc_advantage = max(sig_r2_disc - lin_r2_disc, 0.0)
        cont_advantage = max(sig_r2_cont - lin_r2_cont, 0.0)

        if cont_advantage > 1e-6:
            msi = disc_advantage / cont_advantage
        elif disc_advantage > 1e-6:
            msi = float("inf")
        else:
            msi = 1.0  # Both metrics show similar (non-)linearity

        # RSS for AIC/BIC
        rss_lin_disc = float(np.sum(lin_res_disc ** 2))
        rss_sig_disc = float(np.sum(sig_res_disc ** 2))
        rss_lin_cont = float(np.sum(lin_res_cont ** 2))
        rss_sig_cont = float(np.sum(sig_res_cont ** 2))

        # AIC/BIC model comparison (linear: 2 params, sigmoid: 4 params)
        n = len(entries)
        k_lin = 2   # slope, intercept
        k_sig = 4   # L, k, x0, b
        # Guard against zero RSS (perfect fit) to avoid log(0)
        eps = 1e-30
        aic_lin_disc = compute_aic(n, max(rss_lin_disc, eps), k_lin)
        aic_sig_disc = compute_aic(n, max(rss_sig_disc, eps), k_sig)
        bic_lin_disc = compute_bic(n, max(rss_lin_disc, eps), k_lin)
        bic_sig_disc = compute_bic(n, max(rss_sig_disc, eps), k_sig)

        scores[task_name] = {
            "msi": msi,
            "n_tokens": comparison["n_tokens"],
            "metric_type": comparison["metric_type"],
            "linear_r2_discontinuous": lin_r2_disc,
            "sigmoid_r2_discontinuous": sig_r2_disc,
            "linear_r2_continuous": lin_r2_cont,
            "sigmoid_r2_continuous": sig_r2_cont,
            "disc_advantage": disc_advantage,
            "cont_advantage": cont_advantage,
            "rss_linear_disc": rss_lin_disc,
            "rss_sigmoid_disc": rss_sig_disc,
            "rss_linear_cont": rss_lin_cont,
            "rss_sigmoid_cont": rss_sig_cont,
            "aic_linear_disc": aic_lin_disc,
            "aic_sigmoid_disc": aic_sig_disc,
            "bic_linear_disc": bic_lin_disc,
            "bic_sigmoid_disc": bic_sig_disc,
            "sigmoid_preferred_aic": aic_sig_disc < aic_lin_disc,
            "sigmoid_preferred_bic": bic_sig_disc < bic_lin_disc,
            "n_points": n,
        }

    return scores


# ── Analysis 3: Synthetic Demonstration ──────────────────────────────────────

def generate_synthetic_demo(
    seed: int = 42,
    n_points: int = 20,
    n_tokens: int = 5,
    p_min: float = 0.3,
    p_max: float = 0.95,
) -> dict:
    """Generate synthetic data demonstrating the metric artifact.

    Creates data where per-token accuracy improves linearly with
    log(model_size), then shows how exact-match scoring creates
    an apparent phase transition.

    Args:
        seed: Random seed for reproducibility.
        n_points: Number of model sizes to simulate.
        n_tokens: Number of tokens in the answer.
        p_min: Minimum per-token accuracy.
        p_max: Maximum per-token accuracy.

    Returns:
        Dict with arrays: log_params, per_token_acc, exact_match,
        partial_credit, token_edit_distance.
    """
    rng = np.random.default_rng(seed)

    # Simulate model sizes from 100M to 500B
    log_params = np.linspace(-1, 2.7, n_points)  # log10(params_b)

    # Per-token accuracy improves linearly with log(params)
    per_token_acc_true = p_min + (p_max - p_min) * (
        (log_params - log_params[0]) / (log_params[-1] - log_params[0])
    )

    # Add small noise
    noise = rng.normal(0, 0.015, n_points)
    per_token_acc = np.clip(per_token_acc_true + noise, 0.0, 1.0)

    # Compute metrics
    exact_match = np.array([
        exact_match_from_token_accuracy(p, n_tokens) for p in per_token_acc
    ])
    partial_credit = np.array([
        partial_credit_from_token_accuracy(p, n_tokens) for p in per_token_acc
    ])
    ted = np.array([
        token_edit_distance(p, n_tokens) for p in per_token_acc
    ])

    return {
        "log_params": log_params,
        "per_token_acc": per_token_acc,
        "exact_match": exact_match,
        "partial_credit": partial_credit,
        "token_edit_distance": ted,
        "n_tokens": n_tokens,
        "seed": seed,
    }


# ── Analysis 4: MMLU Scaling ────────────────────────────────────────────────

def compute_mmlu_analysis() -> dict:
    """Analyze MMLU scaling across model families.

    MMLU uses multiple-choice accuracy, which is continuous enough
    to show relatively smooth scaling. This contrasts with BIG-Bench
    tasks that use exact string match.

    Returns:
        Dict with per-family fit results and overall statistics.
    """
    data = get_mmlu_data()
    families = get_model_families(data)

    family_results = {}
    for family in families:
        family_data = [d for d in data if d["family"] == family]
        if len(family_data) < 3:
            continue

        log_params = np.array([np.log10(d["params_b"]) for d in family_data])
        accuracies = np.array([d["accuracy"] for d in family_data])

        _, lin_r2, _ = linear_fit(log_params, accuracies)
        _, sig_r2, _ = sigmoid_fit(log_params, accuracies)

        family_results[family] = {
            "models": [d["model"] for d in family_data],
            "params_b": [d["params_b"] for d in family_data],
            "accuracies": [d["accuracy"] for d in family_data],
            "linear_r2": lin_r2,
            "sigmoid_r2": sig_r2,
            "prefers_sigmoid": sig_r2 > lin_r2 + 0.05,
        }

    # Overall analysis across all models
    all_log_params = np.array([np.log10(d["params_b"]) for d in data])
    all_accuracies = np.array([d["accuracy"] for d in data])
    _, overall_lin_r2, _ = linear_fit(all_log_params, all_accuracies)
    _, overall_sig_r2, _ = sigmoid_fit(all_log_params, all_accuracies)

    return {
        "families": family_results,
        "overall_linear_r2": overall_lin_r2,
        "overall_sigmoid_r2": overall_sig_r2,
        "n_models": len(data),
    }


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_full_analysis(seed: int = 42) -> dict:
    """Run all analyses and return combined results.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: metric_comparisons, nonlinearity_scores,
        synthetic_demo, mmlu_analysis.
    """
    np.random.seed(seed)

    # Analysis 1: Metric comparisons for all tasks
    metric_comparisons = {}
    for task_name in get_bigbench_tasks():
        metric_comparisons[task_name] = compute_metric_comparison(task_name)

    # Analysis 2: Nonlinearity scores
    nonlinearity_scores = compute_nonlinearity_scores()

    # Analysis 3: Synthetic demonstration
    synthetic_demo = generate_synthetic_demo(seed=seed)

    # Analysis 4: MMLU scaling
    mmlu_analysis = compute_mmlu_analysis()

    return {
        "metric_comparisons": metric_comparisons,
        "nonlinearity_scores": nonlinearity_scores,
        "synthetic_demo": synthetic_demo,
        "mmlu_analysis": mmlu_analysis,
        "seed": seed,
    }
