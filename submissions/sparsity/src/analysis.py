"""Main analysis: run all experiments and collect results.

Runs 8 training experiments (4 hidden widths x 2 tasks) and computes
correlations between sparsity metrics and generalization.
"""

import time

import torch
from scipy import stats

from src.models import create_model
from src.data import make_modular_addition_data, make_regression_data
from src.trainer import train_with_tracking


# Experiment configuration
HIDDEN_WIDTHS = [32, 64, 128, 256]
N_EPOCHS = 3000
TRACK_EVERY = 50
SEED = 42

# Per-task hyperparameters
# Modular addition: high LR + strong weight decay for grokking regime
MOD_ADD_LR = 1e-2
MOD_ADD_WD = 1.0

# Regression: moderate LR + weight decay
REG_LR = 1e-2
REG_WD = 0.1
N_BOOTSTRAP = 800


CORRELATION_SPECS = [
    ("dead_frac_vs_gen_gap", "final_dead_frac", "gen_gap"),
    ("dead_frac_vs_test_acc", "final_dead_frac", "final_test_acc"),
    ("zero_frac_vs_gen_gap", "final_zero_frac", "gen_gap"),
    ("zero_frac_vs_test_acc", "final_zero_frac", "final_test_acc"),
    ("zero_frac_change_vs_test_acc", "zero_frac_change", "final_test_acc"),
    ("sparsity_change_vs_test_acc", "sparsity_change", "final_test_acc"),
]


def _bootstrap_spearman_ci(
    x: list[float],
    y: list[float],
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Estimate a 95% bootstrap CI for Spearman rho."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0

    generator = torch.Generator().manual_seed(seed)
    bootstrapped_rhos = []

    for _ in range(n_bootstrap):
        sample_idx = torch.randint(0, n, (n,), generator=generator).tolist()
        x_sample = [x[i] for i in sample_idx]
        y_sample = [y[i] for i in sample_idx]
        if len(set(x_sample)) <= 1 or len(set(y_sample)) <= 1:
            continue
        rho, _ = stats.spearmanr(x_sample, y_sample)
        if rho == rho:  # filter NaN
            bootstrapped_rhos.append(float(rho))

    if not bootstrapped_rhos:
        return 0.0, 0.0

    bootstrapped_rhos.sort()
    low_idx = int(0.025 * (len(bootstrapped_rhos) - 1))
    high_idx = int(0.975 * (len(bootstrapped_rhos) - 1))
    return bootstrapped_rhos[low_idx], bootstrapped_rhos[high_idx]


def _compute_correlation_table(
    summaries: list[dict],
    *,
    seed: int,
) -> dict:
    """Compute all configured Spearman correlations for a summary subset."""
    correlations = {}
    for idx, (name, x_key, y_key) in enumerate(CORRELATION_SPECS):
        x_vals = [s[x_key] for s in summaries]
        y_vals = [s[y_key] for s in summaries]

        if len(set(x_vals)) > 1 and len(set(y_vals)) > 1:
            rho, p_val = stats.spearmanr(x_vals, y_vals)
            ci_low, ci_high = _bootstrap_spearman_ci(
                x_vals,
                y_vals,
                seed=seed + idx,
            )
            correlations[name] = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n": len(x_vals),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        else:
            correlations[name] = {
                "rho": 0.0,
                "p_value": 1.0,
                "n": len(x_vals),
                "ci_low": 0.0,
                "ci_high": 0.0,
            }

    return correlations


def run_single_experiment(
    task_type: str,
    hidden_dim: int,
    data: dict,
    lr: float,
    weight_decay: float,
    seed: int = SEED,
) -> dict:
    """Run one training experiment and return results.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    hidden_dim : int
        Hidden layer width.
    data : dict
        Dataset from make_modular_addition_data or make_regression_data.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Experiment configuration and training history.
    """
    model = create_model(
        input_dim=data["input_dim"],
        hidden_dim=hidden_dim,
        output_dim=data["output_dim"],
        seed=seed,
    )

    history = train_with_tracking(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        task_type=task_type,
        n_epochs=N_EPOCHS,
        lr=lr,
        weight_decay=weight_decay,
        track_every=TRACK_EVERY,
        seed=seed,
    )

    return {
        "task": data["task_name"],
        "task_type": task_type,
        "hidden_dim": hidden_dim,
        "n_epochs": N_EPOCHS,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "history": history,
    }


def compute_sparsity_generalization_correlation(experiments: list) -> dict:
    """Compute correlation between final sparsity and generalization gap.

    For each experiment, extracts:
    - Final dead neuron fraction and near-dead fraction
    - Generalization gap (train_acc - test_acc at end of training)
    - Final test accuracy

    Then computes Spearman correlations with 95% bootstrap confidence
    intervals, both pooled and task-stratified.

    Parameters
    ----------
    experiments : list of dict
        List of experiment results from run_single_experiment.

    Returns
    -------
    dict
        Correlation statistics (pooled and per-task) and per-experiment summaries.
    """
    summaries = []
    for exp in experiments:
        h = exp["history"]
        final_idx = -1
        summary = {
            "task": exp["task"],
            "hidden_dim": exp["hidden_dim"],
            "final_dead_frac": h["dead_neuron_fraction"][final_idx],
            "final_near_dead_frac": h["near_dead_fraction"][final_idx],
            "final_zero_frac": h["zero_fraction"][final_idx],
            "final_entropy": h["activation_entropy"][final_idx],
            "final_mean_mag": h["mean_activation_magnitude"][final_idx],
            "final_train_acc": h["train_acc"][final_idx],
            "final_test_acc": h["test_acc"][final_idx],
            "gen_gap": h["train_acc"][final_idx] - h["test_acc"][final_idx],
            "initial_dead_frac": h["dead_neuron_fraction"][0],
            "initial_zero_frac": h["zero_fraction"][0],
            "sparsity_change": (
                h["dead_neuron_fraction"][final_idx]
                - h["dead_neuron_fraction"][0]
            ),
            "zero_frac_change": (
                h["zero_fraction"][final_idx]
                - h["zero_fraction"][0]
            ),
        }
        summaries.append(summary)

    correlations = _compute_correlation_table(summaries, seed=SEED)

    # Also compute task-stratified correlations to reduce pooled-task confounding.
    correlations_by_task = {}
    tasks = sorted({s["task"] for s in summaries})
    for task_idx, task in enumerate(tasks):
        task_summaries = [s for s in summaries if s["task"] == task]
        correlations_by_task[task] = _compute_correlation_table(
            task_summaries,
            seed=SEED + 1000 + task_idx * 100,
        )

    return {
        "experiment_summaries": summaries,
        "correlations": correlations,
        "correlations_by_task": correlations_by_task,
    }


def detect_grokking_sparsity_transition(experiment: dict) -> dict:
    """Detect whether grokking coincides with a sparsity transition.

    Looks for a sharp increase in test accuracy (grokking) and checks
    whether it coincides with a change in activation sparsity metrics.

    Parameters
    ----------
    experiment : dict
        Single experiment result.

    Returns
    -------
    dict
        Grokking detection results.
    """
    h = experiment["history"]
    epochs = h["epochs"]
    test_accs = h["test_acc"]
    dead_fracs = h["dead_neuron_fraction"]
    zero_fracs = h["zero_fraction"]
    train_accs = h["train_acc"]

    # Detect grokking: epoch where test acc first exceeds 0.8
    # while train acc was already > 0.95
    grokking_epoch = None
    for i, (ep, ta, tra) in enumerate(zip(epochs, test_accs, train_accs)):
        if ta > 0.8 and tra > 0.95 and i > 2:
            # Check that test acc was low before (within 5 steps)
            prev_idx = max(0, i - 5)
            if test_accs[prev_idx] < 0.4:
                grokking_epoch = ep
                break

    if grokking_epoch is None:
        # Check for partial grokking: significant test acc improvement
        max_test = max(test_accs)
        min_test = min(test_accs[:len(test_accs) // 2 + 1])
        return {
            "grokking_detected": False,
            "grokking_epoch": None,
            "max_test_acc": max_test,
            "sparsity_at_grokking": None,
            "sparsity_before_grokking": None,
            "sparsity_transition_detected": False,
            "zero_frac_initial": zero_fracs[0] if zero_fracs else None,
            "zero_frac_final": zero_fracs[-1] if zero_fracs else None,
        }

    # Find index of grokking epoch
    grok_idx = epochs.index(grokking_epoch)
    pre_grok_idx = max(0, grok_idx - 5)

    sparsity_at = dead_fracs[grok_idx]
    sparsity_before = dead_fracs[pre_grok_idx]
    sparsity_change = abs(sparsity_at - sparsity_before)

    zero_at = zero_fracs[grok_idx]
    zero_before = zero_fracs[pre_grok_idx]
    zero_change = abs(zero_at - zero_before)

    return {
        "grokking_detected": True,
        "grokking_epoch": grokking_epoch,
        "max_test_acc": max(test_accs),
        "sparsity_at_grokking": sparsity_at,
        "sparsity_before_grokking": sparsity_before,
        "sparsity_transition_detected": sparsity_change > 0.03 or zero_change > 0.03,
        "sparsity_change_magnitude": sparsity_change,
        "zero_frac_at_grokking": zero_at,
        "zero_frac_before_grokking": zero_before,
        "zero_frac_change": zero_change,
    }


def run_all_experiments() -> dict:
    """Run the full experiment suite.

    8 experiments: 4 hidden widths x 2 tasks (modular addition, regression).

    Returns
    -------
    dict
        Full results including experiments, correlations, grokking analysis.
    """
    print("[1/4] Generating datasets...")
    mod_data = make_modular_addition_data(seed=SEED)
    reg_data = make_regression_data(seed=SEED)

    experiments = []
    total = len(HIDDEN_WIDTHS) * 2
    run_idx = 0

    print(f"[2/4] Running {total} training experiments...")
    t0 = time.time()

    for hidden_dim in HIDDEN_WIDTHS:
        for task_name, task_type, data, lr, wd in [
            ("modular_addition", "classification", mod_data, MOD_ADD_LR, MOD_ADD_WD),
            ("regression", "regression", reg_data, REG_LR, REG_WD),
        ]:
            run_idx += 1
            print(f"  [{run_idx}/{total}] {task_name} h={hidden_dim} lr={lr} wd={wd}...",
                  end=" ", flush=True)
            t1 = time.time()
            result = run_single_experiment(
                task_type=task_type,
                hidden_dim=hidden_dim,
                data=data,
                lr=lr,
                weight_decay=wd,
                seed=SEED,
            )
            elapsed = time.time() - t1
            h = result["history"]
            final_dead = h["dead_neuron_fraction"][-1]
            final_zero = h["zero_fraction"][-1]
            final_test = h["test_acc"][-1]
            print(f"done ({elapsed:.1f}s) dead={final_dead:.3f} "
                  f"zero_frac={final_zero:.3f} test_acc={final_test:.3f}")
            experiments.append(result)

    total_time = time.time() - t0
    print(f"  Total training time: {total_time:.1f}s")

    print("[3/4] Computing correlations...")
    correlation_results = compute_sparsity_generalization_correlation(experiments)

    print("[4/4] Analyzing grokking-sparsity transitions...")
    grokking_results = []
    for exp in experiments:
        if "modular_addition" in exp["task"]:
            gr = detect_grokking_sparsity_transition(exp)
            gr["hidden_dim"] = exp["hidden_dim"]
            grokking_results.append(gr)

    return {
        "experiments": experiments,
        "correlations": correlation_results["correlations"],
        "correlations_by_task": correlation_results["correlations_by_task"],
        "experiment_summaries": correlation_results["experiment_summaries"],
        "grokking_analysis": grokking_results,
        "config": {
            "hidden_widths": HIDDEN_WIDTHS,
            "n_epochs": N_EPOCHS,
            "track_every": TRACK_EVERY,
            "mod_add_lr": MOD_ADD_LR,
            "mod_add_wd": MOD_ADD_WD,
            "reg_lr": REG_LR,
            "reg_wd": REG_WD,
            "seed": SEED,
        },
    }
