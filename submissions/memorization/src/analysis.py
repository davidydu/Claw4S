# src/analysis.py
"""Analysis: interpolation threshold detection and transition sharpness."""

import numpy as np
from scipy.optimize import curve_fit


def sigmoid(x: np.ndarray, threshold: float, sharpness: float) -> np.ndarray:
    """Sigmoid function for fitting: acc = 1 / (1 + exp(-sharpness * (x - threshold))).

    Args:
        x: Log of parameter counts.
        threshold: Log-param value at 50% accuracy midpoint.
        sharpness: Steepness of the transition (higher = sharper).

    Returns:
        Predicted accuracy values in [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-sharpness * (x - threshold)))


def fit_sigmoid(
    params: list[int],
    accuracies: list[float],
    chance_level: float = 0.1,
) -> dict:
    """Fit sigmoid to train_acc vs log(#params) curve.

    Args:
        params: List of parameter counts.
        accuracies: List of training accuracies (0 to 1).
        chance_level: Chance-level accuracy (1/n_classes).

    Returns:
        Dictionary with threshold, sharpness, r_squared, and fitted values.
    """
    log_params = np.log10(np.array(params, dtype=np.float64))
    accs = np.array(accuracies, dtype=np.float64)

    # Normalize accuracies to [0, 1] range from chance_level
    # so sigmoid maps from chance to 1.0
    accs_normalized = (accs - chance_level) / (1.0 - chance_level)
    accs_normalized = np.clip(accs_normalized, 0.0, 1.0)

    # Initial guesses
    mid_idx = len(log_params) // 2
    p0 = [log_params[mid_idx], 5.0]

    try:
        popt, _ = curve_fit(
            sigmoid,
            log_params,
            accs_normalized,
            p0=p0,
            bounds=([log_params.min() - 1, 0.1], [log_params.max() + 1, 50.0]),
            maxfev=10000,
        )
        threshold_log = popt[0]
        sharpness = popt[1]

        # Compute R^2
        predicted_norm = sigmoid(log_params, *popt)
        predicted = predicted_norm * (1.0 - chance_level) + chance_level
        ss_res = np.sum((accs - predicted) ** 2)
        ss_tot = np.sum((accs - np.mean(accs)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "threshold_log10": float(threshold_log),
            "threshold_params": float(10 ** threshold_log),
            "sharpness": float(sharpness),
            "r_squared": float(r_squared),
            "fitted_accs": predicted.tolist(),
            "fit_success": True,
        }
    except (RuntimeError, ValueError) as e:
        return {
            "threshold_log10": float("nan"),
            "threshold_params": float("nan"),
            "sharpness": float("nan"),
            "r_squared": 0.0,
            "fitted_accs": [],
            "fit_success": False,
            "fit_error": str(e),
        }


def detect_threshold(
    params: list[int],
    accuracies: list[float],
    acc_target: float = 0.99,
) -> dict:
    """Detect interpolation threshold: smallest param count achieving target accuracy.

    Args:
        params: List of parameter counts (sorted ascending).
        accuracies: Corresponding training accuracies.
        acc_target: Accuracy target for "memorized" (default 99%).

    Returns:
        Dictionary with threshold info.
    """
    threshold_params = None
    threshold_idx = None

    for i, (p, a) in enumerate(zip(params, accuracies)):
        if a >= acc_target:
            threshold_params = p
            threshold_idx = i
            break

    return {
        "acc_target": acc_target,
        "threshold_params": threshold_params,
        "threshold_idx": threshold_idx,
        "achieved": threshold_params is not None,
    }


def analyze_results(sweep_results: dict) -> dict:
    """Full analysis of sweep results.

    Args:
        sweep_results: Output from sweep.run_sweep().

    Returns:
        Analysis dictionary with thresholds and sigmoid fits per label type.
    """
    results = sweep_results["results"]
    metadata = sweep_results["metadata"]
    n_train = metadata["n_train"]
    n_classes = metadata["n_classes"]
    chance_level = 1.0 / n_classes

    analysis = {
        "n_train": n_train,
        "chance_level": chance_level,
        "label_types": {},
    }

    for label_type in metadata["label_types"]:
        lt_results = [r for r in results if r["label_type"] == label_type]
        lt_results.sort(key=lambda r: r["n_params"])

        params = [r["n_params"] for r in lt_results]
        train_accs = [r["train_acc"] for r in lt_results]
        test_accs = [r["test_acc"] for r in lt_results]

        # Detect threshold
        threshold = detect_threshold(params, train_accs, acc_target=0.99)

        # Fit sigmoid
        sig_fit = fit_sigmoid(params, train_accs, chance_level=chance_level)

        # Summary statistics
        max_train_acc = max(train_accs)
        mean_test_acc = sum(test_accs) / len(test_accs)
        params_to_samples_ratio = (
            threshold["threshold_params"] / n_train
            if threshold["threshold_params"] is not None
            else None
        )

        analysis["label_types"][label_type] = {
            "params": params,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "threshold": threshold,
            "sigmoid_fit": sig_fit,
            "max_train_acc": max_train_acc,
            "mean_test_acc": mean_test_acc,
            "params_to_samples_ratio": params_to_samples_ratio,
        }

    # Comparative analysis
    random_threshold = analysis["label_types"]["random"]["threshold"]["threshold_params"]
    struct_threshold = analysis["label_types"]["structured"]["threshold"]["threshold_params"]

    if random_threshold is not None and struct_threshold is not None:
        analysis["threshold_ratio"] = random_threshold / struct_threshold
    else:
        analysis["threshold_ratio"] = None

    random_sharpness = analysis["label_types"]["random"]["sigmoid_fit"]["sharpness"]
    struct_sharpness = analysis["label_types"]["structured"]["sigmoid_fit"]["sharpness"]
    analysis["random_sharpness"] = random_sharpness
    analysis["structured_sharpness"] = struct_sharpness

    return analysis
