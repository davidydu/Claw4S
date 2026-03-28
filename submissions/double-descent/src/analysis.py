"""Analysis utilities for double descent experiments.

Functions to detect and quantify the double descent phenomenon
from sweep results.
"""

import hashlib
import json

import numpy as np


def find_interpolation_peak(sweep_results: list[dict]) -> tuple[int, float]:
    """Find the width with the highest test loss (the interpolation peak).

    Args:
        sweep_results: List of dicts with 'width' and 'test_loss' keys.

    Returns:
        (peak_width, peak_test_loss)
    """
    if not sweep_results:
        raise ValueError("Empty sweep results")

    peak = max(sweep_results, key=lambda r: r["test_loss"])
    return peak["width"], peak["test_loss"]


def find_minimum_test_loss(sweep_results: list[dict]) -> tuple[int, float]:
    """Find the width with the lowest test loss.

    Args:
        sweep_results: List of dicts from sweep.

    Returns:
        (best_width, min_test_loss)
    """
    if not sweep_results:
        raise ValueError("Empty sweep results")

    best = min(sweep_results, key=lambda r: r["test_loss"])
    return best["width"], best["test_loss"]


def compute_double_descent_ratio(sweep_results: list[dict]) -> float:
    """Compute peak-to-minimum ratio of test loss.

    A ratio >> 1 indicates a pronounced double descent peak.

    Args:
        sweep_results: List of dicts from sweep.

    Returns:
        peak_test_loss / min_test_loss
    """
    _, peak_loss = find_interpolation_peak(sweep_results)
    _, min_loss = find_minimum_test_loss(sweep_results)

    if min_loss <= 0:
        return float("inf")

    return peak_loss / min_loss


def detect_double_descent(sweep_results: list[dict], threshold: float = 1.5) -> dict:
    """Detect whether a model-wise double descent pattern is present.

    Checks if:
    1. Peak test loss is at an interior width (not boundary).
    2. Test loss recovers (decreases) after the peak.
    3. Peak-to-minimum ratio exceeds threshold.

    Args:
        sweep_results: List of dicts sorted by width.
        threshold: Minimum peak-to-minimum ratio to count as double descent.

    Returns:
        Dict with detection results.
    """
    if len(sweep_results) < 3:
        return {
            "detected": False,
            "message": "Need at least 3 data points",
            "peak_width": 0, "peak_test_loss": 0.0,
            "min_width": 0, "min_test_loss": 0.0,
            "ratio": 0.0,
        }

    sorted_results = sorted(sweep_results, key=lambda r: r["width"])

    peak_width, peak_loss = find_interpolation_peak(sorted_results)
    min_width, min_loss = find_minimum_test_loss(sorted_results)
    ratio = compute_double_descent_ratio(sorted_results)

    first_width = sorted_results[0]["width"]
    last_width = sorted_results[-1]["width"]
    peak_is_interior = (peak_width != first_width and peak_width != last_width)

    # Check that test loss after peak eventually goes below peak
    peak_idx = next(
        i for i, r in enumerate(sorted_results) if r["width"] == peak_width
    )
    post_peak = sorted_results[peak_idx + 1:] if peak_idx < len(sorted_results) - 1 else []
    recovers = any(r["test_loss"] < peak_loss * 0.5 for r in post_peak)

    detected = peak_is_interior and recovers and ratio >= threshold

    message = (
        f"Double descent {'detected' if detected else 'not detected'}. "
        f"Peak at width={peak_width} (test_loss={peak_loss:.4f}), "
        f"min at width={min_width} (test_loss={min_loss:.4f}), "
        f"ratio={ratio:.2f}x."
    )

    return {
        "detected": detected,
        "peak_width": peak_width,
        "peak_test_loss": peak_loss,
        "min_width": min_width,
        "min_test_loss": min_loss,
        "ratio": ratio,
        "message": message,
    }


def detect_epoch_wise_double_descent(
    epochs: list[int],
    test_losses: list[float],
) -> dict:
    """Detect epoch-wise double descent pattern.

    Looks for: decrease -> increase -> decrease in test loss.

    Args:
        epochs: List of epoch numbers.
        test_losses: Corresponding test losses.

    Returns:
        Dict with detection results.
    """
    if len(test_losses) < 5:
        return {
            "detected": False,
            "message": "Need at least 5 data points",
            "first_min_epoch": 0,
            "peak_epoch": 0,
        }

    # Find first local minimum (end of initial descent)
    first_min_idx = 0
    for i in range(1, len(test_losses) - 1):
        if test_losses[i] <= test_losses[i - 1]:
            first_min_idx = i
        else:
            break

    # Find peak after first minimum
    peak_idx = first_min_idx
    for i in range(first_min_idx + 1, len(test_losses)):
        if test_losses[i] > test_losses[peak_idx]:
            peak_idx = i
        elif test_losses[i] < test_losses[peak_idx] * 0.9:
            break

    # Check if there's a descent after the peak
    if peak_idx >= len(test_losses) - 1:
        return {
            "detected": False,
            "message": "No second descent found after peak",
            "first_min_epoch": epochs[first_min_idx],
            "peak_epoch": epochs[peak_idx],
        }

    post_peak_losses = test_losses[peak_idx:]
    second_descent = min(post_peak_losses) < test_losses[peak_idx] * 0.8

    detected = (
        peak_idx > first_min_idx
        and test_losses[peak_idx] > test_losses[first_min_idx]
        and second_descent
    )

    return {
        "detected": detected,
        "first_min_epoch": epochs[first_min_idx],
        "peak_epoch": epochs[peak_idx],
        "message": (
            f"Epoch-wise double descent {'detected' if detected else 'not detected'}. "
            f"First min at epoch {epochs[first_min_idx]}, "
            f"peak at epoch {epochs[peak_idx]}."
        ),
    }


def compute_variance_bands(
    variance_results: list[dict],
) -> dict:
    """Compute mean and std of test loss across seeds for each width.

    Args:
        variance_results: List of dicts with 'seed' and 'results' keys.

    Returns:
        Dict with: widths, mean_test_loss, std_test_loss, n_seeds.
    """
    if not variance_results:
        return {"widths": [], "mean_test_loss": [], "std_test_loss": [], "n_seeds": 0}

    # All seeds should have the same widths
    widths = [r["width"] for r in variance_results[0]["results"]]
    n_widths = len(widths)
    n_seeds = len(variance_results)

    # Collect test losses: shape (n_seeds, n_widths)
    all_losses = np.zeros((n_seeds, n_widths))
    for s_idx, seed_data in enumerate(variance_results):
        for w_idx, result in enumerate(seed_data["results"]):
            all_losses[s_idx, w_idx] = result["test_loss"]

    mean_loss = all_losses.mean(axis=0).tolist()
    std_loss = all_losses.std(axis=0).tolist()

    return {
        "widths": widths,
        "mean_test_loss": mean_loss,
        "std_test_loss": std_loss,
        "n_seeds": n_seeds,
    }


def _round_floats(value, digits: int = 12):
    """Recursively round floating-point values for stable hashing."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_floats(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, digits) for k, v in value.items()}
    return value


def compute_results_fingerprint(all_results: dict) -> str:
    """Compute a stable fingerprint of scientific outputs.

    Excludes volatile timing/hash fields so reruns with identical outputs
    yield the same fingerprint.
    """
    meta = all_results.get("metadata", {})

    canonical = {
        "random_features": all_results.get("random_features", {}),
        "mlp_sweep": all_results.get("mlp_sweep", []),
        "epoch_wise": all_results.get("epoch_wise", {}),
        "variance": all_results.get("variance", []),
        "metadata": {
            "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "d": meta.get("d"),
            "seed": meta.get("seed"),
            "lr": meta.get("lr"),
            "noise_levels": meta.get("noise_levels"),
            "rf_widths": meta.get("rf_widths"),
            "mlp_widths": meta.get("mlp_widths"),
            "rf_interpolation_threshold": meta.get("rf_interpolation_threshold"),
            "mlp_interpolation_threshold": meta.get("mlp_interpolation_threshold"),
            "mlp_epochs": meta.get("mlp_epochs"),
            "epoch_wise_max_epochs": meta.get("epoch_wise_max_epochs"),
            "variance_seeds": meta.get("variance_seeds"),
            "variance_noise_std": meta.get("variance_noise_std"),
        },
    }

    payload = json.dumps(
        _round_floats(canonical),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
