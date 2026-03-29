"""Phase transition detection and cross-correlation lag analysis.

Detects phase transitions in gradient norm and test metric trajectories,
then computes the lag between them to test whether gradient norm transitions
precede generalization transitions.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def smooth(signal: list[float], window: int = 51, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing. Adjusts window if signal is short."""
    arr = np.array(signal, dtype=np.float64)
    n = len(arr)
    # Window must be <= signal length and odd
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window % 2 == 0:
        window += 1
    # polyorder must be < window
    if window <= polyorder:
        return arr
    return savgol_filter(arr, window, polyorder)


def detect_transition_epoch(
    signal: list[float],
    epochs: list[int],
    direction: str = "increase",
    smooth_window: int = 51,
) -> dict:
    """Detect the epoch of steepest change in a signal.

    Uses the derivative of the smoothed signal to find the point of
    maximum rate of change.

    Args:
        signal: time series values.
        epochs: corresponding epoch numbers.
        direction: 'increase' to find steepest rise, 'decrease' for steepest fall.
        smooth_window: Savitzky-Golay window size.

    Returns:
        dict with 'transition_epoch', 'transition_idx', 'derivative' array.
    """
    smoothed = smooth(signal, window=smooth_window)
    derivative = np.gradient(smoothed)

    if direction == "increase":
        idx = int(np.argmax(derivative))
    else:
        idx = int(np.argmin(derivative))

    return {
        "transition_epoch": epochs[idx],
        "transition_idx": idx,
        "derivative": derivative,
        "smoothed": smoothed,
    }


def compute_gradient_norm_rate(
    grad_norms: dict[str, list[float]],
) -> list[float]:
    """Compute combined gradient norm rate of change across layers.

    Returns the L2 norm of all per-layer gradient norms at each time step.
    """
    layer_names = list(grad_norms.keys())
    n = len(grad_norms[layer_names[0]])
    combined = np.zeros(n, dtype=np.float64)
    for name in layer_names:
        combined += np.array(grad_norms[name], dtype=np.float64) ** 2
    return (combined ** 0.5).tolist()


def cross_correlation_lag(
    signal_a: list[float],
    signal_b: list[float],
    max_lag: int = 500,
) -> dict:
    """Compute cross-correlation between two signals and find optimal lag.

    Tests whether signal_a leads signal_b (positive lag means A precedes B).

    Args:
        signal_a: the signal hypothesized to lead (gradient norm derivative).
        signal_b: the signal hypothesized to lag (test metric derivative).
        max_lag: maximum lag in number of samples to test.

    Returns:
        dict with 'best_lag', 'best_correlation', 'lags', 'correlations'.
    """
    a = np.array(signal_a, dtype=np.float64)
    b = np.array(signal_b, dtype=np.float64)

    # Normalize
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    n = len(a)
    max_lag = min(max_lag, n // 2)
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []

    for lag in lags:
        if lag >= 0:
            corr_val = np.mean(a[:n - lag] * b[lag:]) if lag < n else 0.0
        else:
            corr_val = np.mean(a[-lag:] * b[:n + lag]) if -lag < n else 0.0
        correlations.append(float(corr_val))

    abs_corr = np.abs(np.array(correlations, dtype=np.float64))
    best_abs = abs_corr.max()
    # Deterministic tie-break: prefer the lag closest to zero.
    tied = np.where(np.isclose(abs_corr, best_abs, rtol=1e-12, atol=1e-12))[0]
    if len(tied) == 1:
        best_idx = int(tied[0])
    else:
        best_idx = int(min(tied, key=lambda i: abs(lags[int(i)])))
    return {
        "best_lag": lags[best_idx],
        "best_correlation": correlations[best_idx],
        "lags": lags,
        "correlations": correlations,
    }


def detect_peak_epoch(
    signal: list[float],
    epochs: list[int],
    smooth_window: int = 51,
    skip_frac: float = 0.02,
) -> dict:
    """Detect the epoch where a signal reaches its peak.

    For gradient norms during grokking, the peak marks the transition from
    the memorization phase (rising norms) to the generalization phase
    (falling norms). Skips initial transient.

    Args:
        signal: time series values.
        epochs: corresponding epoch numbers.
        smooth_window: Savitzky-Golay window size.
        skip_frac: fraction of initial epochs to skip (avoids init transient).

    Returns:
        dict with 'transition_epoch', 'transition_idx', 'smoothed' array.
    """
    if len(signal) != len(epochs):
        raise ValueError("signal and epochs must have the same length")
    if not signal:
        raise ValueError("signal must be non-empty")

    smoothed = smooth(signal, window=smooth_window)
    if len(smoothed) == 1:
        return {
            "transition_epoch": epochs[0],
            "transition_idx": 0,
            "smoothed": smoothed,
        }

    skip = int(len(smoothed) * skip_frac)
    if skip >= len(smoothed):
        skip = 0

    # Find peak after initial transient (if any)
    tail = smoothed[skip:]
    idx = skip + int(np.argmax(tail))

    return {
        "transition_epoch": epochs[idx],
        "transition_idx": idx,
        "smoothed": smoothed,
    }


def analyze_run(result: dict) -> dict:
    """Full analysis of a single training run.

    Detects phase transitions in gradient norms and test metrics,
    computes the lag between them.

    Args:
        result: output of trainer.train_and_track().

    Returns:
        dict with transition epochs, lag, and summary statistics.
    """
    epochs = result["epochs"]
    test_metric = result["test_metric"]
    is_classification = result["metric_name"] == "accuracy"

    # Combined gradient norm
    combined_gnorm = compute_gradient_norm_rate(result["grad_norms"])

    # Detect transition in gradient norm: the PEAK of gradient norm marks
    # the transition from memorization to generalization. After the peak,
    # gradient norms decline as the network consolidates efficient circuits.
    gnorm_transition = detect_peak_epoch(combined_gnorm, epochs)

    # Also compute steepest-decrease transition for supplementary analysis
    gnorm_decrease = detect_transition_epoch(
        combined_gnorm, epochs, direction="decrease"
    )

    # Detect transition in test metric (steepest increase)
    metric_transition = detect_transition_epoch(
        test_metric, epochs, direction="increase"
    )

    # Lag: positive = gradient transition PRECEDES metric transition
    lag_epochs = metric_transition["transition_epoch"] - gnorm_transition["transition_epoch"]

    # Cross-correlation on derivatives
    gnorm_deriv = np.gradient(smooth(combined_gnorm)).tolist()
    metric_deriv = np.gradient(smooth(test_metric)).tolist()
    xcorr = cross_correlation_lag(gnorm_deriv, metric_deriv)

    # Per-layer transition analysis
    per_layer = {}
    for layer_name, norms in result["grad_norms"].items():
        lt = detect_peak_epoch(norms, epochs)
        layer_lag = metric_transition["transition_epoch"] - lt["transition_epoch"]
        per_layer[layer_name] = {
            "transition_epoch": lt["transition_epoch"],
            "lag_vs_metric": layer_lag,
        }

    # Pearson correlation between gradient norm and test metric
    if len(combined_gnorm) > 2:
        pearson_r, pearson_p = pearsonr(combined_gnorm, test_metric)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    return {
        "task_name": result["task_name"],
        "frac": result["frac"],
        "gnorm_transition_epoch": gnorm_transition["transition_epoch"],
        "gnorm_steepest_decrease_epoch": gnorm_decrease["transition_epoch"],
        "metric_transition_epoch": metric_transition["transition_epoch"],
        "lag_epochs": lag_epochs,
        "lag_positive": lag_epochs > 0,
        "xcorr_best_lag": xcorr["best_lag"],
        "xcorr_best_correlation": xcorr["best_correlation"],
        "per_layer": per_layer,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "final_train_metric": result["train_metric"][-1],
        "final_test_metric": result["test_metric"][-1],
        "combined_gnorm": combined_gnorm,
        "gnorm_smoothed": gnorm_transition["smoothed"].tolist(),
        "metric_smoothed": metric_transition["smoothed"].tolist(),
    }
