"""Benford's Law analysis: leading digit extraction and statistical tests.

Provides functions to extract leading digits from neural network weights,
compute observed digit distributions, and test conformity to Benford's Law
using chi-squared, MAD, and KL divergence.
"""

import math

import numpy as np
from scipy import stats


def benford_expected():
    """Return Benford's Law expected probabilities for digits 1-9.

    P(d) = log10(1 + 1/d)

    Returns:
        Dict mapping digit (int 1-9) to expected probability (float).
    """
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def extract_leading_digits(weights):
    """Extract the leading significant digit from each weight value.

    Takes absolute values, excludes zeros and tiny values (< 1e-10),
    then extracts the first non-zero digit.

    Args:
        weights: numpy array of weight values.

    Returns:
        numpy array of leading digits (integers 1-9).
    """
    w = np.abs(weights).flatten()

    # Exclude zeros and tiny values
    w = w[w > 1e-10]

    if len(w) == 0:
        return np.array([], dtype=np.int64)

    # Extract leading digit: normalize to [1, 10) then take floor
    # log10(|w|) gives the exponent; subtract floor to get mantissa
    log_vals = np.log10(w)
    mantissa = log_vals - np.floor(log_vals)
    leading = np.floor(10**mantissa).astype(np.int64)

    # Clamp to [1, 9] (numerical edge cases)
    leading = np.clip(leading, 1, 9)

    return leading


def compute_digit_distribution(digits):
    """Compute the observed frequency distribution of leading digits.

    Args:
        digits: numpy array of leading digits (integers 1-9).

    Returns:
        Dict mapping digit (int 1-9) to observed proportion (float).
        Returns uniform 1/9 for each digit if input is empty.
    """
    if len(digits) == 0:
        return {d: 1.0 / 9 for d in range(1, 10)}

    n = len(digits)
    counts = {}
    for d in range(1, 10):
        counts[d] = np.sum(digits == d) / n

    return counts


def chi_squared_test(observed, n):
    """Perform chi-squared goodness-of-fit test against Benford's Law.

    Args:
        observed: Dict mapping digit (1-9) to observed proportion.
        n: Total number of observations.

    Returns:
        Tuple of (chi2_statistic, p_value, degrees_of_freedom).
    """
    expected = benford_expected()
    df = 8  # 9 categories - 1

    chi2 = 0.0
    for d in range(1, 10):
        obs_count = observed[d] * n
        exp_count = expected[d] * n
        if exp_count > 0:
            chi2 += (obs_count - exp_count) ** 2 / exp_count

    p_value = 1.0 - stats.chi2.cdf(chi2, df)

    return chi2, p_value, df


def mad_from_benford(observed):
    """Compute Mean Absolute Deviation from Benford's expected distribution.

    MAD = (1/9) * sum_d |observed_d - expected_d|

    Nigrini's classification thresholds:
    - < 0.006: Close conformity
    - 0.006-0.012: Acceptable conformity
    - 0.012-0.015: Marginal conformity
    - > 0.015: Nonconformity

    Args:
        observed: Dict mapping digit (1-9) to observed proportion.

    Returns:
        MAD value as float.
    """
    expected = benford_expected()
    total_dev = sum(abs(observed[d] - expected[d]) for d in range(1, 10))
    return total_dev / 9


def classify_mad(mad_value):
    """Classify MAD value using Nigrini's thresholds.

    Args:
        mad_value: Mean Absolute Deviation from Benford.

    Returns:
        String classification: "close", "acceptable", "marginal", or "nonconformity".
    """
    if mad_value < 0.006:
        return "close"
    elif mad_value < 0.012:
        return "acceptable"
    elif mad_value < 0.015:
        return "marginal"
    else:
        return "nonconformity"


def kl_divergence(observed):
    """Compute KL divergence from observed distribution to Benford's Law.

    D_KL(observed || benford) = sum_d observed_d * log(observed_d / expected_d)

    Args:
        observed: Dict mapping digit (1-9) to observed proportion.

    Returns:
        KL divergence as float (in nats). Returns inf if any observed is 0.
    """
    expected = benford_expected()
    kl = 0.0
    for d in range(1, 10):
        if observed[d] <= 0:
            return float("inf")
        kl += observed[d] * math.log(observed[d] / expected[d])
    return kl


def analyze_snapshot(state_dict, layer_filter="weight"):
    """Analyze a model weight snapshot for Benford's Law conformity.

    Args:
        state_dict: Model state dict (as from model.state_dict()).
        layer_filter: Only analyze parameters whose name contains this string.

    Returns:
        Dict with keys:
        - "per_layer": dict mapping layer name to analysis results
        - "aggregate": analysis results for all weights combined
        Each analysis result contains: observed_dist, n_weights, chi2, p_value,
        mad, mad_class, kl_div.
    """
    all_weights = []
    per_layer = {}

    for name, tensor in state_dict.items():
        if layer_filter not in name:
            continue

        weights_np = tensor.cpu().numpy()
        digits = extract_leading_digits(weights_np)

        if len(digits) == 0:
            continue

        dist = compute_digit_distribution(digits)
        n = len(digits)
        chi2, p_val, df = chi_squared_test(dist, n)
        mad = mad_from_benford(dist)
        kl = kl_divergence(dist)

        per_layer[name] = {
            "observed_dist": {str(k): v for k, v in dist.items()},
            "n_weights": n,
            "chi2": chi2,
            "p_value": p_val,
            "mad": mad,
            "mad_class": classify_mad(mad),
            "kl_div": kl,
        }

        all_weights.append(weights_np.flatten())

    # Aggregate analysis
    aggregate = {}
    if all_weights:
        combined = np.concatenate(all_weights)
        digits = extract_leading_digits(combined)
        dist = compute_digit_distribution(digits)
        n = len(digits)
        chi2, p_val, df = chi_squared_test(dist, n)
        mad = mad_from_benford(dist)
        kl = kl_divergence(dist)

        aggregate = {
            "observed_dist": {str(k): v for k, v in dist.items()},
            "n_weights": n,
            "chi2": chi2,
            "p_value": p_val,
            "mad": mad,
            "mad_class": classify_mad(mad),
            "kl_div": kl,
        }

    return {"per_layer": per_layer, "aggregate": aggregate}


def generate_control_weights(n=10000, seed=42):
    """Generate control weight distributions for comparison.

    Args:
        n: Number of weights to generate.
        seed: Random seed.

    Returns:
        Dict mapping control name to analysis results.
    """
    rng = np.random.RandomState(seed)

    controls = {}

    # Uniform random in [-1, 1]
    uniform_weights = rng.uniform(-1, 1, size=n).astype(np.float32)
    uniform_digits = extract_leading_digits(uniform_weights)
    uniform_dist = compute_digit_distribution(uniform_digits)
    n_u = len(uniform_digits)
    chi2_u, p_u, _ = chi_squared_test(uniform_dist, n_u)
    mad_u = mad_from_benford(uniform_dist)
    kl_u = kl_divergence(uniform_dist)
    controls["uniform"] = {
        "observed_dist": {str(k): v for k, v in uniform_dist.items()},
        "n_weights": n_u,
        "chi2": chi2_u,
        "p_value": p_u,
        "mad": mad_u,
        "mad_class": classify_mad(mad_u),
        "kl_div": kl_u,
    }

    # Normal random N(0, 0.01)
    normal_weights = rng.normal(0, 0.01, size=n).astype(np.float32)
    normal_digits = extract_leading_digits(normal_weights)
    normal_dist = compute_digit_distribution(normal_digits)
    n_n = len(normal_digits)
    chi2_n, p_n, _ = chi_squared_test(normal_dist, n_n)
    mad_n = mad_from_benford(normal_dist)
    kl_n = kl_divergence(normal_dist)
    controls["normal"] = {
        "observed_dist": {str(k): v for k, v in normal_dist.items()},
        "n_weights": n_n,
        "chi2": chi2_n,
        "p_value": p_n,
        "mad": mad_n,
        "mad_class": classify_mad(mad_n),
        "kl_div": kl_n,
    }

    # Kaiming uniform (simulating PyTorch default init)
    fan_in = 128
    bound = math.sqrt(1.0 / fan_in)
    kaiming_weights = rng.uniform(-bound, bound, size=n).astype(np.float32)
    kaiming_digits = extract_leading_digits(kaiming_weights)
    kaiming_dist = compute_digit_distribution(kaiming_digits)
    n_k = len(kaiming_digits)
    chi2_k, p_k, _ = chi_squared_test(kaiming_dist, n_k)
    mad_k = mad_from_benford(kaiming_dist)
    kl_k = kl_divergence(kaiming_dist)
    controls["kaiming_uniform"] = {
        "observed_dist": {str(k): v for k, v in kaiming_dist.items()},
        "n_weights": n_k,
        "chi2": chi2_k,
        "p_value": p_k,
        "mad": mad_k,
        "mad_class": classify_mad(mad_k),
        "kl_div": kl_k,
    }

    return controls
