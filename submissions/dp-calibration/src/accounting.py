"""Privacy accounting methods for Gaussian mechanism noise calibration.

Implements four methods to compute the privacy loss epsilon for a given
noise multiplier sigma, number of composition steps T, and failure
probability delta:

1. Naive (linear) composition
2. Advanced composition (Dwork, Rothblum, Vadhan 2010)
3. Renyi DP accounting (Mironov 2017)
4. Gaussian DP / f-DP accounting (Dong, Roth, Su 2019)

All methods assume the Gaussian mechanism with sensitivity 1.
"""

import math
import numpy as np
from scipy import special
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Naive composition
# ---------------------------------------------------------------------------

def epsilon_naive(sigma: float, T: int, delta: float) -> float:
    """Naive (linear) composition: epsilon_total = T * epsilon_step.

    Single-step epsilon for the Gaussian mechanism at given delta:
        epsilon_step = sqrt(2 * ln(1.25 / delta)) / sigma

    Reference: Dwork & Roth, "The Algorithmic Foundations of DP" (2014), Thm A.1
    """
    if sigma <= 0 or T <= 0 or delta <= 0 or delta >= 1:
        return float("inf")
    eps_step = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
    return T * eps_step


# ---------------------------------------------------------------------------
# 2. Advanced composition (Dwork, Rothblum, Vadhan 2010)
# ---------------------------------------------------------------------------

def epsilon_advanced(sigma: float, T: int, delta: float) -> float:
    """Advanced composition theorem.

    For T applications of an (eps_step, 0)-DP mechanism, the advanced
    composition theorem gives (eps_total, delta)-DP where:

        eps_total = sqrt(2*T * ln(1/delta)) * eps_step + T * eps_step * (e^eps_step - 1)

    For the Gaussian mechanism, each step is (eps_step, 0)-DP in the
    concentrated/zero-CDP sense is not applicable, so we use the
    heterogeneous formulation: each step is (eps_step, delta_step)-DP,
    and we optimize over the per-step delta allocation.

    We use a clean formulation: allocate delta' for the composition
    slack and use the remaining budget per step. The per-step eps uses
    the same Gaussian mechanism formula with the full delta (not split),
    and the composition benefit comes from sqrt(T) scaling instead of T.

        eps_step = sqrt(2 * ln(1.25 / delta)) / sigma   (same as naive)
        eps_total = sqrt(2*T * ln(1/delta')) * eps_step + T * eps_step * (e^eps_step - 1)

    where delta' is set to optimize the bound. We search over delta'.

    The naive bound always holds as a fallback (return min).

    Reference: Dwork, Rothblum, Vadhan (2010), Theorem 3.3
    """
    if sigma <= 0 or T <= 0 or delta <= 0 or delta >= 1:
        return float("inf")

    # Naive bound always applies as a fallback
    naive_eps = epsilon_naive(sigma, T, delta)

    # Try multiple delta' allocations and pick the best
    best_adv = float("inf")
    for frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        delta_prime = delta * frac
        delta_step = delta * (1.0 - frac)

        if delta_prime <= 0 or delta_step <= 0:
            continue

        eps_step = math.sqrt(2.0 * math.log(1.25 / delta_step)) / sigma

        # Guard against overflow in exp(eps_step)
        if eps_step > 500:
            continue

        term1 = math.sqrt(2.0 * T * math.log(1.0 / delta_prime)) * eps_step
        term2 = T * eps_step * (math.exp(eps_step) - 1.0)
        adv_eps = term1 + term2

        if adv_eps < best_adv:
            best_adv = adv_eps

    return min(best_adv, naive_eps)


# ---------------------------------------------------------------------------
# 3. Renyi DP accounting (Mironov 2017)
# ---------------------------------------------------------------------------

# Default RDP orders to evaluate
DEFAULT_RDP_ORDERS = [2, 4, 8, 16, 32, 64, 128, 256]


def _rdp_gaussian_single_order(sigma: float, alpha: float) -> float:
    """RDP of a single Gaussian mechanism step at order alpha.

    For the Gaussian mechanism with sensitivity 1 and noise sigma:
        RDP(alpha) = alpha / (2 * sigma^2)

    Reference: Mironov (2017), Proposition 3
    """
    return alpha / (2.0 * sigma ** 2)


def _rdp_to_eps(rdp_value: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee to (epsilon, delta)-DP.

    epsilon = rdp_value - ln(delta) / (alpha - 1)

    with the tighter conversion from Balle et al. (2020):
        epsilon = rdp_value - (ln(delta) + ln(alpha-1)) / (alpha-1) - ln(1 - 1/alpha)

    Reference: Balle, Gaboardi, Zanella-Beguelin (2020), Proposition 3
    """
    if alpha <= 1:
        return float("inf")
    # Tighter conversion (Balle et al. 2020)
    log_term = math.log(delta) + math.log(alpha - 1)
    eps = rdp_value - log_term / (alpha - 1.0) - math.log(1.0 - 1.0 / alpha)
    return max(eps, 0.0)


def epsilon_rdp(sigma: float, T: int, delta: float,
                orders: list[float] | None = None) -> float:
    """RDP-based privacy accounting.

    1. Compute RDP at each order alpha for a single step.
    2. Compose T steps: RDP_total(alpha) = T * RDP_step(alpha).
    3. Convert each order to (epsilon, delta)-DP.
    4. Return the minimum epsilon across all orders.

    Reference: Mironov (2017); Balle et al. (2020) for conversion
    """
    if sigma <= 0 or T <= 0 or delta <= 0 or delta >= 1:
        return float("inf")

    if orders is None:
        orders = DEFAULT_RDP_ORDERS

    best_eps = float("inf")
    for alpha in orders:
        rdp_step = _rdp_gaussian_single_order(sigma, alpha)
        rdp_total = T * rdp_step
        eps = _rdp_to_eps(rdp_total, alpha, delta)
        if eps < best_eps:
            best_eps = eps
    return best_eps


# ---------------------------------------------------------------------------
# 4. Gaussian DP / f-DP (Dong, Roth, Su 2019)
# ---------------------------------------------------------------------------

def epsilon_gdp(sigma: float, T: int, delta: float) -> float:
    """Gaussian DP (f-DP) accounting via the central limit theorem.

    For the Gaussian mechanism with sensitivity 1 and noise sigma,
    each step is mu-GDP with mu = 1/sigma. After T compositions by
    the CLT for f-DP:

        mu_total = sqrt(T) / sigma

    Then convert mu-GDP to (epsilon, delta)-DP:
        delta = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)

    We solve for epsilon numerically using the dual:
        epsilon = mu_total * Phi^{-1}(1 - delta) - mu_total^2 / 2

    which is valid as a lower bound. We use the tighter formula from
    Dong et al. (2019) Corollary 2.13.

    Reference: Dong, Roth, Su (2019), "Gaussian Differential Privacy"
    """
    if sigma <= 0 or T <= 0 or delta <= 0 or delta >= 1:
        return float("inf")

    mu = math.sqrt(T) / sigma

    # Use the analytic formula: solve for eps in
    #   delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)
    # via binary search
    eps = _solve_gdp_epsilon(mu, delta)
    return max(eps, 0.0)


def _gdp_delta_at_eps(mu: float, eps: float) -> float:
    """Compute delta for a given epsilon under mu-GDP.

    delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2)

    Uses log-space computation for the second term to avoid overflow
    when eps is large.
    """
    arg1 = -eps / mu + mu / 2.0
    arg2 = -eps / mu - mu / 2.0
    t1 = norm.cdf(arg1)

    # Compute exp(eps) * Phi(arg2) in log-space to avoid overflow
    log_phi2 = norm.logcdf(arg2)
    log_t2 = eps + log_phi2

    if log_t2 > 500:
        # exp(eps) * Phi(arg2) is huge, delta is very negative => clamp to 0
        return 0.0
    if log_t2 < -700:
        # exp(eps) * Phi(arg2) is negligible
        return max(t1, 0.0)

    t2 = math.exp(log_t2)
    return max(t1 - t2, 0.0)


def _solve_gdp_epsilon(mu: float, delta: float) -> float:
    """Solve for epsilon given mu-GDP parameter and target delta.

    Binary search on epsilon in [0, upper_bound].
    delta(eps) is monotonically decreasing in eps.
    """
    # Upper bound: use the analytic approximation
    eps_upper = mu * norm.ppf(1.0 - delta) + mu ** 2 / 2.0
    eps_upper = max(eps_upper, 1.0)
    # Expand upper bound if needed
    while _gdp_delta_at_eps(mu, eps_upper) > delta:
        eps_upper *= 2.0

    eps_lower = 0.0
    for _ in range(200):  # sufficient for double precision
        eps_mid = (eps_lower + eps_upper) / 2.0
        d_mid = _gdp_delta_at_eps(mu, eps_mid)
        if d_mid > delta:
            eps_lower = eps_mid
        else:
            eps_upper = eps_mid
        if eps_upper - eps_lower < 1e-12:
            break

    return (eps_lower + eps_upper) / 2.0


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

METHOD_NAMES = ["naive", "advanced", "rdp", "gdp"]

METHODS = {
    "naive": epsilon_naive,
    "advanced": epsilon_advanced,
    "rdp": epsilon_rdp,
    "gdp": epsilon_gdp,
}


def compute_epsilon(method: str, sigma: float, T: int, delta: float) -> float:
    """Compute epsilon for a given method, sigma, T, delta."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Choose from {METHOD_NAMES}")
    return METHODS[method](sigma=sigma, T=T, delta=delta)


def compute_all_epsilons(sigma: float, T: int, delta: float) -> dict[str, float]:
    """Compute epsilon for all four methods."""
    return {name: fn(sigma=sigma, T=T, delta=delta)
            for name, fn in METHODS.items()}
