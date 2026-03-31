"""Random Matrix Theory analysis of neural network weight matrices.

Computes eigenvalue spectra of weight correlation matrices and compares
them to the Marchenko-Pastur (MP) distribution from RMT. The MP law
describes the expected eigenvalue distribution of (1/M) * W^T W when
W is an M x N random matrix with i.i.d. entries.
"""

import warnings

import numpy as np
from scipy import stats
from scipy import integrate


def compute_mp_bounds(
    gamma: float,
    sigma_sq: float,
) -> tuple[float, float]:
    """Compute Marchenko-Pastur bulk edge eigenvalues.

    Args:
        gamma: Aspect ratio N/M of the weight matrix.
        sigma_sq: Variance of matrix entries.

    Returns:
        (lambda_minus, lambda_plus): Lower and upper bounds of the MP bulk.
    """
    sqrt_gamma = np.sqrt(gamma)
    lambda_plus = sigma_sq * (1.0 + sqrt_gamma) ** 2
    lambda_minus = sigma_sq * (1.0 - sqrt_gamma) ** 2
    return float(lambda_minus), float(lambda_plus)


def marchenko_pastur_pdf(
    x: np.ndarray,
    gamma: float,
    sigma_sq: float,
) -> np.ndarray:
    """Evaluate Marchenko-Pastur probability density function.

    Args:
        x: Points at which to evaluate the PDF.
        gamma: Aspect ratio N/M.
        sigma_sq: Variance of matrix entries.

    Returns:
        PDF values at each point in x. Zero outside [lambda_-, lambda_+].
    """
    x = np.asarray(x, dtype=np.float64)
    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)

    pdf = np.zeros_like(x)
    mask = (x > lam_min) & (x < lam_max) & (x > 0)

    if mask.any():
        xi = x[mask]
        numerator = np.sqrt((lam_max - xi) * (xi - lam_min))
        denominator = 2.0 * np.pi * sigma_sq * gamma * xi
        pdf[mask] = numerator / denominator

    return pdf


def marchenko_pastur_cdf(
    x: np.ndarray,
    gamma: float,
    sigma_sq: float,
    n_points: int = 2000,
) -> np.ndarray:
    """Evaluate Marchenko-Pastur cumulative distribution function.

    Uses numerical integration of the PDF.

    Args:
        x: Points at which to evaluate the CDF.
        gamma: Aspect ratio N/M.
        sigma_sq: Variance of matrix entries.
        n_points: Number of integration points.

    Returns:
        CDF values at each point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)

    cdf = np.zeros_like(x)

    # Create fine grid for integration
    grid = np.linspace(lam_min, lam_max, n_points)
    pdf_vals = marchenko_pastur_pdf(grid, gamma, sigma_sq)

    # Cumulative trapezoidal integration
    cumulative = integrate.cumulative_trapezoid(pdf_vals, grid, initial=0.0)
    # Normalize to ensure CDF reaches 1 (handle numerical integration error)
    if cumulative[-1] > 0:
        cumulative = cumulative / cumulative[-1]

    for i, xi in enumerate(x):
        if xi <= lam_min:
            cdf[i] = 0.0
        elif xi >= lam_max:
            cdf[i] = 1.0
        else:
            # Interpolate
            idx = np.searchsorted(grid, xi)
            if idx == 0:
                cdf[i] = 0.0
            elif idx >= len(grid):
                cdf[i] = 1.0
            else:
                # Linear interpolation
                frac = (xi - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
                cdf[i] = cumulative[idx - 1] + frac * (
                    cumulative[idx] - cumulative[idx - 1]
                )

    return cdf


def analyze_weight_matrix(
    W: np.ndarray,
    layer_name: str = "",
) -> dict:
    """Analyze a single weight matrix against the Marchenko-Pastur distribution.

    Computes the correlation matrix C = (1/M) * W^T W, then compares
    its eigenvalue spectrum to the MP prediction.

    Args:
        W: Weight matrix of shape (out_features, in_features) = (M, N).
        layer_name: Optional name for labeling.

    Returns:
        Dict containing eigenvalues, MP parameters, and deviation metrics:
        - eigenvalues: sorted eigenvalues of C
        - gamma, sigma_sq: MP parameters
        - lambda_minus, lambda_plus: MP bulk edges
        - ks_statistic: KS distance from MP distribution
        - ks_pvalue: p-value of KS test
        - outlier_fraction: fraction of eigenvalues outside MP bulk
        - spectral_norm_ratio: max eigenvalue / lambda_plus
        - kl_divergence: binned KL divergence from MP PDF
    """
    W = np.asarray(W, dtype=np.float64)
    M, N = W.shape

    # Ensure M >= N for well-defined MP; if not, transpose
    if M < N:
        W = W.T
        M, N = W.shape

    # Handle degenerate case: N=1 means only 1 eigenvalue, MP is meaningless
    if N < 2:
        return {
            "layer_name": layer_name,
            "shape": (int(M), int(N)),
            "gamma": float(N / M),
            "sigma_sq": float(np.var(W)),
            "lambda_minus": 0.0,
            "lambda_plus": 0.0,
            "max_eigenvalue": 0.0,
            "min_eigenvalue": 0.0,
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "outlier_fraction": 0.0,
            "n_outliers": 0,
            "spectral_norm_ratio": 0.0,
            "kl_divergence": 0.0,
            "n_eigenvalues": int(N),
            "eigenvalues": [0.0] * N,
        }

    gamma = N / M
    sigma_sq = float(np.var(W))

    # Correlation matrix: (1/M) * W^T W
    # Suppress transient float64 matmul warnings on certain matrix sizes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = (1.0 / M) * (W.T @ W)
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)

    # MP bounds
    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)

    # KS test: compare empirical eigenvalue CDF to MP CDF
    # Filter to eigenvalues within a reasonable range for the test
    # (include all eigenvalues to capture deviations)
    if len(eigenvalues) > 1 and sigma_sq > 1e-12:
        emp_cdf = np.arange(1, len(eigenvalues) + 1) / len(eigenvalues)
        theo_cdf = marchenko_pastur_cdf(eigenvalues, gamma, sigma_sq)
        ks_stat = float(np.max(np.abs(emp_cdf - theo_cdf)))

        # Approximate p-value using scipy KS distribution
        n_eff = len(eigenvalues)
        ks_pvalue = float(stats.kstwobign.sf(ks_stat * np.sqrt(n_eff)))
    else:
        ks_stat = 0.0
        ks_pvalue = 1.0

    # Outlier fraction: eigenvalues outside [lambda_-, lambda_+]
    n_outliers = int(np.sum((eigenvalues < lam_min - 1e-10) |
                            (eigenvalues > lam_max + 1e-10)))
    outlier_fraction = n_outliers / len(eigenvalues) if len(eigenvalues) > 0 else 0.0

    # Spectral norm ratio: max eigenvalue / lambda_plus
    max_eig = float(eigenvalues[-1])
    spectral_norm_ratio = max_eig / lam_max if lam_max > 1e-12 else 0.0

    # KL divergence (binned): approximate KL(empirical || MP)
    kl_div = _compute_binned_kl(eigenvalues, gamma, sigma_sq, n_bins=50)

    return {
        "layer_name": layer_name,
        "shape": (int(W.shape[0]), int(W.shape[1])),
        "gamma": float(gamma),
        "sigma_sq": float(sigma_sq),
        "lambda_minus": float(lam_min),
        "lambda_plus": float(lam_max),
        "max_eigenvalue": float(max_eig),
        "min_eigenvalue": float(eigenvalues[0]),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "outlier_fraction": float(outlier_fraction),
        "n_outliers": int(n_outliers),
        "spectral_norm_ratio": float(spectral_norm_ratio),
        "kl_divergence": float(kl_div),
        "n_eigenvalues": int(len(eigenvalues)),
        "eigenvalues": eigenvalues.tolist(),
    }


def _compute_binned_kl(
    eigenvalues: np.ndarray,
    gamma: float,
    sigma_sq: float,
    n_bins: int = 50,
) -> float:
    """Compute binned KL divergence between empirical and MP distributions.

    KL(P_empirical || P_MP) using histogram binning.

    Args:
        eigenvalues: Sorted eigenvalue array.
        gamma: Aspect ratio.
        sigma_sq: Entry variance.
        n_bins: Number of histogram bins.

    Returns:
        KL divergence (nats). Returns 0 if computation fails.
    """
    if len(eigenvalues) < 2 or sigma_sq < 1e-12:
        return 0.0

    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)

    # Bin range covers both empirical and theoretical support
    bin_lo = max(0, min(eigenvalues[0], lam_min) - 0.01)
    bin_hi = max(eigenvalues[-1], lam_max) + 0.01
    bins = np.linspace(bin_lo, bin_hi, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # Empirical histogram (normalized to PDF)
    emp_counts, _ = np.histogram(eigenvalues, bins=bins)
    emp_pdf = emp_counts / (len(eigenvalues) * bin_width)

    # Theoretical MP PDF at bin centers
    theo_pdf = marchenko_pastur_pdf(bin_centers, gamma, sigma_sq)

    # Smooth both distributions to avoid log(0)
    eps = 1e-10
    emp_pdf = emp_pdf + eps
    theo_pdf = theo_pdf + eps

    # Normalize to valid probability distributions
    emp_pdf = emp_pdf / (emp_pdf.sum() * bin_width)
    theo_pdf = theo_pdf / (theo_pdf.sum() * bin_width)

    # KL divergence: sum p * log(p/q)
    kl = float(np.sum(emp_pdf * np.log(emp_pdf / theo_pdf) * bin_width))

    return max(0.0, kl)  # Clamp to non-negative


def analyze_model_weights(
    weight_matrices: list[tuple[str, np.ndarray]],
    model_label: str = "",
) -> list[dict]:
    """Analyze all weight matrices from a model.

    Args:
        weight_matrices: List of (layer_name, weight_array) tuples.
        model_label: Label for the model configuration.

    Returns:
        List of analysis dicts, one per layer.
    """
    results = []
    for layer_name, W in weight_matrices:
        analysis = analyze_weight_matrix(W, layer_name=layer_name)
        analysis["model_label"] = model_label
        results.append(analysis)
    return results
