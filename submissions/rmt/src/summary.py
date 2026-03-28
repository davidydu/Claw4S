"""Summary statistics and reproducibility helpers for RMT outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import stats


def compute_paired_delta_summary(
    trained_analysis: list[dict],
    untrained_analysis: list[dict],
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute paired KS-delta summary statistics.

    Pairs are matched by (model_label, layer_name). Deltas are computed as:
    trained_ks - untrained_ks.
    """
    trained_by_key = {
        (entry["model_label"], entry["layer_name"]): entry
        for entry in trained_analysis
    }
    untrained_by_key = {
        (entry["model_label"], entry["layer_name"]): entry
        for entry in untrained_analysis
    }
    shared_keys = sorted(set(trained_by_key.keys()) & set(untrained_by_key.keys()))

    deltas = np.array(
        [
            trained_by_key[key]["ks_statistic"] - untrained_by_key[key]["ks_statistic"]
            for key in shared_keys
        ],
        dtype=np.float64,
    )

    n_pairs = int(len(deltas))
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_ties": 0,
            "positive_fraction": 0.0,
            "avg_delta": 0.0,
            "median_delta": 0.0,
            "std_delta": 0.0,
            "sign_test_pvalue": 1.0,
            "bootstrap_ci_low": 0.0,
            "bootstrap_ci_high": 0.0,
        }

    n_positive = int(np.sum(deltas > 0))
    n_negative = int(np.sum(deltas < 0))
    n_ties = int(np.sum(deltas == 0))

    n_non_ties = n_positive + n_negative
    if n_non_ties > 0:
        sign_pvalue = float(
            stats.binomtest(k=n_positive, n=n_non_ties, p=0.5, alternative="greater").pvalue
        )
    else:
        sign_pvalue = 1.0

    rng = np.random.default_rng(seed)
    bootstrap_draws = rng.choice(deltas, size=(bootstrap_samples, n_pairs), replace=True)
    bootstrap_means = bootstrap_draws.mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    avg_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n_pairs > 1 else 0.0

    return {
        "n_pairs": n_pairs,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_ties": n_ties,
        "positive_fraction": float(round(n_positive / n_pairs, 10)),
        "avg_delta": float(round(avg_delta, 10)),
        "median_delta": float(round(median_delta, 10)),
        "std_delta": float(round(std_delta, 10)),
        "sign_test_pvalue": float(round(sign_pvalue, 12)),
        "bootstrap_ci_low": float(round(float(ci_low), 10)),
        "bootstrap_ci_high": float(round(float(ci_high), 10)),
    }


def compute_sha256(path: str | Path) -> str:
    """Compute SHA256 checksum of a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(
    paths: list[str | Path],
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write a deterministic SHA256 manifest.

    Manifest format follows: "<hex>  <relative-path>" per line.
    """
    manifest = Path(manifest_path)
    base = Path(base_dir) if base_dir is not None else manifest.parent

    normalized_paths = sorted(Path(p) for p in paths)
    checksums: dict[str, str] = {}
    lines: list[str] = []

    for path in normalized_paths:
        rel_path = path.relative_to(base)
        checksum = compute_sha256(path)
        rel_str = rel_path.as_posix()
        checksums[rel_str] = checksum
        lines.append(f"{checksum}  {rel_str}")

    manifest.write_text("\n".join(lines) + "\n")
    return checksums


def verify_checksum_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> list[str]:
    """Validate file hashes declared in a checksum manifest.

    Returns a list of validation errors. Empty list means the manifest is valid.
    """
    manifest = Path(manifest_path)
    if not manifest.is_file():
        return [f"Missing checksum manifest: {manifest.as_posix()}"]

    base = Path(base_dir) if base_dir is not None else manifest.parent
    errors: list[str] = []

    for raw_line in manifest.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected_hash, rel_path = line.split("  ", maxsplit=1)
        except ValueError:
            errors.append(f"Malformed checksum line: {raw_line}")
            continue

        target = base / rel_path
        if not target.is_file():
            errors.append(f"Missing file listed in manifest: {rel_path}")
            continue

        actual_hash = compute_sha256(target)
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    return errors
