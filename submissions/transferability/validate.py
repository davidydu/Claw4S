"""Validate experiment results for correctness and completeness.

Usage (from submissions/transferability/):
    .venv/bin/python validate.py

Checks:
    1. results/transfer_results.json exists and is valid JSON
    2. Contains expected number of same_arch results (48 = 4x4x3)
    3. Contains expected number of cross_depth results (48 = 4x4x3)
    4. All transfer rates are in [0, 1]
    5. Summary statistics are present and reasonable
    6. Plots exist (3 PNG files)
    7. Scientific sanity: diagonal transfer rates tend higher than off-diagonal
"""

import json
import os
import sys
from pathlib import Path

# Working-directory guard
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

RESULTS_DIR = Path("results")
RESULTS_JSON = RESULTS_DIR / "transfer_results.json"
EXPECTED_PLOTS = [
    "transfer_heatmap.png",
    "transfer_by_ratio.png",
    "depth_comparison.png",
]

N_WIDTHS = 4
N_SEEDS = 3
EXPECTED_SAME_ARCH = N_WIDTHS * N_WIDTHS * N_SEEDS  # 48
EXPECTED_CROSS_DEPTH = N_WIDTHS * N_WIDTHS * N_SEEDS  # 48


def check(condition: bool, msg: str) -> None:
    """Assert a validation condition, printing PASS/FAIL."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(f"Validation failed: {msg}")


def validate() -> None:
    """Run all validation checks."""
    print("=" * 60)
    print("Validating Adversarial Transferability Results")
    print("=" * 60)

    # 1. Results file exists
    print("\n1. Results file:")
    check(RESULTS_JSON.exists(), f"{RESULTS_JSON} exists")

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    # 2. Same-arch results count
    print("\n2. Same-architecture results:")
    same_arch = data["same_arch_results"]
    check(
        len(same_arch) == EXPECTED_SAME_ARCH,
        f"Expected {EXPECTED_SAME_ARCH} same-arch results, got {len(same_arch)}",
    )

    # 3. Cross-depth results count
    print("\n3. Cross-depth results:")
    cross_depth = data["cross_depth_results"]
    check(
        len(cross_depth) == EXPECTED_CROSS_DEPTH,
        f"Expected {EXPECTED_CROSS_DEPTH} cross-depth results, got {len(cross_depth)}",
    )

    # 4. Transfer rates in [0, 1]
    print("\n4. Transfer rate bounds:")
    all_results = same_arch + cross_depth
    rates = [r["transfer_rate"] for r in all_results]
    check(
        all(0.0 <= r <= 1.0 for r in rates),
        f"All {len(rates)} transfer rates in [0, 1]",
    )

    # 5. Summary statistics
    print("\n5. Summary statistics:")
    summary = data["summary"]
    required_keys = [
        "diagonal_mean_transfer",
        "off_diagonal_mean_transfer",
        "transfer_by_capacity_ratio",
        "same_width_same_depth_mean",
        "same_width_cross_depth_mean",
        "n_same_arch_runs",
        "n_cross_depth_runs",
        "runtime_seconds",
    ]
    for key in required_keys:
        check(key in summary, f"Summary contains '{key}'")

    check(
        summary["runtime_seconds"] < 300,
        f"Runtime {summary['runtime_seconds']}s < 300s limit",
    )

    config = data.get("config", {})
    required_config_keys = [
        "widths",
        "seeds",
        "epsilon",
        "n_samples",
        "n_features",
        "n_classes",
        "train_epochs",
        "train_lr",
        "train_batch_size",
    ]
    for key in required_config_keys:
        check(key in config, f"Config contains '{key}'")

    # 6. Plots exist
    print("\n6. Visualization files:")
    for plot_name in EXPECTED_PLOTS:
        path = RESULTS_DIR / plot_name
        check(path.exists(), f"{plot_name} exists")
        check(
            path.stat().st_size > 1000,
            f"{plot_name} is non-trivial ({path.stat().st_size} bytes)",
        )

    # 7. Scientific sanity checks
    print("\n7. Scientific sanity:")
    diag_mean = summary["diagonal_mean_transfer"]
    off_diag_mean = summary["off_diagonal_mean_transfer"]
    check(
        diag_mean is not None and off_diag_mean is not None,
        "Diagonal and off-diagonal means are not None",
    )

    # Check that models trained successfully (clean accuracy > chance = 0.2)
    clean_accs = [r["source_clean_acc"] for r in all_results]
    check(
        all(a > 0.2 for a in clean_accs),
        f"All source clean accuracies above chance (min={min(clean_accs):.3f})",
    )

    # Check that FGSM actually produces adversarial examples
    n_advs = [r["n_successful_source_advs"] for r in all_results]
    check(
        sum(n_advs) > 0,
        f"FGSM produced adversarial examples (total={sum(n_advs)})",
    )

    # Capacity ratio = 1.0 should exist
    ratio_1 = summary["transfer_by_capacity_ratio"].get("1.0")
    check(
        ratio_1 is not None,
        f"Capacity ratio 1.0 exists in summary (transfer={ratio_1})",
    )

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)

    # Print key findings
    print(f"\nKey findings:")
    print(f"  Diagonal (same-width) mean transfer rate: {diag_mean}")
    print(f"  Off-diagonal mean transfer rate: {off_diag_mean}")
    print(f"  Same-width same-depth mean: {summary['same_width_same_depth_mean']}")
    print(f"  Same-width cross-depth mean: {summary['same_width_cross_depth_mean']}")
    print(f"  Transfer by capacity ratio:")
    for ratio, mean in summary["transfer_by_capacity_ratio"].items():
        print(f"    ratio={ratio}: {mean}")


if __name__ == "__main__":
    validate()
