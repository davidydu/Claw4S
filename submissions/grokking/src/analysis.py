"""Phase classification and analysis for grokking experiments.

Classifies each training run into one of four phases:
  - CONFUSION: neither memorizes nor generalizes
  - MEMORIZATION: memorizes but does not generalize
  - GROKKING: delayed generalization (memorizes first, then generalizes)
  - COMPREHENSION: fast generalization without delay
"""

from enum import Enum

from src.train import TrainResult


class Phase(Enum):
    """Possible learning outcomes for a training run."""

    CONFUSION = "confusion"
    MEMORIZATION = "memorization"
    GROKKING = "grokking"
    COMPREHENSION = "comprehension"


# Threshold for considering accuracy as "achieved"
ACC_THRESHOLD = 0.95

# Minimum gap (in epochs) between train and test reaching threshold
# to classify as grokking vs comprehension
GROKKING_GAP_THRESHOLD = 200


def classify_phase(result: TrainResult) -> Phase:
    """Classify a training run into a learning phase.

    Classification rules:
      1. If final train_acc < 95%: CONFUSION (can't even memorize)
      2. If final test_acc < 95%: MEMORIZATION (memorized but didn't generalize)
      3. If grokking gap > 200 epochs: GROKKING (delayed generalization)
      4. Otherwise: COMPREHENSION (fast generalization)

    Args:
        result: TrainResult from a training run.

    Returns:
        Phase classification.
    """
    train_reached = result.epoch_train_95 is not None
    test_reached = result.epoch_test_95 is not None

    if not train_reached or result.final_train_acc < ACC_THRESHOLD:
        return Phase.CONFUSION

    if not test_reached or result.final_test_acc < ACC_THRESHOLD:
        return Phase.MEMORIZATION

    gap = compute_grokking_gap(result)
    if gap is not None and gap > GROKKING_GAP_THRESHOLD:
        return Phase.GROKKING

    return Phase.COMPREHENSION


def compute_grokking_gap(result: TrainResult) -> int | None:
    """Compute the epoch gap between train and test reaching 95% accuracy.

    Args:
        result: TrainResult from a training run.

    Returns:
        Gap in epochs, or None if either threshold was not reached.
    """
    if result.epoch_train_95 is None or result.epoch_test_95 is None:
        return None
    return result.epoch_test_95 - result.epoch_train_95


def aggregate_results(sweep_results: list[dict]) -> dict:
    """Compute summary statistics across all sweep results.

    Args:
        sweep_results: List of dicts, each with "phase" (Phase) and
            "grokking_gap" (int or None) keys.

    Returns:
        Dictionary with:
          - phase_counts: dict mapping Phase value to count
          - total_runs: total number of runs
          - grokking_fraction: fraction of runs that grokked
          - mean_grokking_gap: mean gap for grokking runs (or None)
          - max_grokking_gap: max gap for grokking runs (or None)
    """
    phase_counts = {p.value: 0 for p in Phase}
    grokking_gaps = []

    for r in sweep_results:
        phase = r["phase"]
        phase_val = phase.value if isinstance(phase, Phase) else phase
        phase_counts[phase_val] = phase_counts.get(phase_val, 0) + 1

        if phase_val == Phase.GROKKING.value and r.get("grokking_gap") is not None:
            grokking_gaps.append(r["grokking_gap"])

    total = len(sweep_results)

    return {
        "phase_counts": phase_counts,
        "total_runs": total,
        "grokking_fraction": phase_counts.get(Phase.GROKKING.value, 0) / max(total, 1),
        "mean_grokking_gap": (
            sum(grokking_gaps) / len(grokking_gaps) if grokking_gaps else None
        ),
        "max_grokking_gap": max(grokking_gaps) if grokking_gaps else None,
    }
