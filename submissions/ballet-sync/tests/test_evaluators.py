# tests/test_evaluators.py
import numpy as np
from src.evaluators import (
    KuramotoOrderEvaluator, SpatialAlignmentEvaluator,
    VelocitySynchronyEvaluator, PairwiseEntrainmentEvaluator,
    EvaluatorPanel, BaseEvaluator,
)


def _synced_history(n=12, T=1000):
    """All agents at same phase throughout."""
    return np.zeros((T, n))


def _random_history(n=12, T=1000, seed=42):
    """Random phases."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (T, n))


def _positions(n=12):
    side = int(np.ceil(np.sqrt(n)))
    pos = []
    for i in range(n):
        row, col = divmod(i, side)
        pos.append([col, row])
    return np.array(pos, dtype=float)


def _adjacency_all(n=12):
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def test_kuramoto_order_synced():
    """Synced history should score ~1.0."""
    ev = KuramotoOrderEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score > 0.95


def test_kuramoto_order_random():
    """Random history should score near 0."""
    ev = KuramotoOrderEvaluator()
    result = ev.evaluate(_random_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score < 0.3


def test_velocity_synchrony_synced():
    """Synced (constant phase) -> zero velocity variance -> score ~1.0."""
    ev = VelocitySynchronyEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score > 0.9


def test_spatial_alignment_synced():
    """Synced history should get high spatial alignment score."""
    ev = SpatialAlignmentEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score > 0.5


def test_spatial_alignment_has_evidence():
    """Spatial alignment result should contain mean_phase_spread in evidence."""
    ev = SpatialAlignmentEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert "mean_phase_spread" in result.evidence


def test_pairwise_entrainment_synced():
    """Synced history -> all pairs entrained -> score ~1.0."""
    ev = PairwiseEntrainmentEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score > 0.9


def test_pairwise_entrainment_random():
    """Random history -> few pairs entrained -> low score."""
    ev = PairwiseEntrainmentEvaluator()
    result = ev.evaluate(_random_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert result.sync_score < 0.3


def test_panel_majority_synced():
    """Synced history should get majority vote = True."""
    panel = EvaluatorPanel()
    results = panel.evaluate_all(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    verdict = panel.aggregate(results, "majority")
    assert verdict is True


def test_panel_majority_random():
    """Random history should get majority vote = False."""
    panel = EvaluatorPanel()
    results = panel.evaluate_all(_random_history(), _positions(), _adjacency_all(), sigma=0.5)
    verdict = panel.aggregate(results, "majority")
    assert verdict is False


def test_evaluator_result_has_evidence():
    """Each result should have evidence dict."""
    ev = KuramotoOrderEvaluator()
    result = ev.evaluate(_synced_history(), _positions(), _adjacency_all(), sigma=0.5)
    assert isinstance(result.evidence, dict)
