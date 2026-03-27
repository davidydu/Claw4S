"""Tests for experiment runner."""

from src.experiment import run_single_experiment, aggregate_results


def test_single_experiment_structure():
    """Single experiment returns expected keys."""
    result = run_single_experiment(hidden_width=16, seed=42,
                                   shift_magnitudes=[0.0, 1.0])
    assert 'hidden_width' in result
    assert 'seed' in result
    assert 'n_params' in result
    assert 'shifts' in result
    assert result['hidden_width'] == 16
    assert result['seed'] == 42
    assert '0.0' in result['shifts']
    assert '1.0' in result['shifts']


def test_single_experiment_metrics():
    """Single experiment has valid metric values."""
    result = run_single_experiment(hidden_width=32, seed=42,
                                   shift_magnitudes=[0.0])
    shift_data = result['shifts']['0.0']
    assert 0.0 <= shift_data['ece'] <= 1.0
    assert 0.0 <= shift_data['accuracy'] <= 1.0
    assert 0.0 <= shift_data['brier_score'] <= 2.0
    assert 0.0 <= shift_data['mean_confidence'] <= 1.0


def test_aggregate_results():
    """Aggregation computes mean and std across seeds."""
    r1 = run_single_experiment(hidden_width=16, seed=42,
                                shift_magnitudes=[0.0])
    r2 = run_single_experiment(hidden_width=16, seed=43,
                                shift_magnitudes=[0.0])
    agg = aggregate_results([r1, r2])

    assert len(agg) == 1  # One (width, shift) combo
    assert agg[0]['hidden_width'] == 16
    assert agg[0]['n_seeds'] == 2
    assert 0.0 <= agg[0]['ece_mean'] <= 1.0
    assert agg[0]['ece_std'] >= 0.0


def test_reproducibility():
    """Same seed produces same results."""
    r1 = run_single_experiment(hidden_width=64, seed=42,
                                shift_magnitudes=[0.0, 1.0])
    r2 = run_single_experiment(hidden_width=64, seed=42,
                                shift_magnitudes=[0.0, 1.0])
    for shift_key in ['0.0', '1.0']:
        assert abs(r1['shifts'][shift_key]['ece'] -
                   r2['shifts'][shift_key]['ece']) < 1e-10
