"""Tests for src/data.py -- hardcoded benchmark data."""

from src.data import (
    get_bigbench_tasks,
    get_bigbench_data,
    get_mmlu_data,
    get_model_families,
    BIGBENCH_TASKS,
    MMLU_DATA,
)


def test_bigbench_tasks_exist():
    """At least 4 BIG-Bench tasks are defined."""
    tasks = get_bigbench_tasks()
    assert len(tasks) >= 4, f"Expected >= 4 tasks, got {len(tasks)}"


def test_bigbench_data_structure():
    """Each BIG-Bench data entry has required fields."""
    for task_name in get_bigbench_tasks():
        entries = get_bigbench_data(task_name)
        assert len(entries) >= 3, f"Task {task_name}: expected >= 3 data points"
        for entry in entries:
            assert "model" in entry, f"Missing 'model' in {task_name}"
            assert "params_b" in entry, f"Missing 'params_b' in {task_name}"
            assert "accuracy" in entry, f"Missing 'accuracy' in {task_name}"
            assert "family" in entry, f"Missing 'family' in {task_name}"


def test_mmlu_data_exists():
    """At least 3 model families have MMLU data."""
    families = get_model_families(MMLU_DATA)
    assert len(families) >= 3, f"Expected >= 3 MMLU families, got {len(families)}"


def test_param_counts_positive():
    """All parameter counts are positive."""
    for task_name in get_bigbench_tasks():
        for entry in get_bigbench_data(task_name):
            assert entry["params_b"] > 0, (
                f"Non-positive params in {task_name}: {entry['params_b']}"
            )
    for entry in MMLU_DATA:
        assert entry["params_b"] > 0, (
            f"Non-positive params in MMLU: {entry['params_b']}"
        )


def test_accuracy_in_range():
    """All accuracy values are in [0, 1]."""
    for task_name in get_bigbench_tasks():
        for entry in get_bigbench_data(task_name):
            assert 0.0 <= entry["accuracy"] <= 1.0, (
                f"Out-of-range accuracy in {task_name}: {entry['accuracy']}"
            )
    for entry in MMLU_DATA:
        assert 0.0 <= entry["accuracy"] <= 1.0, (
            f"Out-of-range MMLU accuracy: {entry['accuracy']}"
        )


def test_bigbench_tasks_have_citations():
    """Each task has a source citation."""
    for task_name, task_info in BIGBENCH_TASKS.items():
        assert "citation" in task_info, f"Missing citation for {task_name}"
        assert len(task_info["citation"]) > 10, (
            f"Citation too short for {task_name}"
        )


def test_bigbench_tasks_have_metric_type():
    """Each task has a metric_type field (discontinuous or continuous)."""
    for task_name, task_info in BIGBENCH_TASKS.items():
        assert "metric_type" in task_info, f"Missing metric_type for {task_name}"
        assert task_info["metric_type"] in ("exact_match", "multiple_choice"), (
            f"Invalid metric_type for {task_name}: {task_info['metric_type']}"
        )


def test_mmlu_data_has_model_names():
    """All MMLU entries have model names."""
    for entry in MMLU_DATA:
        assert "model" in entry and len(entry["model"]) > 0
