"""Tests for experiment configuration helpers."""

import pytest

from src.config import parse_int_list, resolve_run_config


def test_parse_int_list():
    """Comma-separated list parser returns integers in order."""
    assert parse_int_list("64,128,256", field_name="hidden_sizes") == [64, 128, 256]


def test_parse_int_list_rejects_invalid_values():
    """Parser rejects non-positive or malformed values."""
    with pytest.raises(ValueError):
        parse_int_list("64,0,128", field_name="hidden_sizes")

    with pytest.raises(ValueError):
        parse_int_list("64,abc,128", field_name="hidden_sizes")


def test_resolve_run_config_quick_profile():
    """Quick profile should reduce runtime settings vs full defaults."""
    full = resolve_run_config(quick=False)
    quick = resolve_run_config(quick=True)

    assert quick.epochs < full.epochs
    assert len(quick.hidden_sizes) < len(full.hidden_sizes)
    assert max(quick.snapshot_epochs) <= quick.epochs
    assert quick.controls_n < full.controls_n


def test_resolve_run_config_filters_snapshot_epochs():
    """Snapshot epochs above the training horizon are dropped."""
    cfg = resolve_run_config(
        quick=False,
        epochs=200,
        snapshot_epochs=[0, 100, 500],
    )
    assert cfg.snapshot_epochs == [0, 100, 200]
