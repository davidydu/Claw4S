"""Tests for run.py CLI parsing helpers."""

import argparse

from run import build_arg_parser, parse_hidden_dims


def test_parse_hidden_dims_accepts_comma_list():
    """Comma-separated dims should parse into positive integer list."""
    assert parse_hidden_dims("32,64,128") == [32, 64, 128]
    assert parse_hidden_dims(" 16 , 32 ") == [16, 32]


def test_parse_hidden_dims_rejects_empty_or_nonpositive():
    """Invalid hidden-dim strings should raise argparse errors."""
    for bad in ("", " ", "32,0", "32,-1"):
        try:
            parse_hidden_dims(bad)
            assert False, f"Expected argparse.ArgumentTypeError for: {bad!r}"
        except argparse.ArgumentTypeError:
            pass


def test_arg_parser_applies_custom_overrides():
    """CLI parser should apply user-provided run overrides."""
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--seed",
            "7",
            "--hidden-dims",
            "24,48",
            "--mod-epochs",
            "100",
            "--reg-epochs",
            "120",
            "--learning-rate",
            "0.005",
            "--batch-size",
            "128",
            "--modulus",
            "89",
            "--reg-samples",
            "500",
            "--output-dir",
            "results_tmp",
            "--quiet",
        ]
    )

    assert args.seed == 7
    assert args.hidden_dims == [24, 48]
    assert args.mod_epochs == 100
    assert args.reg_epochs == 120
    assert args.learning_rate == 0.005
    assert args.batch_size == 128
    assert args.modulus == 89
    assert args.reg_samples == 500
    assert args.output_dir == "results_tmp"
    assert args.quiet is True
