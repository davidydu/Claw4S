"""Tests for result validation script strictness."""

import json
from pathlib import Path

from validate import validate_results


NOISE_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
NOISE_KEYS = [f"{n:.0%}" for n in NOISE_FRACS]
SEEDS = [42, 43, 44]
ARCH_CONFIGS = {
    "shallow-wide": (1, 200),
    "medium": (2, 70),
    "deep-narrow": (4, 35),
}
WIDTHS = [16, 32, 64, 128, 256]


def _build_valid_raw_results() -> list[dict]:
    rows: list[dict] = []

    for arch_name, (depth, width) in ARCH_CONFIGS.items():
        for noise in NOISE_FRACS:
            for seed in SEEDS:
                train = 0.90 - 0.4 * noise
                test = 0.92 - 0.1 * noise
                rows.append(
                    {
                        "arch": arch_name,
                        "depth": depth,
                        "width": width,
                        "n_params": 4000 + depth * width,
                        "noise_frac": noise,
                        "seed": seed,
                        "train_acc": round(train, 4),
                        "test_acc": round(test, 4),
                        "gen_gap": round(train - test, 4),
                        "wall_seconds": 0.2,
                    }
                )

    for width in WIDTHS:
        for noise in NOISE_FRACS:
            for seed in SEEDS:
                train = 0.88 - 0.45 * noise
                test = 0.90 - 0.12 * noise
                rows.append(
                    {
                        "arch": f"d2_w{width}",
                        "depth": 2,
                        "width": width,
                        "n_params": 3000 + width,
                        "noise_frac": noise,
                        "seed": seed,
                        "train_acc": round(train, 4),
                        "test_acc": round(test, 4),
                        "gen_gap": round(train - test, 4),
                        "wall_seconds": 0.2,
                    }
                )

    return rows


def _build_valid_summary() -> dict:
    arch_sweep = {}
    for arch_name in ARCH_CONFIGS:
        arch_sweep[arch_name] = {}
        for noise_key, noise in zip(NOISE_KEYS, NOISE_FRACS):
            train = 0.90 - 0.4 * noise
            test = 0.92 - 0.1 * noise
            arch_sweep[arch_name][noise_key] = {
                "test_acc_mean": round(test, 4),
                "test_acc_std": 0.01,
                "train_acc_mean": round(train, 4),
                "train_acc_std": 0.01,
                "gen_gap_mean": round(train - test, 4),
                "gen_gap_std": 0.01,
                "n_runs": 3,
            }

    width_sweep = {}
    for width in WIDTHS:
        wname = f"d2_w{width}"
        width_sweep[wname] = {}
        for noise_key, noise in zip(NOISE_KEYS, NOISE_FRACS):
            train = 0.88 - 0.45 * noise
            test = 0.90 - 0.12 * noise
            width_sweep[wname][noise_key] = {
                "test_acc_mean": round(test, 4),
                "test_acc_std": 0.01,
                "train_acc_mean": round(train, 4),
                "train_acc_std": 0.01,
                "gen_gap_mean": round(train - test, 4),
                "gen_gap_std": 0.01,
                "n_runs": 3,
            }

    return {
        "architecture_sweep": arch_sweep,
        "width_sweep": width_sweep,
        "findings": ["f1", "f2", "f3"],
    }


def _write_png_placeholder(path: Path) -> None:
    # Validation checks non-empty files and warns if very small (<5KB).
    # A byte payload is sufficient for this structural test.
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 6000)


def _write_valid_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw_results.json").write_text(
        json.dumps(_build_valid_raw_results(), indent=2)
    )
    (results_dir / "summary.json").write_text(
        json.dumps(_build_valid_summary(), indent=2)
    )
    _write_png_placeholder(results_dir / "arch_sweep.png")
    _write_png_placeholder(results_dir / "width_sweep.png")


def test_validate_results_accepts_complete_artifacts(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_valid_results_dir(results_dir)

    errors, warnings = validate_results(str(results_dir))

    assert errors == []
    assert warnings == []


def test_validate_results_rejects_duplicate_and_missing_run(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_valid_results_dir(results_dir)

    raw_path = results_dir / "raw_results.json"
    raw = json.loads(raw_path.read_text())
    raw[-1] = raw[0]
    raw_path.write_text(json.dumps(raw, indent=2))

    errors, _ = validate_results(str(results_dir))

    assert errors
    assert any(
        "duplicate" in err.lower() or "missing expected run" in err.lower()
        for err in errors
    )


def test_validate_results_requires_exact_run_count(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_valid_results_dir(results_dir)

    raw_path = results_dir / "raw_results.json"
    raw = json.loads(raw_path.read_text())
    raw_path.write_text(json.dumps(raw[:-1], indent=2))

    errors, _ = validate_results(str(results_dir))

    assert errors
    assert any("expected exactly 168" in err.lower() for err in errors)
