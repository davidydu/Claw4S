"""Canonical output artifact paths shared by run/validate steps."""

from __future__ import annotations

from pathlib import Path


PRIMARY_ARTIFACTS = (
    "results/results.json",
    "results/report.md",
)


def clear_primary_artifacts(base_dir: str | Path = ".") -> list[Path]:
    """Delete primary artifacts if present and return removed paths."""
    root = Path(base_dir)
    removed: list[Path] = []
    for rel_path in PRIMARY_ARTIFACTS:
        path = root / rel_path
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def primary_artifact_paths(base_dir: str | Path = ".") -> list[Path]:
    """Return absolute paths for primary artifacts under *base_dir*."""
    root = Path(base_dir)
    return [root / rel_path for rel_path in PRIMARY_ARTIFACTS]
