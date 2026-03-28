"""Helpers for command-line entrypoints in this submission."""

from pathlib import Path


def ensure_submission_cwd(script_path: str | Path, cwd: str | Path | None = None) -> Path:
    """Require commands to run from the submission directory.

    Args:
        script_path: Path to the entrypoint script being executed.
        cwd: Optional working directory override, primarily for tests.

    Returns:
        The resolved submission directory path.

    Raises:
        RuntimeError: If the current working directory is not the submission root.
    """
    script_dir = Path(script_path).resolve().parent
    current_dir = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()

    if current_dir != script_dir:
        raise RuntimeError(f"{script_dir.name}/{Path(script_path).name} must be executed from {script_dir}")

    return script_dir
