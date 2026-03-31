"""Runtime helpers for top-level submission scripts."""

from pathlib import Path


def ensure_submission_cwd(script_file: str | Path) -> Path:
    """Exit unless the current working directory matches the script directory."""
    submission_dir = Path(script_file).resolve().parent
    cwd = Path.cwd().resolve()

    if cwd != submission_dir:
        print(
            f"ERROR: {Path(script_file).name} must be executed from the submission directory."
        )
        print(f"  Current working directory: {cwd}")
        print(f"  Expected: {submission_dir}")
        raise SystemExit(1)

    return submission_dir
