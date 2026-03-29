"""Run the full feature attribution consistency experiment.

Must be executed from the submission directory:
    submissions/feature-attribution/
"""

import os
import sys

def _ensure_submission_dir(script_file: str) -> str:
    """Ensure commands run from the submission directory.

    This keeps relative paths (e.g., results/) stable even when users launch
    run.py from another working directory.
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    if os.path.abspath(os.getcwd()) != script_dir:
        print(f"Changing working directory to {script_dir}")
        os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return script_dir


def main() -> int:
    """Run the experiment and print headline summary metrics."""
    _ensure_submission_dir(__file__)
    from src.experiment import run_experiment

    results = run_experiment()
    print("\nExperiment complete.")
    print(f"Overall mean Spearman rho: {results['summary']['overall_mean_rho']:.4f}")
    print(f"Substantial disagreement: {results['summary']['substantial_disagreement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
