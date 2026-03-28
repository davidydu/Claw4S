"""Regression tests for visualization output."""

from pathlib import Path
import warnings

from pyparsing import PyparsingDeprecationWarning

# Matplotlib 3.10 emits pyparsing deprecation noise under Python 3.13.
warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)

from src.analysis import run_analysis
from src.visualize import generate_all_figures


def test_generate_all_figures_emits_no_user_warnings(tmp_path):
    """Figure generation should be clean for agent-facing runs."""
    data = run_analysis(seed=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = generate_all_figures(data, output_dir=str(tmp_path))

    layout_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "tight_layout" in str(w.message)
    ]
    assert not layout_warnings, [str(w.message) for w in layout_warnings]

    for path in paths:
        assert Path(path).is_file()
