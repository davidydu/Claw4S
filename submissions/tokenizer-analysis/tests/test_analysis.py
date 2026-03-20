# tests/test_analysis.py
import os
from src.analysis import run_analysis

def test_run_analysis_returns_structured_results():
    """Integration test: run analysis on minimal data."""
    results = run_analysis(
        pairs=["en-fr"],
        max_sentences=10,
        output_dir="/tmp/tokenizer_test_results",
    )
    assert "metadata" in results
    assert "results" in results
    assert len(results["results"]) > 0
    first = results["results"][0]
    assert "tokenizer" in first
    assert "language" in first
    assert "fertility" in first
    assert "bpc" in first
    assert "cross_lingual_tax" in first
    # Verify output file was written
    assert os.path.exists("/tmp/tokenizer_test_results/results.json")
