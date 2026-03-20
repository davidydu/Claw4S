"""Run the full cross-lingual tokenizer analysis and generate report."""

from src.analysis import run_analysis
from src.report import generate_report

results = run_analysis(max_sentences=200)
report = generate_report(results)
print(report)
