import json
import subprocess
import sys
from pathlib import Path


def _make_minimal_results(path: Path) -> None:
    domains = ["career", "wealth", "relationships", "health", "overall"]
    systems = ["bazi", "ziwei", "wuxing"]

    record = {"datetime": "2000-01-01T00:00:00"}
    for system in systems:
        for domain in domains:
            record[f"{system}_{domain}"] = 0.5

    data = {
        "metadata": {
            "num_charts": 1,
            "systems": systems,
            "domains": domains,
        },
        "records": [record],
        "statistics_summary": {
            "correlation": {"career": {"bazi_ziwei": 0.0}},
            "domain_agreement": {"career": {"bazi_ziwei": 1.0}},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_validate_accepts_results_file_flag(tmp_path):
    results_file = tmp_path / "sample_results.json"
    _make_minimal_results(results_file)

    validate_path = Path(__file__).resolve().parents[1] / "validate.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(validate_path),
            "--results-file",
            str(results_file),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Validation passed." in proc.stdout
