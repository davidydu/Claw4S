# tests/test_report.py
from src.report import generate_report


def _make_stat(matchup, memory, preset, *, delta, margin):
    """Create one synthetic stats row for report generation tests."""
    return {
        "matchup": matchup,
        "memory": memory,
        "preset": preset,
        "shocks": False,
        "majority_collusion_rate": 0.0,
        "unanimous_collusion_rate": 0.0,
        "auditor_agreement_rate": 1.0,
        "avg_price": 1.5,
        "nash_price": 1.6,
        "collusion_index": delta,
        "p_value": 1.0,
        "p_value_corrected": 1.0,
        "avg_auditor_scores": {
            "margin": margin,
            "deviation_punishment": 0.0,
            "counterfactual": 0.0,
            "welfare": 0.0,
        },
    }


def test_generate_report_key_findings_use_collusion_index_delta():
    """Key findings should rank conditions by Delta, not margin auditor score."""
    stats = [
        _make_stat("QQ", 1, "e-commerce", delta=-0.400, margin=0.200),
        _make_stat("SS", 3, "commodity", delta=-0.050, margin=0.000),
    ]
    analysis_data = {"records": [{}, {}], "statistics": stats}

    report = generate_report(analysis_data)

    assert "Least collusive condition:** QQ/M1/e-commerce" in report
    assert "Delta: -0.400" in report
    assert "margin score" not in report
