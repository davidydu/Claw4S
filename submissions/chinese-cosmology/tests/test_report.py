from src.report import generate_report


def test_report_includes_correlation_inference_section():
    analysis = {
        "records": [{}],
        "evaluator_results": {"career": []},
        "statistics": {
            "n_records": 100,
            "correlation": {
                "career": {
                    "bazi_ziwei": 0.1,
                    "bazi_wuxing": 0.2,
                    "ziwei_wuxing": 0.3,
                },
            },
            "correlation_inference": {
                "career": {
                    "bazi_ziwei": {
                        "r": 0.1,
                        "ci_lower": 0.01,
                        "ci_upper": 0.19,
                        "p_value": 0.04,
                        "p_value_bonferroni": 0.6,
                        "n": 100,
                    },
                },
            },
            "domain_agreement": {
                "career": {
                    "bazi_ziwei": 0.5,
                    "bazi_wuxing": 0.5,
                    "ziwei_wuxing": 0.5,
                },
            },
            "mutual_information": {"career": 0.01},
            "temporal_patterns": [],
            "conditional_agreement": {},
        },
    }

    report = generate_report(analysis)
    assert "Correlation Inference" in report
    assert "95% CI" in report
    assert "Bonferroni p" in report
    assert "0.1000" in report
