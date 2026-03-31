"""Helpers for choosing representative Zipf fit plots."""


def select_representative_analyses(
    analyses: list[dict],
    max_cases: int = 4,
) -> list[dict]:
    """Select representative analyses with coverage across corpus types."""
    if max_cases <= 0:
        return []

    quotas = {
        "natural_language": max_cases // 2,
        "code": max_cases // 2,
    }
    selected: list[dict] = []
    selected_labels: set[str] = set()
    corpus_type_counts = {key: 0 for key in quotas}
    seen_corpora_by_type = {key: set() for key in quotas}

    for analysis in analyses:
        corpus_type = analysis.get("corpus_type")
        corpus = analysis.get("corpus")
        label = analysis.get("label")
        if not label or not corpus or corpus_type not in quotas:
            continue
        if corpus_type_counts[corpus_type] >= quotas[corpus_type]:
            continue
        if corpus in seen_corpora_by_type[corpus_type]:
            continue
        selected.append(analysis)
        selected_labels.add(label)
        corpus_type_counts[corpus_type] += 1
        seen_corpora_by_type[corpus_type].add(corpus)
        if len(selected) >= max_cases:
            return selected

    for analysis in analyses:
        label = analysis.get("label")
        if not label or label in selected_labels:
            continue
        selected.append(analysis)
        selected_labels.add(label)
        if len(selected) >= max_cases:
            break

    return selected


def build_zipf_fit_plot_plan(
    analyses: list[dict],
    plot_data_by_label: dict[str, dict],
    max_cases: int = 4,
) -> list[dict]:
    """Return plot inputs for representative Zipf fit figures."""
    plan = []
    for analysis in select_representative_analyses(analyses, max_cases=max_cases):
        label = analysis["label"]
        plot_data = plot_data_by_label.get(label)
        if plot_data is None:
            continue
        plan.append(
            {
                "label": label,
                "fit_params": analysis["global_fit"],
                "ranks": plot_data["ranks"],
                "freqs": plot_data["freqs"],
            }
        )
    return plan
