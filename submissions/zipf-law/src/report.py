# src/report.py
"""Generate a markdown report from Zipf analysis results.

Writes the report to output_path and returns the report as a string.
"""

import os
import statistics

from scipy.stats import mannwhitneyu


def generate_report(results: dict, output_path: str = "results/report.md") -> str:
    """Generate a markdown summary report and write it to output_path."""
    meta = results["metadata"]
    analyses = results["analyses"]
    correlation = results.get("correlation", {})

    lines = [
        "# Zipf's Law Breakdown in Token Distributions",
        "",
        f"**Generated:** {meta['timestamp']}",
        f"**Tokenizers:** {meta['num_tokenizers']}",
        f"**Corpora:** {meta['num_corpora']}",
        f"**Random seed:** {meta['seed']}",
        "",
    ]

    # --- Global Zipf Fits Table ---
    lines.append("## Global Zipf-Mandelbrot Fits")
    lines.append("")
    lines.append("| Tokenizer | Corpus | Type | Alpha | q | R^2 | Compression | Breakpoints |")
    lines.append("|-----------|--------|------|-------|---|-----|-------------|-------------|")

    for a in analyses:
        gf = a["global_fit"]
        bp_str = ", ".join(str(b) for b in a.get("breakpoints", []))
        if not bp_str:
            bp_str = "none"
        lines.append(
            f"| {a['tokenizer']} | {a['corpus']} | {a['corpus_type']} "
            f"| {gf['alpha']:.3f} | {gf['q']:.1f} | {gf['r_squared']:.4f} "
            f"| {a.get('compression_ratio', 0):.2f} | {bp_str} |"
        )
    lines.append("")

    # --- Piecewise Exponents Table ---
    lines.append("## Piecewise Zipf Exponents (Head / Body / Tail)")
    lines.append("")
    lines.append("| Tokenizer | Corpus | Head alpha | Body alpha | Tail alpha |")
    lines.append("|-----------|--------|-----------|-----------|-----------|")

    for a in analyses:
        pw = a["piecewise_fit"]
        lines.append(
            f"| {a['tokenizer']} | {a['corpus']} "
            f"| {pw['head']['alpha']:.3f} | {pw['body']['alpha']:.3f} "
            f"| {pw['tail']['alpha']:.3f} |"
        )
    lines.append("")

    # --- Piecewise R^2 Table ---
    lines.append("## Piecewise R^2 (Goodness of Fit by Region)")
    lines.append("")
    lines.append("| Tokenizer | Corpus | Head R^2 | Body R^2 | Tail R^2 |")
    lines.append("|-----------|--------|---------|---------|---------|")

    for a in analyses:
        pw = a["piecewise_fit"]
        lines.append(
            f"| {a['tokenizer']} | {a['corpus']} "
            f"| {pw['head']['r_squared']:.4f} | {pw['body']['r_squared']:.4f} "
            f"| {pw['tail']['r_squared']:.4f} |"
        )
    lines.append("")

    # --- Corpus Type Summary ---
    lines.append("## Summary by Corpus Type")
    lines.append("")

    corpus_types = sorted(set(a["corpus_type"] for a in analyses))
    for ctype in corpus_types:
        ctype_analyses = [a for a in analyses if a["corpus_type"] == ctype]
        alphas = [a["global_fit"]["alpha"] for a in ctype_analyses]
        r2s = [a["global_fit"]["r_squared"] for a in ctype_analyses]
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0
        avg_r2 = sum(r2s) / len(r2s) if r2s else 0
        std_alpha = statistics.stdev(alphas) if len(alphas) >= 2 else 0.0
        std_r2 = statistics.stdev(r2s) if len(r2s) >= 2 else 0.0

        lines.append(f"### {ctype.replace('_', ' ').title()}")
        lines.append(f"- Average alpha: {avg_alpha:.3f} (std dev: {std_alpha:.3f})")
        lines.append(f"- Average R^2: {avg_r2:.4f} (std dev: {std_r2:.4f})")
        lines.append(f"- N analyses: {len(ctype_analyses)}")
        lines.append("")

    # --- Correlation Analysis ---
    lines.append("## Correlation: Zipf Exponent vs Compression Ratio")
    lines.append("")
    if correlation:
        lines.append(f"- **Pearson r:** {correlation.get('pearson_r', 0):.4f} "
                      f"(p = {correlation.get('pearson_p', 1):.4f})")
        lines.append(f"- **Spearman rho:** {correlation.get('spearman_r', 0):.4f} "
                      f"(p = {correlation.get('spearman_p', 1):.4f})")
    else:
        lines.append("- Insufficient data for correlation analysis.")
    lines.append("")

    # --- Key Findings ---
    lines.append("## Key Findings")
    lines.append("")

    # Auto-detect findings
    nl_analyses = [a for a in analyses if a["corpus_type"] == "natural_language"]
    code_analyses = [a for a in analyses if a["corpus_type"] == "code"]

    if nl_analyses and code_analyses:
        nl_alphas = [a["global_fit"]["alpha"] for a in nl_analyses]
        code_alphas = [a["global_fit"]["alpha"] for a in code_analyses]
        nl_avg_alpha = sum(nl_alphas) / len(nl_alphas)
        code_avg_alpha = sum(code_alphas) / len(code_alphas)
        nl_std = statistics.stdev(nl_alphas) if len(nl_alphas) >= 2 else 0.0
        code_std = statistics.stdev(code_alphas) if len(code_alphas) >= 2 else 0.0
        diff = nl_avg_alpha - code_avg_alpha
        if abs(diff) > 0.05:
            direction = "higher" if diff > 0 else "lower"
            lines.append(
                f"- Natural language has {direction} average Zipf exponent "
                f"({nl_avg_alpha:.3f} +/- {nl_std:.3f}) than code "
                f"({code_avg_alpha:.3f} +/- {code_std:.3f}), "
                f"indicating {'steeper' if diff > 0 else 'flatter'} "
                f"frequency distributions."
            )
        # Mann-Whitney U test for code vs NL alpha difference
        if len(nl_alphas) >= 2 and len(code_alphas) >= 2:
            u_stat, u_pvalue = mannwhitneyu(
                code_alphas, nl_alphas, alternative="two-sided"
            )
            lines.append(
                f"- Mann-Whitney U test (code vs NL alpha): "
                f"U = {u_stat:.1f}, p = {u_pvalue:.4f}"
                + (" (significant at p < 0.05)" if u_pvalue < 0.05 else " (not significant at p < 0.05)")
                + "."
            )

    if correlation and abs(correlation.get("pearson_r", 0)) > 0.3:
        direction = "positive" if correlation["pearson_r"] > 0 else "negative"
        strength = "strong" if abs(correlation["pearson_r"]) > 0.7 else "moderate"
        lines.append(
            f"- {strength.capitalize()} {direction} correlation between Zipf exponent "
            f"and compression ratio (r = {correlation['pearson_r']:.3f}), suggesting "
            f"that {'more' if correlation['pearson_r'] > 0 else 'less'} Zipfian "
            f"distributions are associated with {'better' if correlation['pearson_r'] > 0 else 'worse'} "
            f"tokenizer compression."
        )

    # Tail breakdown finding (aggregate)
    tail_zero_count = sum(
        1 for a in analyses
        if abs(a["piecewise_fit"]["tail"]["alpha"]) < 0.01
    )
    if tail_zero_count > 0:
        lines.append(
            f"- In {tail_zero_count}/{len(analyses)} analyses, the tail region "
            f"(bottom 10% of ranks) has near-zero alpha, indicating a flat "
            f"frequency plateau where many tokens appear exactly once. "
            f"This is the primary mode of Zipf breakdown."
        )

    # Extreme tail alphas (notable outliers)
    extreme_tails = [
        a for a in analyses
        if a["piecewise_fit"]["tail"]["alpha"] > 2.0
    ]
    if extreme_tails:
        for a in extreme_tails:
            pw = a["piecewise_fit"]
            lines.append(
                f"- {a['corpus']} ({a['tokenizer']}): extreme tail alpha="
                f"{pw['tail']['alpha']:.1f} (vs head={pw['head']['alpha']:.3f}), "
                f"indicating severe frequency collapse in rare tokens."
            )

    # Head-body divergence (show top 3 most divergent)
    divergences = []
    for a in analyses:
        pw = a["piecewise_fit"]
        div = abs(pw["body"]["alpha"] - pw["head"]["alpha"])
        if div > 0.3:
            divergences.append((a, div))
    divergences.sort(key=lambda x: -x[1])
    for a, div in divergences[:3]:
        pw = a["piecewise_fit"]
        lines.append(
            f"- {a['corpus']} ({a['tokenizer']}): head-body alpha divergence="
            f"{div:.3f} (head={pw['head']['alpha']:.3f}, "
            f"body={pw['body']['alpha']:.3f}), suggesting different "
            f"frequency regimes for common vs mid-frequency tokens."
        )

    lines.append("")

    # --- Limitations ---
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Corpus sizes are small (200 sentences/samples per language), "
                 "limiting statistical power for tail analysis.")
    lines.append("- Only BPE-family tokenizers tested (no unigram/WordPiece comparison).")
    lines.append("- OLS on log-log data has known biases vs MLE for power-law fitting; "
                 "results are valid for comparative analysis but absolute alpha values "
                 "may be biased.")
    lines.append("- Piecewise region boundaries (10%/90%) are arbitrary; different "
                 "boundaries may yield different breakpoint characterizations.")
    lines.append("")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")

    return report
