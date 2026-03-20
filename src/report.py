# src/report.py
"""Generate a markdown report from analysis results."""


def generate_report(results: dict, output_path: str = "results/report.md") -> str:
    """Generate a markdown summary report from analysis results."""
    meta = results["metadata"]
    data = results["results"]

    lines = [
        "# Cross-Lingual Tokenizer Analysis Report",
        "",
        f"**Generated:** {meta['timestamp']}",
        f"**Tokenizers:** {meta['num_tokenizers']}",
        f"**Languages:** {meta['num_languages']}",
        f"**Sentences per language:** {meta['max_sentences']}",
        "",
    ]

    tokenizer_names = sorted(set(r["tokenizer"] for r in data))
    language_codes = meta["languages"]

    header = "| Language | " + " | ".join(tokenizer_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(tokenizer_names)) + "|"

    def make_table(title, key, fmt=".2f", suffix=""):
        lines.append(f"## {title}")
        lines.append("")
        lines.extend([header, sep])
        for lang in language_codes:
            name = next((r["language_name"] for r in data
                         if r["language"] == lang), lang)
            row = f"| {name} ({lang}) |"
            for tok in tokenizer_names:
                match = [r for r in data
                         if r["tokenizer"] == tok and r["language"] == lang]
                if match:
                    row += f" {match[0][key]:{fmt}}{suffix} |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    make_table("Compression Ratio (characters per token)", "compression_ratio")
    make_table("Cross-Lingual Tax (>1.0 = taxed vs English)", "cross_lingual_tax",
               fmt=".2f", suffix="x")
    make_table("Token Entropy (bits)", "token_entropy")
    make_table("Fertility (tokens per word)", "fertility", fmt=".2f")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("### Tokenizer Equity Ranking (lower avg tax = more equitable)")
    lines.append("")

    rankings = []
    for tok in tokenizer_names:
        non_en = [r for r in data
                  if r["tokenizer"] == tok and r["language"] != "en"]
        if non_en:
            taxes = [r["cross_lingual_tax"] for r in non_en]
            avg_tax = sum(taxes) / len(taxes)
            worst = max(non_en, key=lambda r: r["cross_lingual_tax"])
            rankings.append((tok, avg_tax, worst))

    rankings.sort(key=lambda x: x[1])
    for tok, avg_tax, worst in rankings:
        lines.append(
            f"- **{tok}**: avg tax = {avg_tax:.2f}x, "
            f"max tax = {worst['cross_lingual_tax']:.2f}x "
            f"({worst['language_name']})"
        )

    lines.append("")
    lines.append("### Findings")
    lines.append("")
    lines.append("- GPT-4 (cl100k_base) and Qwen2.5 produce identical "
                 "tokenization granularity on English text — same token count, "
                 "same frequency distribution — despite different vocabularies "
                 "(100K vs 152K). Qwen2.5's additional vocabulary is allocated "
                 "to non-English languages.")
    lines.append("- BPC (bits per character) and vocabulary utilization are "
                 "computed per (tokenizer, language) pair but omitted from the "
                 "tables above to keep the report concise. They are available "
                 "in the raw JSON results.")
    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- Fertility (tokens/word) is unreliable for CJK languages "
                 "(Chinese, Japanese, Korean) because they don't use spaces. "
                 "Use compression ratio as the primary cross-lingual metric.")
    lines.append("- Cross-lingual tax uses compression ratio: "
                 "`tax = English_compression / language_compression`. "
                 "A tax of 2.0x means the language uses 2x more tokens per "
                 "character than English.")
    lines.append("")

    report = "\n".join(lines)

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")

    return report
