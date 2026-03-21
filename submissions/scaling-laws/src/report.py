# src/report.py
"""Markdown report generator for scaling-laws analysis results."""

from __future__ import annotations

import os
from datetime import datetime


def generate_report(results: dict) -> str:
    """Generate a markdown report string from the analysis results dict.

    Parameters
    ----------
    results:
        Output of ``run_full_analysis()``.  Safe access patterns are used
        throughout so a partially-failed run still produces a useful report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []

    # ------------------------------------------------------------------ #
    # Title & metadata
    # ------------------------------------------------------------------ #
    metadata = results.get("metadata", {})
    timestamp = metadata.get("timestamp", datetime.utcnow().isoformat())
    seed = metadata.get("seed", "n/a")

    lines.append("# Scaling Laws Analysis Report")
    lines.append(f"\n_Generated: {timestamp} | seed={seed}_\n")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    lines.append("## Summary\n")
    lines.append(
        "We verified neural scaling laws using published data from Cerebras-GPT "
        "(7 sizes) and Pythia (8 sizes). Three loss-scaling formulations (Kaplan, "
        "Chinchilla, Corrected) were fit with parametric bootstrapping (B=1000) and "
        "compared via AIC/BIC. Task-level accuracy scaling was modelled with a "
        "bounded power-law and a sigmoid, and a piecewise breakpoint was detected "
        "for each benchmark. Cross-metric correlation between loss improvement and "
        "accuracy improvement, extrapolation risk, and cross-family transfer error "
        "were evaluated to characterise when scaling predictions generalise.\n"
    )

    # ------------------------------------------------------------------ #
    # Loss Scaling Results
    # ------------------------------------------------------------------ #
    lines.append("## Loss Scaling Results\n")
    loss_scaling = results.get("loss_scaling", {})

    if loss_scaling:
        # Determine best model by AIC
        valid = {
            k: v for k, v in loss_scaling.items() if isinstance(v.get("aic"), float)
        }
        best_aic_name = min(valid, key=lambda k: valid[k]["aic"]) if valid else None

        header = "| Formulation | alpha | alpha CI | L_inf | adj-R² | AIC | BIC |"
        sep    = "|-------------|-------|----------|-------|--------|-----|-----|"
        lines.append(header)
        lines.append(sep)

        display_order = ["kaplan", "chinchilla", "corrected"]
        all_keys = display_order + [k for k in loss_scaling if k not in display_order]

        for name in all_keys:
            if name not in loss_scaling:
                continue
            fit = loss_scaling[name]
            params = fit.get("params", {})
            ci = fit.get("ci", {})

            alpha = params.get("alpha", float("nan"))
            l_inf = params.get("L_inf", params.get("l_inf", params.get("E", float("nan"))))

            # CI for alpha: expect {"alpha": [lo, hi]} or {"alpha_lo": ..., "alpha_hi": ...}
            alpha_ci_raw = ci.get("alpha", ci.get("alpha_ci", None))
            if isinstance(alpha_ci_raw, (list, tuple)) and len(alpha_ci_raw) == 2:
                alpha_ci_str = f"[{alpha_ci_raw[0]:.4f}, {alpha_ci_raw[1]:.4f}]"
            elif "alpha_lo" in ci and "alpha_hi" in ci:
                alpha_ci_str = f"[{ci['alpha_lo']:.4f}, {ci['alpha_hi']:.4f}]"
            else:
                alpha_ci_str = "n/a"

            adj_r2 = fit.get("adj_r_squared", float("nan"))
            aic = fit.get("aic", float("nan"))
            bic = fit.get("bic", float("nan"))

            star = " *" if name == best_aic_name else ""
            label = name.capitalize() + star

            def _fmt(v):
                try:
                    return f"{v:.4f}"
                except (TypeError, ValueError):
                    return "n/a"

            def _fmt3(v):
                try:
                    return f"{v:.3f}"
                except (TypeError, ValueError):
                    return "n/a"

            row = (
                f"| {label} | {_fmt(alpha)} | {alpha_ci_str} "
                f"| {_fmt(l_inf)} | {_fmt3(adj_r2)} | {_fmt(aic)} | {_fmt(bic)} |"
            )
            lines.append(row)

        if best_aic_name:
            lines.append(
                f"\n_* Best model by AIC: **{best_aic_name.capitalize()}**._\n"
            )
        else:
            lines.append("")
    else:
        lines.append("_Loss scaling results unavailable._\n")

    # ------------------------------------------------------------------ #
    # Task Scaling Results
    # ------------------------------------------------------------------ #
    lines.append("## Task Scaling Results\n")
    task_scaling = results.get("task_scaling", {})

    if task_scaling:
        header = "| Task | Power-Law adj-R² | Sigmoid adj-R² | Breakpoint Index |"
        sep    = "|------|-----------------|----------------|------------------|"
        lines.append(header)
        lines.append(sep)

        for task_name, task_result in task_scaling.items():
            bpl = task_result.get("bounded_power_law", {})
            sig = task_result.get("sigmoid", {})
            bp  = task_result.get("breakpoint", {})

            pl_r2  = bpl.get("adj_r_squared", float("nan"))
            sig_r2 = sig.get("adj_r_squared", float("nan"))
            bp_idx = bp.get("breakpoint_idx", "n/a")

            def _fmt3(v):
                try:
                    return f"{v:.3f}"
                except (TypeError, ValueError):
                    return "n/a"

            row = (
                f"| {task_name} | {_fmt3(pl_r2)} | {_fmt3(sig_r2)} "
                f"| {bp_idx} |"
            )
            lines.append(row)

        lines.append("")
    else:
        lines.append("_Task scaling results unavailable._\n")

    # ------------------------------------------------------------------ #
    # Cross-Metric Correlation
    # ------------------------------------------------------------------ #
    lines.append("## Cross-Metric Correlation\n")
    cross = results.get("cross_metric", {})

    if cross:
        pr   = cross.get("pearson_r",  float("nan"))
        pp   = cross.get("pearson_p",  float("nan"))
        sr   = cross.get("spearman_r", float("nan"))
        sp   = cross.get("spearman_p", float("nan"))

        def _fmt3(v):
            try:
                return f"{v:.3f}"
            except (TypeError, ValueError):
                return "n/a"

        lines.append(
            f"Pearson r = {_fmt3(pr)} (p = {_fmt3(pp)}); "
            f"Spearman rho = {_fmt3(sr)} (p = {_fmt3(sp)}) "
            f"between delta-loss and delta-accuracy across model pairs.\n"
        )
    else:
        lines.append("_Cross-metric correlation results unavailable._\n")

    # ------------------------------------------------------------------ #
    # Extrapolation Risk
    # ------------------------------------------------------------------ #
    lines.append("## Extrapolation Risk\n")
    extrap = results.get("extrapolation", {})

    if extrap:
        loss_mape = extrap.get("loss_mape", float("nan"))
        task_mape = extrap.get("task_mape_avg", float("nan"))

        def _fmt3(v):
            try:
                return f"{v:.3f}"
            except (TypeError, ValueError):
                return "n/a"

        try:
            ratio = loss_mape / task_mape
            ratio_str = f"{ratio:.3f}"
        except (TypeError, ZeroDivisionError):
            ratio_str = "n/a"

        lines.append(
            f"Loss MAPE = {_fmt3(loss_mape)}; "
            f"Average Task MAPE = {_fmt3(task_mape)}; "
            f"Ratio (loss / task) = {ratio_str}.\n"
        )
    else:
        lines.append("_Extrapolation risk results unavailable._\n")

    # ------------------------------------------------------------------ #
    # Cross-Family Transfer
    # ------------------------------------------------------------------ #
    lines.append("## Cross-Family Transfer\n")
    cf = results.get("cross_family", {})

    if cf:
        avg_err = cf.get("avg_transfer_error", float("nan"))

        def _fmt3(v):
            try:
                return f"{v:.3f}"
            except (TypeError, ValueError):
                return "n/a"

        lines.append(
            f"Average transfer error (Cerebras-GPT → Pythia) = {_fmt3(avg_err)}.\n"
        )
    else:
        lines.append("_Cross-family transfer results unavailable._\n")

    # ------------------------------------------------------------------ #
    # Methodology
    # ------------------------------------------------------------------ #
    lines.append("## Methodology\n")
    lines.append(
        "Loss-scaling parameters were estimated by nonlinear least-squares with "
        "parametric bootstrap (B=1000) to construct 95% confidence intervals. "
        "Model selection used AIC and BIC to penalise over-parameterisation. "
        "Task accuracy was fit with a bounded power-law (acc(N) = 1 − a·N^(−α)) "
        "and a sigmoid in log-N space; the better fit was chosen by adjusted R². "
        "Piecewise linear breakpoint detection identified phase transitions for "
        "each benchmark. Cross-metric correlation used paired (loss, accuracy) "
        "improvements across consecutive model sizes.\n"
    )

    # ------------------------------------------------------------------ #
    # Limitations
    # ------------------------------------------------------------------ #
    lines.append("## Limitations\n")
    lines.append(
        "- **Small sample size** (n=7 for Cerebras-GPT, n=8 for Pythia) limits "
        "statistical power of all fits.\n"
        "- **HellaSwag excluded from Pythia** data due to missing evaluations, "
        "reducing comparability.\n"
        "- **Chinchilla identifiability**: when D ∝ N the joint (α, β) parameters "
        "are not separately identifiable from cross-entropy alone.\n"
        "- **Breakpoint detection** has low statistical power at these sample sizes; "
        "detected breakpoints should be interpreted cautiously.\n"
    )

    return "\n".join(lines)


def save_report(report: str, path: str = "results/report.md") -> None:
    """Write the report string to *path*, creating parent directories as needed.

    Parameters
    ----------
    report:
        Markdown string returned by :func:`generate_report`.
    path:
        Destination file path.  Defaults to ``results/report.md``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
