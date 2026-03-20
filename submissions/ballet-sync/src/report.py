"""Generate markdown report and matplotlib figures from analysis results."""


def generate_report(analysis_data):
    """Generate a markdown report from Kuramoto analysis results.

    Parameters
    ----------
    analysis_data : dict with 'records' and 'statistics' keys,
                    plus optional 'phase_transition', 'finite_size_scaling',
                    'critical_exponents', 'evaluator_agreement' top-level keys.

    Returns
    -------
    str : markdown report text
    """
    import numpy as np

    records = analysis_data.get("records", [])
    stats = analysis_data.get("statistics", [])
    topologies = sorted(set(r["topology"] for r in records))

    lines = []
    lines.append("# Emergent Synchronization in Ballet Corps: Analysis Report\n")
    lines.append(f"**Total simulations:** {len(records)}")
    lines.append(f"**Conditions:** {len(stats)}")
    lines.append(f"**Topologies:** {', '.join(topologies)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Phase transition summary — K_c per topology
    # ------------------------------------------------------------------
    lines.append("## Phase Transition Summary\n")
    lines.append("Critical coupling K_c estimated per topology (sigmoid fit, σ=0.3 + 0.8 averaged):\n")
    lines.append("| Topology | K_c (sigmoid) | K_c (susceptibility) | Bootstrap 95% CI |")
    lines.append("|----------|--------------|----------------------|-----------------|")

    pt = analysis_data.get("phase_transition", {})
    for topo in topologies:
        topo_data = pt.get(topo, {})
        kc_sig = topo_data.get("kc_sigmoid", float("nan"))
        kc_sus = topo_data.get("kc_susceptibility", float("nan"))
        ci_lo = topo_data.get("kc_ci_low", float("nan"))
        ci_hi = topo_data.get("kc_ci_high", float("nan"))
        lines.append(
            f"| {topo} "
            f"| {kc_sig:.3f} "
            f"| {kc_sus:.3f} "
            f"| [{ci_lo:.3f}, {ci_hi:.3f}] |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Analytical vs empirical K_c (all-to-all)
    # ------------------------------------------------------------------
    lines.append("## Analytical vs. Empirical K_c (All-to-All)\n")
    lines.append("For all-to-all Kuramoto with Gaussian frequencies: K_c = 2σ√(2π)/π ≈ 1.596σ\n")
    lines.append("| σ | Analytical K_c | Empirical K_c | Difference |")
    lines.append("|---|--------------|--------------|------------|")

    ata = pt.get("all-to-all", {})
    for sigma_val, kc_name in [(0.3, "kc_sigmoid_s03"), (0.8, "kc_sigmoid_s08")]:
        analytical = 2 * sigma_val * (2 * 3.141592653589793) ** 0.5 / 3.141592653589793
        empirical = ata.get(kc_name, float("nan"))
        diff = abs(empirical - analytical) if not (empirical != empirical) else float("nan")
        lines.append(
            f"| {sigma_val} | {analytical:.3f} | {empirical:.3f} | {diff:.3f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Critical exponent β table
    # ------------------------------------------------------------------
    lines.append("## Critical Exponent β\n")
    lines.append("Power-law fit r ∝ (K − K_c)^β for K > K_c:\n")
    lines.append("| Topology | σ | β | R² |")
    lines.append("|----------|---|---|-----|")

    ce = analysis_data.get("critical_exponents", {})
    for topo in topologies:
        for sigma_val in [0.3, 0.8]:
            key = f"{topo}_s{int(sigma_val*10):02d}"
            entry = ce.get(key, {})
            beta = entry.get("beta", float("nan"))
            r2 = entry.get("r_squared", float("nan"))
            lines.append(f"| {topo} | {sigma_val} | {beta:.3f} | {r2:.3f} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Evaluator agreement matrix
    # ------------------------------------------------------------------
    lines.append("## Evaluator Agreement Matrix\n")
    lines.append("Pairwise agreement rate (fraction of conditions where both evaluators agree on sync/no-sync):\n")

    ev_names = ["kuramoto_order", "spatial_alignment", "velocity_synchrony", "pairwise_entrainment"]
    ev_labels = ["KuramotoOrder", "SpatialAlign", "VelocitySync", "PairwiseEntrain"]

    # Build agreement matrix from records
    agreement = [[0.0] * 4 for _ in range(4)]
    count = 0
    for rec in records:
        scores = rec.get("evaluator_scores", {})
        verdicts = [scores.get(n, 0.0) >= 0.5 for n in ev_names]
        for a in range(4):
            for b in range(4):
                if verdicts[a] == verdicts[b]:
                    agreement[a][b] += 1
        count += 1
    if count > 0:
        agreement = [[v / count for v in row] for row in agreement]

    header = "| Evaluator | " + " | ".join(ev_labels) + " |"
    separator = "|-----------|" + "---|" * 4
    lines.append(header)
    lines.append(separator)
    for i, label in enumerate(ev_labels):
        cells = " | ".join(f"{agreement[i][j]:.0%}" for j in range(4))
        lines.append(f"| {label} | {cells} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Finite-size scaling
    # ------------------------------------------------------------------
    lines.append("## Finite-Size Scaling\n")
    lines.append("K_c(N) fit: K_c(N) = K_c(∞) + a·N^(−ν)\n")
    lines.append("| Topology | K_c(∞) | ν | K_c(N=6) | K_c(N=12) | K_c(N=24) |")
    lines.append("|----------|--------|---|---------|----------|----------|")

    fss = analysis_data.get("finite_size_scaling", {})
    for topo in topologies:
        topo_fss = fss.get(topo, {})
        kc_inf = topo_fss.get("kc_inf", float("nan"))
        nu = topo_fss.get("nu", float("nan"))
        kc6 = topo_fss.get("kc_n6", float("nan"))
        kc12 = topo_fss.get("kc_n12", float("nan"))
        kc24 = topo_fss.get("kc_n24", float("nan"))
        lines.append(
            f"| {topo} | {kc_inf:.3f} | {nu:.3f} | {kc6:.3f} | {kc12:.3f} | {kc24:.3f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    lines.append("## Key Findings\n")

    # Topology with lowest K_c (easiest to synchronize)
    if pt:
        kc_vals = {topo: pt.get(topo, {}).get("kc_sigmoid", float("inf"))
                   for topo in topologies}
        easiest = min(kc_vals, key=kc_vals.get)
        hardest = max(kc_vals, key=kc_vals.get)
        lines.append(f"- **Easiest topology to synchronize:** {easiest} (K_c = {kc_vals[easiest]:.3f})")
        lines.append(f"- **Hardest topology to synchronize:** {hardest} (K_c = {kc_vals[hardest]:.3f})")

    # Evaluator agreement summary
    avg_agreement = sum(
        sum(agreement[i][j] for j in range(4) if j != i) / 3
        for i in range(4)
    ) / 4
    lines.append(f"- **Overall inter-evaluator agreement:** {avg_agreement:.1%}")

    # All-to-all β summary
    ata_ce = ce.get("all-to-all_s03", {})
    if ata_ce:
        beta_ata = ata_ce.get("beta", float("nan"))
        lines.append(
            f"- **All-to-all critical exponent β:** {beta_ata:.3f} "
            f"(mean-field theory predicts β = 0.5)"
        )

    lines.append("")
    return "\n".join(lines)


def generate_figures(analysis_data, output_dir="results/figures"):
    """Generate 6 matplotlib PNG figures from Kuramoto analysis results.

    Figures created:
    1. phase_transition.png        — r(K) curves per topology
    2. topology_comparison.png     — K_c bar chart with error bars
    3. susceptibility.png          — χ(K) curves per topology
    4. critical_exponent.png       — log-log plot of r vs (K - K_c)
    5. finite_size_scaling.png     — K_c(N) with fit curve
    6. evaluator_agreement.png     — pairwise agreement heatmap

    Parameters
    ----------
    analysis_data : dict with 'records', 'statistics', and analysis sub-dicts
    output_dir    : directory to write PNGs (created if absent)
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    records = analysis_data.get("records", [])
    stats = analysis_data.get("statistics", [])
    pt = analysis_data.get("phase_transition", {})
    ce = analysis_data.get("critical_exponents", {})
    fss = analysis_data.get("finite_size_scaling", {})

    topologies = sorted(set(r["topology"] for r in records))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(topologies)))
    topo_color = dict(zip(topologies, colors))

    # ------------------------------------------------------------------
    # Figure 1: Phase transition curves r(K) per topology
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for topo in topologies:
        topo_stats = [s for s in stats if s["topology"] == topo]
        if not topo_stats:
            continue
        k_vals = sorted(set(s["K"] for s in topo_stats))
        r_means = []
        r_stds = []
        for k in k_vals:
            group = [s["mean_r"] for s in topo_stats if s["K"] == k]
            r_means.append(float(np.mean(group)))
            r_stds.append(float(np.std(group)))
        r_means = np.array(r_means)
        r_stds = np.array(r_stds)
        ax.plot(k_vals, r_means, marker="o", markersize=4,
                label=topo, color=topo_color[topo])
        ax.fill_between(k_vals, r_means - r_stds, r_means + r_stds,
                        alpha=0.15, color=topo_color[topo])

    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Order parameter r")
    ax.set_title("Phase Transition: r(K) per Topology")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase_transition.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: K_c bar chart per topology with error bars
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    kc_vals = []
    kc_errs = []
    topo_labels = []
    for topo in topologies:
        topo_data = pt.get(topo, {})
        kc = topo_data.get("kc_sigmoid", float("nan"))
        ci_lo = topo_data.get("kc_ci_low", kc)
        ci_hi = topo_data.get("kc_ci_high", kc)
        kc_vals.append(kc)
        kc_errs.append((kc - ci_lo, ci_hi - kc))
        topo_labels.append(topo)

    x = np.arange(len(topo_labels))
    err_lo = [e[0] for e in kc_errs]
    err_hi = [e[1] for e in kc_errs]
    bars = ax.bar(x, kc_vals, color=[topo_color[t] for t in topo_labels],
                  yerr=[err_lo, err_hi], capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, rotation=20, ha="right")
    ax.set_ylabel("Critical coupling K_c")
    ax.set_title("K_c Comparison by Topology (with 95% CI)")
    ax.set_ylim(bottom=0)

    # Annotate bars with values
    for bar, val in zip(bars, kc_vals):
        if val == val:  # not nan
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "topology_comparison.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Susceptibility χ(K) per topology
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for topo in topologies:
        topo_stats = [s for s in stats if s["topology"] == topo]
        if not topo_stats:
            continue
        k_vals = sorted(set(s["K"] for s in topo_stats))
        chi_vals = []
        for k in k_vals:
            group = [s for s in topo_stats if s["K"] == k]
            n = group[0]["n"] if group else 12
            r_vars = np.var([s["mean_r"] for s in group]) if len(group) > 1 else 0.0
            chi_vals.append(float(n * r_vars))
        ax.plot(k_vals, chi_vals, marker="s", markersize=4,
                label=topo, color=topo_color[topo])

    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Susceptibility χ = N·Var(r)")
    ax.set_title("Susceptibility χ(K) per Topology")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "susceptibility.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Critical exponent log-log plot r vs (K - K_c)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for topo in topologies:
        topo_data = pt.get(topo, {})
        kc = topo_data.get("kc_sigmoid", None)
        if kc is None or kc != kc:
            continue
        topo_stats = [s for s in stats if s["topology"] == topo]
        k_vals = np.array(sorted(set(s["K"] for s in topo_stats)))
        r_means = np.array([
            float(np.mean([s["mean_r"] for s in topo_stats if s["K"] == k]))
            for k in k_vals
        ])
        mask = (k_vals > kc) & (r_means > 0)
        dk = k_vals[mask] - kc
        r = r_means[mask]
        if len(dk) >= 2:
            ax.scatter(np.log(dk), np.log(r), s=25, label=topo,
                       color=topo_color[topo], alpha=0.8)
            # Fit line
            coeffs = np.polyfit(np.log(dk), np.log(r), 1)
            x_line = np.log(dk)
            ax.plot(x_line, np.polyval(coeffs, x_line), "--",
                    color=topo_color[topo], alpha=0.6,
                    label=f"{topo} β={coeffs[0]:.2f}")

    ax.set_xlabel("log(K − K_c)")
    ax.set_ylabel("log(r)")
    ax.set_title("Critical Exponent β: log-log plot r vs (K − K_c)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "critical_exponent.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Finite-size scaling K_c(N) with fit curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    n_vals_plot = np.array([6, 12, 24], dtype=float)

    for topo in topologies:
        topo_fss = fss.get(topo, {})
        kc_ns = [topo_fss.get(f"kc_n{n}", float("nan")) for n in [6, 12, 24]]
        valid = [(n, k) for n, k in zip(n_vals_plot, kc_ns) if k == k]
        if not valid:
            continue
        ns, ks = zip(*valid)
        ax.scatter(ns, ks, s=60, color=topo_color[topo], zorder=3)

        # Draw fit curve if we have kc_inf and nu
        kc_inf = topo_fss.get("kc_inf", float("nan"))
        a_fss = topo_fss.get("a_fss", float("nan"))
        nu = topo_fss.get("nu", float("nan"))
        if all(v == v for v in [kc_inf, a_fss, nu]):
            n_fit = np.linspace(5, 30, 100)
            kc_fit = kc_inf + a_fss * n_fit ** (-nu)
            ax.plot(n_fit, kc_fit, "--", color=topo_color[topo],
                    label=f"{topo} K_c(∞)={kc_inf:.2f}", alpha=0.8)
        else:
            ax.plot(ns, ks, "-", color=topo_color[topo], label=topo, alpha=0.7)

    ax.set_xlabel("Group size N")
    ax.set_ylabel("Critical coupling K_c(N)")
    ax.set_title("Finite-Size Scaling: K_c(N) per Topology")
    ax.set_xticks([6, 12, 24])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "finite_size_scaling.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Evaluator agreement heatmap
    # ------------------------------------------------------------------
    ev_names = ["kuramoto_order", "spatial_alignment", "velocity_synchrony", "pairwise_entrainment"]
    ev_labels = ["Kuramoto\nOrder", "Spatial\nAlign", "Velocity\nSync", "Pairwise\nEntrain"]

    agreement = np.zeros((4, 4))
    count = 0
    for rec in records:
        scores = rec.get("evaluator_scores", {})
        verdicts = [scores.get(n, 0.0) >= 0.5 for n in ev_names]
        for a in range(4):
            for b in range(4):
                if verdicts[a] == verdicts[b]:
                    agreement[a, b] += 1
        count += 1
    if count > 0:
        agreement /= count

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agreement, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(ev_labels, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(ev_labels, fontsize=9)
    ax.set_title("Evaluator Pairwise Agreement Rate")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{agreement[i, j]:.0%}", ha="center", va="center",
                    fontsize=10, color="black" if agreement[i, j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "evaluator_agreement.png"), dpi=150)
    plt.close(fig)
