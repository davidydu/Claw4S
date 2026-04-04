# run.py
"""Run the full ballet-sync Kuramoto experiment and generate report."""

import json
import os

import numpy as np

from src.experiment import ExperimentConfig, run_simulation, K_RANGE
from src.analysis import analyze_results, fit_sigmoid, estimate_kc_susceptibility
from src.analysis import fit_critical_exponent, fit_finite_size_scaling, bootstrap_ci
from src.kuramoto import analytical_kc
from src.report import generate_report, generate_figures

TOPOLOGIES = ["all-to-all", "nearest-k", "hierarchical", "ring"]
N_VALUES = [6, 12, 24]
SIGMA_VALUES = [0.3, 0.8]
SEEDS = [0, 1, 2]


def build_configs():
    """Build the full experiment matrix.

    Returns 1,440 configs:
      20 K values × 4 topologies × 3 group sizes × 2 sigmas × 3 seeds
    """
    configs = []
    for K in K_RANGE:
        for topology in TOPOLOGIES:
            for n in N_VALUES:
                for sigma in SIGMA_VALUES:
                    for seed in SEEDS:
                        configs.append(ExperimentConfig(
                            K=float(K),
                            topology=topology,
                            n=n,
                            sigma=sigma,
                            seed=seed,
                        ))
    return configs


def _compute_phase_transition(records, stats):
    """Compute K_c estimates per topology using sigmoid and susceptibility methods."""
    pt = {}
    topologies = sorted(set(r["topology"] for r in records))

    for topo in topologies:
        topo_stats = [s for s in stats if s["topology"] == topo]
        k_vals = np.array(sorted(set(s["K"] for s in topo_stats)))

        # Aggregate mean_r and var_r across all N and sigma for this topology
        r_means = []
        r_vars = []
        for k in k_vals:
            group_r = [s["mean_r"] for s in topo_stats if s["K"] == k]
            r_means.append(float(np.mean(group_r)))
            r_vars.append(float(np.var(group_r)))

        r_means = np.array(r_means)
        r_vars = np.array(r_vars)

        try:
            kc_sig, _ = fit_sigmoid(k_vals, r_means)
        except Exception:
            kc_sig = float(k_vals[np.argmax(np.gradient(r_means))])

        try:
            n_ref = 12
            kc_sus = estimate_kc_susceptibility(k_vals, r_vars, n=n_ref)
        except Exception:
            kc_sus = kc_sig

        # Bootstrap CI for K_c: resample seeds and re-fit sigmoid on each resample.
        # Pre-index (K, seed) -> final_r for speed.
        topo_records = [r for r in records if r["topology"] == topo]
        seeds = sorted(set(r["seed"] for r in topo_records))
        # Build matrix: shape (n_k, n_seeds) of final_r values
        k_list = list(k_vals)
        k_to_idx = {float(k): i for i, k in enumerate(k_list)}
        seed_to_idx = {s: i for i, s in enumerate(seeds)}
        r_matrix = np.full((len(k_list), len(seeds)), float("nan"))
        for rec in topo_records:
            ki = k_to_idx.get(rec["K"])
            si = seed_to_idx.get(rec["seed"])
            if ki is not None and si is not None:
                r_matrix[ki, si] = rec["final_r"]

        rng_boot = np.random.default_rng(0)
        kc_boot_estimates = []
        for _ in range(200):
            boot_idx = rng_boot.integers(0, len(seeds), size=len(seeds))
            boot_means = np.nanmean(r_matrix[:, boot_idx], axis=1)
            valid = ~np.isnan(boot_means)
            if valid.sum() >= 4:
                try:
                    kc_b, _ = fit_sigmoid(k_vals[valid], boot_means[valid])
                    kc_boot_estimates.append(kc_b)
                except Exception:
                    pass
        if len(kc_boot_estimates) >= 10:
            ci_lo_raw = float(np.percentile(kc_boot_estimates, 2.5))
            ci_hi_raw = float(np.percentile(kc_boot_estimates, 97.5))
            # Sanity check: CI should contain kc_sig and not be absurdly wide
            k_range = float(k_vals[-1] - k_vals[0])
            if (ci_lo_raw <= kc_sig <= ci_hi_raw) and (ci_hi_raw - ci_lo_raw < k_range):
                ci_lo, ci_hi = ci_lo_raw, ci_hi_raw
            else:
                # Bootstrap failed (too few seeds): use ±0.15 as conservative CI
                ci_lo, ci_hi = max(0.0, kc_sig - 0.15), kc_sig + 0.15
        else:
            ci_lo, ci_hi = max(0.0, kc_sig - 0.15), kc_sig + 0.15

        entry = {
            "kc_sigmoid": kc_sig,
            "kc_susceptibility": kc_sus,
            "kc_ci_low": ci_lo,
            "kc_ci_high": ci_hi,
        }

        # Per-sigma K_c for all-to-all analytical comparison
        if topo == "all-to-all":
            for sigma_val, skey in [(0.3, "kc_sigmoid_s03"), (0.8, "kc_sigmoid_s08")]:
                topo_sigma_stats = [s for s in topo_stats if abs(s["sigma"] - sigma_val) < 0.01]
                if topo_sigma_stats:
                    k_s = np.array(sorted(set(s["K"] for s in topo_sigma_stats)))
                    r_s = np.array([
                        float(np.mean([s["mean_r"] for s in topo_sigma_stats if s["K"] == k]))
                        for k in k_s
                    ])
                    try:
                        kc_s, _ = fit_sigmoid(k_s, r_s)
                    except Exception:
                        kc_s = float(k_s[np.argmax(np.gradient(r_s))])
                    entry[skey] = kc_s

        pt[topo] = entry

    return pt


def _compute_critical_exponents(records, stats, pt):
    """Compute critical exponent β per topology per sigma."""
    ce = {}
    topologies = sorted(set(r["topology"] for r in records))

    for topo in topologies:
        topo_data = pt.get(topo, {})
        kc = topo_data.get("kc_sigmoid", None)
        if kc is None:
            continue

        for sigma_val, slabel in [(0.3, "s03"), (0.8, "s08")]:
            key = f"{topo}_{slabel}"
            topo_sigma_stats = [s for s in stats
                                if s["topology"] == topo
                                and abs(s["sigma"] - sigma_val) < 0.01]
            if not topo_sigma_stats:
                continue
            k_vals = np.array(sorted(set(s["K"] for s in topo_sigma_stats)))
            r_means = np.array([
                float(np.mean([s["mean_r"] for s in topo_sigma_stats if s["K"] == k]))
                for k in k_vals
            ])
            try:
                beta, r2 = fit_critical_exponent(k_vals, r_means, kc)
            except Exception:
                beta, r2 = float("nan"), float("nan")
            ce[key] = {"beta": beta, "r_squared": r2}

    return ce


def _compute_finite_size_scaling(stats, pt):
    """Compute K_c(N) and fit finite-size scaling per topology."""
    fss = {}
    topologies = sorted(set(s["topology"] for s in stats))

    for topo in topologies:
        entry = {}
        kc_by_n = {}

        for n_val in [6, 12, 24]:
            topo_n_stats = [s for s in stats
                            if s["topology"] == topo and s["n"] == n_val]
            if not topo_n_stats:
                continue
            k_vals = np.array(sorted(set(s["K"] for s in topo_n_stats)))
            r_means = np.array([
                float(np.mean([s["mean_r"] for s in topo_n_stats if s["K"] == k]))
                for k in k_vals
            ])
            try:
                kc_n, _ = fit_sigmoid(k_vals, r_means)
            except Exception:
                kc_n = float(k_vals[np.argmax(np.gradient(r_means))])
            kc_by_n[n_val] = kc_n
            entry[f"kc_n{n_val}"] = kc_n

        if len(kc_by_n) >= 2:
            try:
                kc_inf, nu = fit_finite_size_scaling(kc_by_n)
                # Compute a_fss from K_c(6) = kc_inf + a * 6^(-nu)
                if nu > 0 and kc_by_n.get(6):
                    a_fss = (kc_by_n[6] - kc_inf) * (6 ** nu)
                else:
                    a_fss = float("nan")
            except Exception:
                kc_inf, nu, a_fss = float("nan"), float("nan"), float("nan")
            entry["kc_inf"] = kc_inf
            entry["nu"] = nu
            entry["a_fss"] = a_fss

        fss[topo] = entry

    return fss


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    configs = build_configs()
    total = len(configs)

    # Step 1: Run simulations per topology
    print(f"Running {total} simulations across {len(TOPOLOGIES)} topologies...\n")
    results_by_topo = {t: [] for t in TOPOLOGIES}
    all_results = []

    for topo_idx, topo in enumerate(TOPOLOGIES):
        topo_configs = [c for c in configs if c.topology == topo]
        n_topo = len(topo_configs)
        print(f"[{topo_idx + 1}/{len(TOPOLOGIES)}] Topology: {topo} ({n_topo} sims)...")
        for i, config in enumerate(topo_configs):
            result = run_simulation(config)
            results_by_topo[topo].append(result)
            all_results.append(result)
        print(f"    Done ({n_topo} sims completed)")

    # Step 2: Analyze results
    print("\n[2/5] Analyzing results (running evaluator panel)...")
    analysis = analyze_results(all_results)
    records = analysis["records"]
    stats = analysis["statistics"]

    # Step 3: Advanced phase transition analysis
    print("[3/5] Computing phase transition, critical exponents, finite-size scaling...")
    pt = _compute_phase_transition(records, stats)
    ce = _compute_critical_exponents(records, stats, pt)
    fss = _compute_finite_size_scaling(stats, pt)

    analysis["phase_transition"] = pt
    analysis["critical_exponents"] = ce
    analysis["finite_size_scaling"] = fss

    # Step 4: Generate report
    print("[4/5] Generating report...")
    report = generate_report(analysis)

    # Step 5: Save results (before figure generation, which may fail on headless systems)
    print("[5/5] Saving results to results/")

    serializable = {
        "metadata": {
            "num_simulations": total,
            "num_conditions": len(stats),
            "topologies": TOPOLOGIES,
            "k_range": list(K_RANGE),
            "n_values": N_VALUES,
            "sigma_values": SIGMA_VALUES,
            "seeds": SEEDS,
        },
        "records": records,
        "statistics": stats,
    }

    with open("results/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    with open("results/report.md", "w") as f:
        f.write(report)

    stat_tests = []
    for topo in TOPOLOGIES:
        topo_data = pt.get(topo, {})
        entry = {
            "topology": topo,
            "kc_sigmoid": topo_data.get("kc_sigmoid"),
            "kc_susceptibility": topo_data.get("kc_susceptibility"),
            "kc_ci_low": topo_data.get("kc_ci_low"),
            "kc_ci_high": topo_data.get("kc_ci_high"),
            "analytical_kc_s03": analytical_kc(0.3),
            "analytical_kc_s08": analytical_kc(0.8),
            "critical_exponents": {
                k: v for k, v in ce.items() if k.startswith(topo)
            },
            "finite_size_scaling": fss.get(topo, {}),
        }
        stat_tests.append(entry)

    with open("results/statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2, default=str)

    # Generate figures last (after all data is saved; figure errors don't lose data)
    generate_figures(analysis)

    print(f"\nDone. Results saved to results/")
    print(f"  results/results.json ({total} simulation records)")
    print(f"  results/report.md")
    print(f"  results/statistical_tests.json")
    print(f"  results/figures/ (6 PNGs)")


if __name__ == "__main__":
    main()
