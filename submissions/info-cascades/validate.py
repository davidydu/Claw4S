"""Validate information cascade experiment results.

Checks:
1. Expected number of simulations (216).
2. All 4 agent types present.
3. All 3 signal qualities present.
4. All 3 sequence lengths present.
5. Cascade formation rates are in [0, 1].
6. Cascade accuracy (where defined) is in [0, 1].
7. Symmetry check: results for true_state=0 and true_state=1 are similar.
8. Metrics file has expected number of groups (36).
"""

import json
import sys

EXPECTED_SIMS = 216
EXPECTED_GROUPS = 36  # 4 types x 3 qualities x 3 lengths
EXPECTED_AGENT_TYPES = {"bayesian", "heuristic", "contrarian", "noisy_bayesian"}
EXPECTED_SIGNAL_QUALITIES = {0.6, 0.7, 0.9}
EXPECTED_SEQUENCE_LENGTHS = {10, 20, 50}


def main() -> None:
    errors: list[str] = []

    # Load raw results
    with open("results/raw_results.json") as f:
        raw = json.load(f)

    # Load metrics
    with open("results/metrics.json") as f:
        metrics = json.load(f)

    # Load metadata
    with open("results/metadata.json") as f:
        metadata = json.load(f)

    # Check 1: simulation count
    n_sims = len(raw)
    print(f"Simulations: {n_sims} (expected {EXPECTED_SIMS})")
    if n_sims != EXPECTED_SIMS:
        errors.append(f"Expected {EXPECTED_SIMS} simulations, got {n_sims}")

    # Check 2: agent types
    agent_types = set(r["agent_type"] for r in raw)
    print(f"Agent types: {sorted(agent_types)}")
    if agent_types != EXPECTED_AGENT_TYPES:
        errors.append(f"Expected agent types {EXPECTED_AGENT_TYPES}, got {agent_types}")

    # Check 3: signal qualities
    signal_qualities = set(r["signal_quality"] for r in raw)
    print(f"Signal qualities: {sorted(signal_qualities)}")
    if signal_qualities != EXPECTED_SIGNAL_QUALITIES:
        errors.append(f"Expected signal qualities {EXPECTED_SIGNAL_QUALITIES}, got {signal_qualities}")

    # Check 4: sequence lengths
    seq_lengths = set(r["n_agents"] for r in raw)
    print(f"Sequence lengths: {sorted(seq_lengths)}")
    if seq_lengths != EXPECTED_SEQUENCE_LENGTHS:
        errors.append(f"Expected sequence lengths {EXPECTED_SEQUENCE_LENGTHS}, got {seq_lengths}")

    # Check 5: metric groups
    n_groups = len(metrics)
    print(f"Metric groups: {n_groups} (expected {EXPECTED_GROUPS})")
    if n_groups != EXPECTED_GROUPS:
        errors.append(f"Expected {EXPECTED_GROUPS} metric groups, got {n_groups}")

    # Check 6: cascade formation rates in [0, 1]
    for m in metrics:
        rate = m["cascade_formation_rate"]
        if not (0.0 <= rate <= 1.0):
            errors.append(
                f"Formation rate {rate} out of [0,1] for "
                f"{m['agent_type']}/q={m['signal_quality']}/N={m['n_agents']}"
            )

    # Check 7: cascade accuracy in [0, 1] where defined
    for m in metrics:
        acc = m["cascade_accuracy"]
        if acc is not None and not (0.0 <= acc <= 1.0):
            errors.append(
                f"Cascade accuracy {acc} out of [0,1] for "
                f"{m['agent_type']}/q={m['signal_quality']}/N={m['n_agents']}"
            )

    # Check 8: symmetry — formation rates for state=0 vs state=1 should be similar
    by_state: dict[int, list[float]] = {0: [], 1: []}
    for r in raw:
        by_state[r["true_state"]].append(1.0 if r["cascade_formed"] else 0.0)
    rate_0 = sum(by_state[0]) / len(by_state[0]) if by_state[0] else 0
    rate_1 = sum(by_state[1]) / len(by_state[1]) if by_state[1] else 0
    print(f"Symmetry check: formation rate state=A: {rate_0:.3f}, state=B: {rate_1:.3f}")
    sym_diff = abs(rate_0 - rate_1)
    if sym_diff > 0.15:
        errors.append(
            f"Symmetry violation: formation rates differ by {sym_diff:.3f} "
            f"(state=A: {rate_0:.3f}, state=B: {rate_1:.3f})"
        )

    # Check 9: Bayesian at q=0.9 should have higher accuracy than at q=0.6
    bayes_q06 = [m for m in metrics if m["agent_type"] == "bayesian" and m["signal_quality"] == 0.6]
    bayes_q09 = [m for m in metrics if m["agent_type"] == "bayesian" and m["signal_quality"] == 0.9]
    if bayes_q06 and bayes_q09:
        accs_06 = [m["cascade_accuracy"] for m in bayes_q06 if m["cascade_accuracy"] is not None]
        accs_09 = [m["cascade_accuracy"] for m in bayes_q09 if m["cascade_accuracy"] is not None]
        if accs_06 and accs_09:
            avg_06 = sum(accs_06) / len(accs_06)
            avg_09 = sum(accs_09) / len(accs_09)
            print(f"Bayesian accuracy: q=0.6: {avg_06:.3f}, q=0.9: {avg_09:.3f}")
            if avg_09 < avg_06:
                errors.append(
                    f"Expected Bayesian accuracy higher at q=0.9 ({avg_09:.3f}) "
                    f"than q=0.6 ({avg_06:.3f})"
                )

    # Summary
    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nValidation passed.")


if __name__ == "__main__":
    main()
