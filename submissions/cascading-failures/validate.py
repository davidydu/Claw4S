"""Validate experiment results for completeness and correctness.

Checks:
  1. results.json exists and has expected structure
  2. Correct number of simulations (324 full, 18 diagnostic)
  3. All metrics are present and within valid ranges
  4. All topologies, agent types, and shock conditions are represented
  5. Key scientific predictions hold (sanity checks)
"""

import json
import math
import sys
from collections import Counter


def main() -> None:
    try:
        with open("results/results.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("ERROR: results/results.json not found. Run run.py first.")
        sys.exit(1)

    errors = []
    warnings = []

    # --- Structure checks ---
    meta = data.get("metadata", {})
    n_sims = meta.get("n_simulations", 0)
    is_diag = meta.get("diagnostic", False)
    expected_sims = 18 if is_diag else 324

    print(f"Mode: {'diagnostic' if is_diag else 'full'}")
    print(f"Simulations: {n_sims} (expected {expected_sims})")

    if n_sims != expected_sims:
        errors.append(f"Expected {expected_sims} simulations, got {n_sims}")

    raw = data.get("raw_results", [])
    agg = data.get("aggregated", [])
    n_agg = meta.get("n_conditions", 0)

    if len(raw) != n_sims:
        errors.append(f"metadata.n_simulations={n_sims} but raw_results has {len(raw)} rows")
    if len(agg) != n_agg:
        errors.append(f"metadata.n_conditions={n_agg} but aggregated has {len(agg)} rows")

    # --- Metric range checks ---
    for r in raw:
        cs = r.get("cascade_size", -1)
        if not (0.0 <= cs <= 1.0):
            errors.append(f"cascade_size out of [0,1]: {cs} in {r.get('topology')}/{r.get('agent_type')}")

        sr = r.get("systemic_risk", -1)
        if math.isfinite(sr) and sr < 0:
            errors.append(f"Negative systemic_risk: {sr}")

        speed = r.get("cascade_speed")
        if speed is not None and math.isfinite(speed) and speed < 0:
            errors.append(f"Negative cascade_speed: {speed}")

        loc = r.get("shock_location")
        is_hub = r.get("shock_node_is_hub")
        has_non_hub = r.get("has_non_hub_nodes")
        if loc not in {"hub", "random"}:
            errors.append(f"Invalid shock_location: {loc}")
        if not isinstance(is_hub, bool):
            errors.append("Missing/invalid shock_node_is_hub flag in raw_results row")
        if not isinstance(has_non_hub, bool):
            errors.append("Missing/invalid has_non_hub_nodes flag in raw_results row")
        if loc == "hub" and is_hub is False:
            errors.append("Hub condition used a non-hub shock node")
        if loc == "random" and has_non_hub is True and is_hub is True:
            errors.append("Random condition selected a hub despite non-hub nodes existing")

    # Every condition should include exactly 3 seed replicates.
    condition_counts = Counter(
        (
            r.get("topology"),
            r.get("agent_type"),
            r.get("shock_magnitude"),
            r.get("shock_location"),
        )
        for r in raw
    )
    for key, count in sorted(condition_counts.items()):
        if count != 3:
            errors.append(f"Condition {key} has {count} replicates (expected 3)")

    # --- Coverage checks (full experiment only) ---
    if not is_diag:
        topos = set(r["topology"] for r in raw)
        agents = set(r["agent_type"] for r in raw)
        mags = set(r["shock_magnitude"] for r in raw)
        locs = set(r["shock_location"] for r in raw)

        expected_topos = {"chain", "ring", "star", "erdos_renyi", "scale_free", "fully_connected"}
        expected_agents = {"robust", "fragile", "averaging"}
        expected_mags = {2.0, 10.0, 50.0}
        expected_locs = {"hub", "random"}

        if topos != expected_topos:
            errors.append(f"Missing topologies: {expected_topos - topos}")
        if agents != expected_agents:
            errors.append(f"Missing agent types: {expected_agents - agents}")
        if mags != expected_mags:
            errors.append(f"Missing shock magnitudes: {expected_mags - mags}")
        if locs != expected_locs:
            errors.append(f"Missing shock locations: {expected_locs - locs}")

        # --- Scientific sanity checks ---
        # Averaging agents should have lower mean cascade than fragile
        avg_cascade = [r["cascade_size"] for r in raw if r["agent_type"] == "averaging"]
        frag_cascade = [r["cascade_size"] for r in raw if r["agent_type"] == "fragile"]
        if avg_cascade and frag_cascade:
            avg_mean = sum(avg_cascade) / len(avg_cascade)
            frag_mean = sum(frag_cascade) / len(frag_cascade)
            print(f"Averaging agent mean cascade: {avg_mean:.3f}")
            print(f"Fragile agent mean cascade:   {frag_mean:.3f}")
            if avg_mean > frag_mean:
                warnings.append("Averaging agents have HIGHER cascade than fragile (unexpected)")

        # Severe shock should cause larger cascades than mild
        mild = [r["cascade_size"] for r in raw if r["shock_magnitude"] == 2.0]
        severe = [r["cascade_size"] for r in raw if r["shock_magnitude"] == 50.0]
        if mild and severe:
            mild_mean = sum(mild) / len(mild)
            severe_mean = sum(severe) / len(severe)
            print(f"Mild shock mean cascade:   {mild_mean:.3f}")
            print(f"Severe shock mean cascade: {severe_mean:.3f}")
            if severe_mean < mild_mean:
                warnings.append("Severe shocks cause SMALLER cascades than mild (unexpected)")

    # --- Report ---
    print(f"Aggregated conditions: {n_agg}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nValidation passed.")


if __name__ == "__main__":
    main()
