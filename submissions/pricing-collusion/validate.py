# validate.py
"""Validate experiment results for completeness and correctness."""

import json
import sys


def main():
    try:
        with open("results/results.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: results/results.json not found. Run `run.py` first.")
        sys.exit(1)

    num_sims = data["metadata"]["num_simulations"]
    num_conditions = data["metadata"]["num_conditions"]
    num_records = len(data["records"])

    print(f"Simulations: {num_sims}")
    print(f"Conditions:  {num_conditions}")
    print(f"Records:     {num_records} (expected {num_sims})")

    errors = []

    # Check record count
    if num_records != num_sims:
        errors.append(f"Expected {num_sims} records, got {num_records}")

    # Check all matchups present
    expected_matchups = set(data["metadata"]["matchups"])
    found_matchups = set(r["matchup"] for r in data["records"])
    if found_matchups != expected_matchups:
        errors.append(f"Missing matchups: {expected_matchups - found_matchups}")

    # Check all presets present
    expected_presets = set(data["metadata"]["presets"])
    found_presets = set(r["preset"] for r in data["records"])
    if found_presets != expected_presets:
        errors.append(f"Missing presets: {expected_presets - found_presets}")

    # Check auditor scores present and valid (4 auditors expected)
    for r in data["records"]:
        scores = r.get("auditor_scores", {})
        if len(scores) < 4:
            errors.append(f"Record {r['matchup']}/M{r['memory']}/{r['preset']}/seed{r['seed']}"
                          f" has {len(scores)} auditor scores (expected 4: margin, "
                          f"deviation_punishment, counterfactual, welfare)")
            break
        for name, score in scores.items():
            if not (0.0 <= score <= 1.0):
                errors.append(f"Auditor {name} score {score} out of [0, 1] range")
                break

    # Check Nash < monopoly for no-shock conditions only
    # (cost shocks can push Nash above price_max in tight markets, making Nash >= monopoly)
    for s in data["statistics"]:
        if not s["shocks"] and s["nash_price"] >= s["monopoly_price"]:
            errors.append(f"Nash >= monopoly for {s['matchup']}/{s['preset']}")

    # Check competitive control has low collusion
    competitive_records = [r for r in data["records"]
                           if r["matchup"] == "Q-Competitive" and not r["shocks"]]
    if competitive_records:
        avg_margin = sum(r["auditor_scores"].get("margin", 0)
                         for r in competitive_records) / len(competitive_records)
        print(f"\nCompetitive control avg margin score: {avg_margin:.3f}")
        if avg_margin > 0.5:
            errors.append(f"Competitive control has high collusion score ({avg_margin:.3f})"
                          " — possible calibration issue")

    # Summary stats (using Bonferroni-corrected p-values)
    print(f"\nStatistical conditions: {num_conditions}")
    sig_conditions = [s for s in data["statistics"]
                      if s.get("p_value_corrected", s["p_value"]) < 0.05
                      and s.get("collusion_index", 0) > 0]
    print(f"Conditions with significant supra-Nash pricing (Bonferroni): "
          f"{len(sig_conditions)}/{num_conditions}")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nValidation passed.")


if __name__ == "__main__":
    main()
