"""Run the full RMT analysis pipeline.

Trains tiny MLPs on modular arithmetic and regression, then analyzes
weight matrix eigenvalue spectra against the Marchenko-Pastur distribution.
"""

import json
import os
import sys

import numpy as np
import torch

# Working-directory guard: must be run from the submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from the submissions/rmt/ directory.")
    print("  cd submissions/rmt/ && .venv/bin/python run.py")
    sys.exit(1)

from src.data import generate_modular_addition, generate_regression
from src.model import TinyMLP
from src.train import train_model
from src.rmt_analysis import analyze_model_weights
from src.plots import plot_eigenvalue_spectra, plot_ks_summary
from src.report import generate_report, save_report

SEED = 42
HIDDEN_DIMS = [32, 64, 128, 256]
MOD_EPOCHS = 500
REG_EPOCHS = 500
LR = 1e-3


def main() -> None:
    print("=" * 60)
    print("Random Matrix Theory Analysis of Neural Network Weights")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Step 1: Generate datasets
    print("\n[1/5] Generating datasets...")
    mod_data = generate_modular_addition(p=97, seed=SEED)
    reg_data = generate_regression(n_samples=1000, seed=SEED)
    print(f"  Modular addition: {mod_data['n_train']} train, {mod_data['n_test']} test")
    print(f"  Regression: {reg_data['n_train']} train, {reg_data['n_test']} test")

    # Step 2: Train models
    print("\n[2/5] Training models...")
    training_results = []
    trained_models = {}
    untrained_models = {}

    for hidden_dim in HIDDEN_DIMS:
        for task_name, data, epochs in [
            ("mod97", mod_data, MOD_EPOCHS),
            ("regression", reg_data, REG_EPOCHS),
        ]:
            model_label = f"{task_name}_h{hidden_dim}"
            print(f"\n  Training {model_label}...")

            # Save untrained model for baseline comparison
            torch.manual_seed(SEED)
            untrained = TinyMLP(data["input_dim"], hidden_dim, data["output_dim"])
            untrained_models[model_label] = untrained

            # Train a fresh copy
            torch.manual_seed(SEED)
            model = TinyMLP(data["input_dim"], hidden_dim, data["output_dim"])
            history = train_model(
                model=model,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                task=data["task"],
                epochs=epochs,
                lr=LR,
                seed=SEED,
                verbose=True,
            )

            trained_models[model_label] = model
            training_results.append({
                "model_label": model_label,
                "task": task_name,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "final_loss": history["final_loss"],
                **{k: v for k, v in history.items()
                   if k in ("final_accuracy", "final_mse")},
            })

    # Step 3: Analyze weight matrices
    print("\n[3/5] Analyzing weight matrices (RMT)...")
    trained_analysis = []
    untrained_analysis = []

    for model_label in sorted(trained_models.keys()):
        print(f"  Analyzing {model_label}...")

        # Trained weights
        trained_weights = [
            (name, W.numpy())
            for name, W in trained_models[model_label].get_weight_matrices()
        ]
        trained_results = analyze_model_weights(trained_weights, model_label)
        trained_analysis.extend(trained_results)

        # Untrained baseline
        untrained_weights = [
            (name, W.numpy())
            for name, W in untrained_models[model_label].get_weight_matrices()
        ]
        untrained_results = analyze_model_weights(
            untrained_weights, model_label
        )
        untrained_analysis.extend(untrained_results)

        # Print summary
        for tr, ut in zip(trained_results, untrained_results):
            print(
                f"    {tr['layer_name']}: "
                f"KS trained={tr['ks_statistic']:.4f} vs "
                f"untrained={ut['ks_statistic']:.4f}, "
                f"outliers={tr['outlier_fraction']:.2f}"
            )

    # Step 4: Generate plots
    print("\n[4/5] Generating plots...")
    os.makedirs("results", exist_ok=True)

    spectra_path = plot_eigenvalue_spectra(
        trained_analysis, save_path="results/eigenvalue_spectra.png"
    )
    print(f"  Saved {spectra_path}")

    ks_path = plot_ks_summary(
        trained_analysis, save_path="results/ks_summary.png"
    )
    print(f"  Saved {ks_path}")

    # Also plot untrained for comparison
    untrained_spectra_path = plot_eigenvalue_spectra(
        untrained_analysis, save_path="results/eigenvalue_spectra_untrained.png"
    )
    print(f"  Saved {untrained_spectra_path}")

    # Step 5: Save results and report
    print("\n[5/5] Saving results and report...")

    # Strip eigenvalue arrays from JSON to keep file size reasonable
    def strip_eigenvalues(analysis_list: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in r.items() if k != "eigenvalues"}
            for r in analysis_list
        ]

    results_data = {
        "metadata": {
            "seed": SEED,
            "hidden_dims": HIDDEN_DIMS,
            "mod_epochs": MOD_EPOCHS,
            "reg_epochs": REG_EPOCHS,
            "learning_rate": LR,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "training_results": training_results,
        "trained_analysis": strip_eigenvalues(trained_analysis),
        "untrained_analysis": strip_eigenvalues(untrained_analysis),
    }

    results_path = os.path.join("results", "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Saved {results_path}")

    report = generate_report(results_data)
    report_path = save_report(report, "results/report.md")
    print(f"  Saved {report_path}")

    print("\n" + "=" * 60)
    print("Analysis complete. See results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
