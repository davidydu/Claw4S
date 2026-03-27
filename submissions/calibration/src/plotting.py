"""Plotting functions for calibration analysis.

Generates reliability diagrams, ECE vs shift plots, and confidence histograms.
All plots are saved as PDF for inclusion in the LaTeX research note.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Any


def plot_ece_vs_shift(aggregated: list[dict], output_dir: str) -> str:
    """Plot ECE vs distribution shift magnitude for each model width.

    This is the main result figure: shows how calibration degrades under
    shift for models of different capacity.

    Args:
        aggregated: List of aggregated result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    widths = sorted(set(r['hidden_width'] for r in aggregated))
    shifts = sorted(set(r['shift_magnitude'] for r in aggregated))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))

    for i, width in enumerate(widths):
        eces = []
        stds = []
        for shift in shifts:
            match = [r for r in aggregated
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                eces.append(match[0]['ece_mean'])
                stds.append(match[0]['ece_std'])
            else:
                eces.append(np.nan)
                stds.append(0.0)

        ax.errorbar(shifts, eces, yerr=stds, marker='o', linewidth=2,
                   capsize=4, label=f'width={width}', color=colors[i])

    ax.set_xlabel('Distribution Shift Magnitude', fontsize=12)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax.set_title('Calibration Degradation Under Distribution Shift', fontsize=13)
    ax.legend(title='Hidden Width', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, 'ece_vs_shift.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_accuracy_vs_shift(aggregated: list[dict], output_dir: str) -> str:
    """Plot accuracy vs distribution shift for each model width.

    Args:
        aggregated: List of aggregated result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    widths = sorted(set(r['hidden_width'] for r in aggregated))
    shifts = sorted(set(r['shift_magnitude'] for r in aggregated))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))

    for i, width in enumerate(widths):
        accs = []
        stds = []
        for shift in shifts:
            match = [r for r in aggregated
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                accs.append(match[0]['accuracy_mean'])
                stds.append(match[0]['accuracy_std'])
            else:
                accs.append(np.nan)
                stds.append(0.0)

        ax.errorbar(shifts, accs, yerr=stds, marker='s', linewidth=2,
                   capsize=4, label=f'width={width}', color=colors[i])

    ax.set_xlabel('Distribution Shift Magnitude', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Under Distribution Shift', fontsize=13)
    ax.legend(title='Hidden Width', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    path = os.path.join(output_dir, 'accuracy_vs_shift.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_brier_vs_shift(aggregated: list[dict], output_dir: str) -> str:
    """Plot Brier score vs distribution shift for each model width.

    Args:
        aggregated: List of aggregated result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    widths = sorted(set(r['hidden_width'] for r in aggregated))
    shifts = sorted(set(r['shift_magnitude'] for r in aggregated))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))

    for i, width in enumerate(widths):
        briers = []
        stds = []
        for shift in shifts:
            match = [r for r in aggregated
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                briers.append(match[0]['brier_mean'])
                stds.append(match[0]['brier_std'])
            else:
                briers.append(np.nan)
                stds.append(0.0)

        ax.errorbar(shifts, briers, yerr=stds, marker='^', linewidth=2,
                   capsize=4, label=f'width={width}', color=colors[i])

    ax.set_xlabel('Distribution Shift Magnitude', fontsize=12)
    ax.set_ylabel('Brier Score', fontsize=12)
    ax.set_title('Brier Score Under Distribution Shift', fontsize=13)
    ax.legend(title='Hidden Width', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, 'brier_vs_shift.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_reliability_diagram(raw_results: list[dict],
                             output_dir: str,
                             target_width: int = 256,
                             target_seed: int = 42) -> str:
    """Plot reliability diagrams for one model across shifts.

    Shows how the calibration curve departs from the diagonal under shift.

    Args:
        raw_results: List of per-experiment result dicts.
        output_dir: Directory to save the plot.
        target_width: Which model width to plot.
        target_seed: Which seed to use.

    Returns:
        Path to saved figure.
    """
    # Find the matching experiment
    match = [r for r in raw_results
             if r['hidden_width'] == target_width and r['seed'] == target_seed]
    if not match:
        raise ValueError(f"No results for width={target_width}, seed={target_seed}")

    result = match[0]
    shifts = sorted(result['shifts'].keys(), key=float)

    fig, axes = plt.subplots(1, len(shifts), figsize=(3.5 * len(shifts), 3.5),
                             sharey=True)
    if len(shifts) == 1:
        axes = [axes]

    for ax, shift_key in zip(axes, shifts):
        shift_data = result['shifts'][shift_key]
        rel = shift_data['reliability']
        bin_confs = rel['bin_confs']
        bin_accs = rel['bin_accs']
        bin_counts = rel['bin_counts']

        # Bar chart of reliability
        bin_centers = [(rel['bin_edges'][i] + rel['bin_edges'][i+1]) / 2
                      for i in range(len(bin_confs))]
        bar_width = 1.0 / len(bin_confs)

        # Only plot bins with data
        for j in range(len(bin_confs)):
            if bin_counts[j] > 0:
                gap = bin_accs[j] - bin_confs[j]
                color = '#2196F3' if gap >= 0 else '#F44336'
                ax.bar(bin_centers[j], bin_accs[j], width=bar_width * 0.8,
                      color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
               label='Perfect calibration')
        ax.set_xlabel('Confidence', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'Shift = {shift_key}', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ece = shift_data['ece']
        ax.text(0.05, 0.92, f'ECE={ece:.3f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Reliability Diagrams (width={target_width})', fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, 'reliability_diagrams.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_overconfidence_gap(aggregated: list[dict], output_dir: str) -> str:
    """Plot the confidence-accuracy gap vs shift for each width.

    Overconfidence = mean_confidence - accuracy. Shows how models become
    increasingly overconfident under shift.

    Args:
        aggregated: List of aggregated result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    widths = sorted(set(r['hidden_width'] for r in aggregated))
    shifts = sorted(set(r['shift_magnitude'] for r in aggregated))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))

    for i, width in enumerate(widths):
        gaps = []
        for shift in shifts:
            match = [r for r in aggregated
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                gap = match[0]['confidence_mean'] - match[0]['accuracy_mean']
                gaps.append(gap)
            else:
                gaps.append(np.nan)

        ax.plot(shifts, gaps, marker='D', linewidth=2,
               label=f'width={width}', color=colors[i])

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3,
              label='Perfect calibration')
    ax.set_xlabel('Distribution Shift Magnitude', fontsize=12)
    ax.set_ylabel('Overconfidence Gap (Confidence - Accuracy)', fontsize=12)
    ax.set_title('Overconfidence Under Distribution Shift', fontsize=13)
    ax.legend(title='Hidden Width', fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'overconfidence_gap.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_all_plots(results: dict, output_dir: str) -> list[str]:
    """Generate all analysis plots.

    Args:
        results: Full results dict from run_all_experiments().
        output_dir: Directory to save plots.

    Returns:
        List of paths to generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    print("[plots] Generating ECE vs shift plot...")
    paths.append(plot_ece_vs_shift(results['aggregated'], output_dir))

    print("[plots] Generating accuracy vs shift plot...")
    paths.append(plot_accuracy_vs_shift(results['aggregated'], output_dir))

    print("[plots] Generating Brier score vs shift plot...")
    paths.append(plot_brier_vs_shift(results['aggregated'], output_dir))

    print("[plots] Generating reliability diagrams...")
    paths.append(plot_reliability_diagram(results['raw_results'], output_dir))

    print("[plots] Generating overconfidence gap plot...")
    paths.append(plot_overconfidence_gap(results['aggregated'], output_dir))

    return paths
