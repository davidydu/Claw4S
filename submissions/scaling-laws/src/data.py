"""Hardcoded published results from Cerebras-GPT and Pythia model families.

All values are sourced from published papers and HuggingFace model cards.
No network calls or downloads — data is embedded for full reproducibility.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cerebras-GPT (Dey et al., 2023)
# Paper: arXiv 2304.03208
# Source: HuggingFace model cards at huggingface.co/cerebras/Cerebras-GPT-{size}
# Training: The Pile, Chinchilla-optimal (D ≈ 20N)
# All benchmarks: 0-shot
# ---------------------------------------------------------------------------
CEREBRAS_GPT: dict = {
    "name": "Cerebras-GPT",
    "source": "Dey et al., 2023 (arXiv 2304.03208); HuggingFace model cards",
    "dataset": "The Pile",
    "training_recipe": "Chinchilla-optimal (D ≈ 20N)",
    "models": {
        "111M": {
            "params": 111e6,
            "training_tokens": 2.2e9,
            "pile_test_loss": 2.566,
            "lambada_acc": 0.194,
            "hellaswag_acc": 0.268,
            "piqa_acc": 0.594,
            "winogrande_acc": 0.488,
            "arc_easy_acc": 0.380,
            "arc_challenge_acc": 0.166,
            "openbookqa_acc": 0.118,
        },
        "256M": {
            "params": 256e6,
            "training_tokens": 5.1e9,
            "pile_test_loss": 2.299,
            "lambada_acc": 0.293,
            "hellaswag_acc": 0.274,
            "piqa_acc": 0.613,
            "winogrande_acc": 0.511,
            "arc_easy_acc": 0.410,
            "arc_challenge_acc": 0.170,
            "openbookqa_acc": 0.158,
        },
        "590M": {
            "params": 590e6,
            "training_tokens": 11.8e9,
            "pile_test_loss": 2.184,
            "lambada_acc": 0.366,
            "hellaswag_acc": 0.291,
            "piqa_acc": 0.627,
            "winogrande_acc": 0.498,
            "arc_easy_acc": 0.464,
            "arc_challenge_acc": 0.190,
            "openbookqa_acc": 0.158,
        },
        "1.3B": {
            "params": 1.3e9,
            "training_tokens": 26.3e9,
            "pile_test_loss": 1.996,
            "lambada_acc": 0.462,
            "hellaswag_acc": 0.325,
            "piqa_acc": 0.664,
            "winogrande_acc": 0.521,
            "arc_easy_acc": 0.508,
            "arc_challenge_acc": 0.224,
            "openbookqa_acc": 0.166,
        },
        "2.7B": {
            "params": 2.7e9,
            "training_tokens": 53.0e9,
            "pile_test_loss": 1.834,
            "lambada_acc": 0.567,
            "hellaswag_acc": 0.386,
            "piqa_acc": 0.701,
            "winogrande_acc": 0.559,
            "arc_easy_acc": 0.571,
            "arc_challenge_acc": 0.246,
            "openbookqa_acc": 0.206,
        },
        "6.7B": {
            "params": 6.7e9,
            "training_tokens": 133e9,
            "pile_test_loss": 1.704,
            "lambada_acc": 0.636,
            "hellaswag_acc": 0.447,
            "piqa_acc": 0.739,
            "winogrande_acc": 0.602,
            "arc_easy_acc": 0.643,
            "arc_challenge_acc": 0.282,
            "openbookqa_acc": 0.238,
        },
        "13B": {
            "params": 13e9,
            "training_tokens": 257e9,
            "pile_test_loss": 1.575,
            "lambada_acc": 0.696,
            "hellaswag_acc": 0.513,
            "piqa_acc": 0.766,
            "winogrande_acc": 0.646,
            "arc_easy_acc": 0.714,
            "arc_challenge_acc": 0.367,
            "openbookqa_acc": 0.286,
        },
    },
}


def get_family_data(
    family: dict, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (params, metric_values) arrays from a model family dict."""
    params = []
    values = []
    for model in family["models"].values():
        if metric in model:
            params.append(model["params"])
            values.append(model[metric])
    return np.array(params), np.array(values)


def get_training_tokens(family: dict) -> np.ndarray:
    """Extract training token counts as array (for Chinchilla formulation)."""
    tokens = []
    for model in family["models"].values():
        if "training_tokens" in model:
            tokens.append(model["training_tokens"])
        elif "training_tokens_per_model" in family:
            tokens.append(family["training_tokens_per_model"])
    return np.array(tokens)


def get_benchmark_keys(family: dict) -> list[str]:
    """Return list of benchmark keys present in a model family."""
    first_model = next(iter(family["models"].values()))
    exclude = {"params", "training_tokens", "pile_test_loss", "non_emb_params"}
    return [k for k in first_model if k not in exclude]
