"""Hardcoded benchmark scores for 30+ LLMs from published sources.

All scores are 0-shot accuracy (%) unless noted otherwise. Sources:
- Open LLM Leaderboard v1 (HuggingFace, Apr 2023 - Jun 2024)
  ARC-Challenge 25-shot, HellaSwag 10-shot, MMLU 5-shot, WinoGrande 5-shot,
  TruthfulQA 0-shot, GSM8K 5-shot
- Llama 2 paper (Touvron et al., 2023, arXiv:2307.09288)
- Mistral 7B paper (Jiang et al., 2023, arXiv:2310.06825)
- Pythia paper (Biderman et al., 2023, arXiv:2304.01373)
- Cerebras-GPT paper (Dey et al., 2023, arXiv:2304.03208)
- OPT paper (Zhang et al., 2022, arXiv:2205.01068)
- GPT-NeoX paper (Black et al., 2022, arXiv:2204.06745)
- Falcon blog post (TII, 2023, huggingface.co/blog/falcon)

Scores are from the Open LLM Leaderboard v1 (normalized 25-shot ARC-C,
10-shot HellaSwag, 5-shot MMLU, 5-shot WinoGrande, 0-shot TruthfulQA,
5-shot GSM8K) cross-referenced with original papers where available.
"""

import hashlib
import json
import numpy as np

# Benchmark names (columns)
BENCHMARKS = [
    "ARC-Challenge",
    "HellaSwag",
    "MMLU",
    "WinoGrande",
    "TruthfulQA",
    "GSM8K",
]

SOURCE_MANIFEST = {
    "open_llm_leaderboard_v1": {
        "dataset_name": "Open LLM Leaderboard v1",
        "snapshot_date": "2024-06-15",
        "url": "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
        "notes": (
            "Archived v1 benchmark settings: ARC-Challenge 25-shot, HellaSwag 10-shot, "
            "MMLU 5-shot, WinoGrande 5-shot, TruthfulQA 0-shot, GSM8K 5-shot."
        ),
    },
    "cross_reference_sources": [
        "arXiv:2307.09288 (Llama 2)",
        "arXiv:2310.06825 (Mistral 7B)",
        "arXiv:2304.01373 (Pythia)",
        "arXiv:2304.03208 (Cerebras-GPT)",
        "arXiv:2205.01068 (OPT)",
        "arXiv:2204.06745 (GPT-NeoX)",
        "https://huggingface.co/blog/falcon (Falcon)",
    ],
}

# Model metadata: (name, family, params_billions)
MODEL_INFO = [
    # Llama-2 family (Meta, arXiv:2307.09288)
    ("Llama-2-7B", "Llama-2", 7.0),
    ("Llama-2-13B", "Llama-2", 13.0),
    ("Llama-2-70B", "Llama-2", 70.0),
    # Llama-1 family (Meta, arXiv:2302.13971)
    ("Llama-1-7B", "Llama-1", 7.0),
    ("Llama-1-13B", "Llama-1", 13.0),
    ("Llama-1-30B", "Llama-1", 30.0),
    ("Llama-1-65B", "Llama-1", 65.0),
    # Mistral family (Mistral AI, arXiv:2310.06825)
    ("Mistral-7B", "Mistral", 7.0),
    # Falcon family (TII, huggingface.co/blog/falcon)
    ("Falcon-7B", "Falcon", 7.0),
    ("Falcon-40B", "Falcon", 40.0),
    # Pythia family (EleutherAI, arXiv:2304.01373)
    ("Pythia-70M", "Pythia", 0.07),
    ("Pythia-160M", "Pythia", 0.16),
    ("Pythia-410M", "Pythia", 0.41),
    ("Pythia-1B", "Pythia", 1.0),
    ("Pythia-1.4B", "Pythia", 1.4),
    ("Pythia-2.8B", "Pythia", 2.8),
    ("Pythia-6.9B", "Pythia", 6.9),
    ("Pythia-12B", "Pythia", 12.0),
    # OPT family (Meta, arXiv:2205.01068)
    ("OPT-125M", "OPT", 0.125),
    ("OPT-350M", "OPT", 0.35),
    ("OPT-1.3B", "OPT", 1.3),
    ("OPT-2.7B", "OPT", 2.7),
    ("OPT-6.7B", "OPT", 6.7),
    ("OPT-13B", "OPT", 13.0),
    ("OPT-30B", "OPT", 30.0),
    ("OPT-66B", "OPT", 66.0),
    # GPT-NeoX (EleutherAI, arXiv:2204.06745)
    ("GPT-NeoX-20B", "GPT-NeoX", 20.0),
    # GPT-Neo family (EleutherAI)
    ("GPT-Neo-125M", "GPT-Neo", 0.125),
    ("GPT-Neo-1.3B", "GPT-Neo", 1.3),
    ("GPT-Neo-2.7B", "GPT-Neo", 2.7),
    # Cerebras-GPT family (Cerebras, arXiv:2304.03208)
    ("Cerebras-GPT-111M", "Cerebras-GPT", 0.111),
    ("Cerebras-GPT-256M", "Cerebras-GPT", 0.256),
    ("Cerebras-GPT-590M", "Cerebras-GPT", 0.59),
    ("Cerebras-GPT-1.3B", "Cerebras-GPT", 1.3),
    ("Cerebras-GPT-2.7B", "Cerebras-GPT", 2.7),
    ("Cerebras-GPT-6.7B", "Cerebras-GPT", 6.7),
    ("Cerebras-GPT-13B", "Cerebras-GPT", 13.0),
    # MPT family (MosaicML)
    ("MPT-7B", "MPT", 7.0),
    ("MPT-30B", "MPT", 30.0),
    # StableLM (Stability AI)
    ("StableLM-Base-Alpha-7B", "StableLM", 7.0),
]

# Benchmark scores matrix (rows = models, cols = benchmarks)
# Order: ARC-Challenge, HellaSwag, MMLU, WinoGrande, TruthfulQA, GSM8K
# Sources: Open LLM Leaderboard v1, cross-referenced with original papers
SCORES = np.array([
    # Llama-2 family — Touvron et al. 2023, Open LLM Leaderboard
    [53.1, 77.7, 43.8, 74.0, 39.0, 14.5],   # Llama-2-7B
    [59.4, 82.1, 55.7, 76.7, 36.9, 24.4],   # Llama-2-13B
    [67.3, 87.3, 69.8, 83.7, 44.9, 54.1],   # Llama-2-70B
    # Llama-1 family — Touvron et al. 2023, Open LLM Leaderboard
    [50.9, 76.2, 35.1, 73.0, 34.0, 10.1],   # Llama-1-7B
    [56.3, 79.1, 46.4, 76.2, 34.7, 17.5],   # Llama-1-13B
    [62.5, 82.6, 58.4, 79.6, 37.0, 35.0],   # Llama-1-30B
    [63.5, 84.2, 63.4, 81.1, 39.0, 42.6],   # Llama-1-65B
    # Mistral — Jiang et al. 2023, Open LLM Leaderboard
    [59.6, 83.3, 62.5, 78.4, 42.5, 37.8],   # Mistral-7B
    # Falcon — TII, Open LLM Leaderboard
    [47.9, 78.1, 27.8, 72.5, 35.5,  5.5],   # Falcon-7B
    [61.9, 85.3, 55.4, 79.7, 42.3, 22.7],   # Falcon-40B
    # Pythia family — Biderman et al. 2023, Open LLM Leaderboard
    [21.6, 27.3, 25.9, 51.5, 47.0,  0.5],   # Pythia-70M
    [24.1, 30.4, 24.7, 52.2, 42.1,  0.4],   # Pythia-160M
    [26.2, 40.9, 27.3, 53.1, 40.5,  0.8],   # Pythia-410M
    [30.0, 47.2, 26.0, 53.5, 38.9,  1.1],   # Pythia-1B
    [32.9, 52.1, 25.5, 57.3, 38.7,  1.5],   # Pythia-1.4B
    [36.3, 60.7, 26.8, 60.2, 35.8,  2.0],   # Pythia-2.8B
    [40.6, 67.3, 26.1, 65.0, 32.8,  3.4],   # Pythia-6.9B
    [43.2, 70.9, 27.6, 67.0, 32.4,  4.5],   # Pythia-12B
    # OPT family — Zhang et al. 2022, Open LLM Leaderboard
    [22.4, 31.3, 26.0, 50.5, 42.9,  0.0],   # OPT-125M
    [23.6, 36.7, 26.1, 52.1, 40.8,  0.2],   # OPT-350M
    [29.6, 53.7, 24.7, 59.6, 38.7,  1.1],   # OPT-1.3B
    [31.3, 57.4, 25.4, 61.0, 37.4,  1.5],   # OPT-2.7B
    [37.2, 67.2, 25.0, 65.5, 33.5,  3.0],   # OPT-6.7B
    [39.9, 71.2, 24.9, 68.5, 33.6,  3.5],   # OPT-13B
    [44.7, 74.6, 26.1, 71.4, 35.4,  5.2],   # OPT-30B
    [48.4, 78.6, 27.5, 73.8, 37.1, 10.3],   # OPT-66B
    # GPT-NeoX — Black et al. 2022, Open LLM Leaderboard
    [45.7, 73.5, 25.0, 68.9, 32.2,  5.4],   # GPT-NeoX-20B
    # GPT-Neo — EleutherAI, Open LLM Leaderboard
    [22.1, 30.3, 25.6, 51.1, 45.2,  0.2],   # GPT-Neo-125M
    [31.1, 48.5, 24.8, 57.0, 37.7,  0.8],   # GPT-Neo-1.3B
    [36.0, 55.8, 25.4, 57.6, 36.3,  1.7],   # GPT-Neo-2.7B
    # Cerebras-GPT — Dey et al. 2023, Open LLM Leaderboard
    [19.5, 26.7, 25.0, 48.8, 44.8,  0.3],   # Cerebras-GPT-111M
    [21.1, 28.3, 25.2, 49.2, 42.6,  0.4],   # Cerebras-GPT-256M
    [23.0, 33.6, 25.1, 51.8, 41.5,  0.5],   # Cerebras-GPT-590M
    [27.5, 42.0, 26.6, 53.0, 38.5,  0.9],   # Cerebras-GPT-1.3B
    [31.7, 48.9, 25.6, 56.8, 35.2,  1.6],   # Cerebras-GPT-2.7B
    [38.4, 62.8, 26.5, 63.6, 33.0,  2.8],   # Cerebras-GPT-6.7B
    [43.9, 69.6, 26.7, 67.5, 33.9,  4.0],   # Cerebras-GPT-13B
    # MPT — MosaicML, Open LLM Leaderboard
    [47.7, 77.6, 30.8, 72.3, 33.4,  8.7],   # MPT-7B
    [55.7, 82.7, 46.9, 78.0, 35.9, 15.1],   # MPT-30B
    # StableLM — Stability AI, Open LLM Leaderboard
    [32.7, 41.2, 25.1, 57.8, 41.1,  1.3],   # StableLM-Base-Alpha-7B
])


def get_model_names():
    """Return list of model name strings."""
    return [info[0] for info in MODEL_INFO]


def get_model_families():
    """Return list of model family strings."""
    return [info[1] for info in MODEL_INFO]


def get_model_params():
    """Return array of parameter counts (in billions)."""
    return np.array([info[2] for info in MODEL_INFO])


def get_scores_dataframe():
    """Return scores as a dict with model names as keys and benchmark dicts as values."""
    names = get_model_names()
    result = {}
    for i, name in enumerate(names):
        result[name] = {bm: SCORES[i, j] for j, bm in enumerate(BENCHMARKS)}
    return result


def get_family_indices():
    """Return dict mapping family name -> list of row indices."""
    families = get_model_families()
    family_idx = {}
    for i, fam in enumerate(families):
        if fam not in family_idx:
            family_idx[fam] = []
        family_idx[fam].append(i)
    return family_idx


def get_data_fingerprint():
    """Return a deterministic SHA-256 fingerprint of benchmark metadata and scores."""
    payload = {
        "benchmarks": BENCHMARKS,
        "model_info": MODEL_INFO,
        "scores": np.round(SCORES, 6).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
