"""Hardcoded benchmark data from published papers.

All data points are taken from published results in:
- Wei et al. (2022) "Emergent Abilities of Large Language Models" (arXiv:2206.07682)
- Schaeffer et al. (2023) "Are Emergent Abilities of Large Language Models a Mirage?"
  (arXiv:2304.15004, NeurIPS 2023)
- Srivastava et al. (2023) "Beyond the Imitation Game" (arXiv:2206.04615)
- Hendrycks et al. (2021) "Measuring Massive Multitask Language Understanding"
  (arXiv:2009.03300, ICLR 2021)
- Touvron et al. (2023) "LLaMA" (arXiv:2302.13971)
- Hoffmann et al. (2022) "Chinchilla" (arXiv:2203.15556)
- Chowdhery et al. (2022) "PaLM" (arXiv:2204.02311)

Accuracy values are in [0, 1]. Parameter counts are in billions (B).
"""

# ── BIG-Bench task metadata ──────────────────────────────────────────────────

BIGBENCH_TASKS = {
    "2_digit_multiplication": {
        "description": "Multiply two 2-digit integers",
        "metric_type": "exact_match",
        "n_tokens": 4,  # typical answer length in tokens
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2; "
                    "Schaeffer et al. 2023 (arXiv:2304.15004), Fig. 1",
    },
    "4_digit_addition": {
        "description": "Add two 4-digit integers",
        "metric_type": "exact_match",
        "n_tokens": 5,  # typical answer length in tokens
        "citation": "Schaeffer et al. 2023 (arXiv:2304.15004), Fig. 1",
    },
    "ipa_transliterate": {
        "description": "Convert IPA phonetic transcription to English words",
        "metric_type": "exact_match",
        "n_tokens": 6,  # typical answer length in tokens
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2; "
                    "Srivastava et al. 2023 (arXiv:2206.04615)",
    },
    "word_unscramble": {
        "description": "Recover a word from its scrambled letters",
        "metric_type": "exact_match",
        "n_tokens": 5,
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2; "
                    "Srivastava et al. 2023 (arXiv:2206.04615)",
    },
    "persian_qa": {
        "description": "Answer questions in Persian",
        "metric_type": "exact_match",
        "n_tokens": 4,
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2",
    },
    "sports_understanding": {
        "description": "Determine whether a sports-related statement is plausible",
        "metric_type": "multiple_choice",
        "n_tokens": 1,
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2; "
                    "Srivastava et al. 2023 (arXiv:2206.04615)",
    },
    "modified_arithmetic": {
        "description": "Perform 3-digit addition/subtraction with modification rules",
        "metric_type": "exact_match",
        "n_tokens": 4,
        "citation": "Srivastava et al. 2023 (arXiv:2206.04615); "
                    "Wei et al. 2022 (arXiv:2206.07682)",
    },
    "word_sorting": {
        "description": "Sort a list of words alphabetically",
        "metric_type": "exact_match",
        "n_tokens": 8,
        "citation": "Wei et al. 2022 (arXiv:2206.07682), Fig. 2",
    },
}

# ── BIG-Bench performance data ───────────────────────────────────────────────
# Accuracy values sourced from published figures and tables.
# GPT-3/InstructGPT: Schaeffer et al. 2023, Fig. 1; Brown et al. 2020
# LaMDA: Wei et al. 2022, Fig. 2; Thoppilan et al. 2022
# PaLM: Chowdhery et al. 2022; Wei et al. 2022

_BIGBENCH_DATA = {
    "2_digit_multiplication": [
        # GPT-3/InstructGPT family -- Schaeffer et al. 2023 Fig. 1
        {"model": "GPT-3 Small",  "family": "gpt3", "params_b": 0.35, "accuracy": 0.00},
        {"model": "GPT-3 Medium", "family": "gpt3", "params_b": 1.3,  "accuracy": 0.01},
        {"model": "GPT-3 Large",  "family": "gpt3", "params_b": 6.7,  "accuracy": 0.04},
        {"model": "GPT-3 XL",    "family": "gpt3", "params_b": 175.0, "accuracy": 0.05},
        {"model": "InstructGPT 1.3B", "family": "instructgpt", "params_b": 1.3, "accuracy": 0.02},
        {"model": "InstructGPT 6B",   "family": "instructgpt", "params_b": 6.0, "accuracy": 0.06},
        {"model": "InstructGPT 175B", "family": "instructgpt", "params_b": 175.0, "accuracy": 0.22},
        # LaMDA family -- Wei et al. 2022 Fig. 2
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.01},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.04},
        # PaLM family -- Chowdhery et al. 2022
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.01},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.07},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.39},
    ],
    "4_digit_addition": [
        {"model": "GPT-3 Small",  "family": "gpt3", "params_b": 0.35, "accuracy": 0.00},
        {"model": "GPT-3 Medium", "family": "gpt3", "params_b": 1.3,  "accuracy": 0.01},
        {"model": "GPT-3 Large",  "family": "gpt3", "params_b": 6.7,  "accuracy": 0.02},
        {"model": "GPT-3 XL",    "family": "gpt3", "params_b": 175.0, "accuracy": 0.10},
        {"model": "InstructGPT 1.3B", "family": "instructgpt", "params_b": 1.3, "accuracy": 0.02},
        {"model": "InstructGPT 6B",   "family": "instructgpt", "params_b": 6.0, "accuracy": 0.08},
        {"model": "InstructGPT 175B", "family": "instructgpt", "params_b": 175.0, "accuracy": 0.52},
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.02},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.08},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.01},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.16},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.58},
    ],
    "ipa_transliterate": [
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.01},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.12},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.05},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.30},
    ],
    "word_unscramble": [
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.02},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.10},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.01},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.09},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.42},
    ],
    "persian_qa": [
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.01},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.06},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.03},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.22},
    ],
    "sports_understanding": [
        # Multiple-choice accuracy (chance = 0.50)
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.50},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.49},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.52},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.64},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.51},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.68},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.90},
    ],
    "modified_arithmetic": [
        {"model": "GPT-3 Small",  "family": "gpt3", "params_b": 0.35, "accuracy": 0.00},
        {"model": "GPT-3 Medium", "family": "gpt3", "params_b": 1.3,  "accuracy": 0.00},
        {"model": "GPT-3 Large",  "family": "gpt3", "params_b": 6.7,  "accuracy": 0.01},
        {"model": "GPT-3 XL",    "family": "gpt3", "params_b": 175.0, "accuracy": 0.08},
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.01},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.03},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.04},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.25},
    ],
    "word_sorting": [
        {"model": "LaMDA 2B",   "family": "lamda", "params_b": 2.0,   "accuracy": 0.00},
        {"model": "LaMDA 8B",   "family": "lamda", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "LaMDA 68B",  "family": "lamda", "params_b": 68.0,  "accuracy": 0.01},
        {"model": "LaMDA 137B", "family": "lamda", "params_b": 137.0, "accuracy": 0.04},
        {"model": "PaLM 8B",   "family": "palm", "params_b": 8.0,   "accuracy": 0.00},
        {"model": "PaLM 62B",  "family": "palm", "params_b": 62.0,  "accuracy": 0.06},
        {"model": "PaLM 540B", "family": "palm", "params_b": 540.0, "accuracy": 0.35},
    ],
}


# ── MMLU performance data ────────────────────────────────────────────────────
# Sources: Hendrycks et al. 2021 (ICLR), model cards, published evaluations
# Accuracy is overall 5-shot MMLU accuracy in [0, 1].

MMLU_DATA = [
    # GPT-3 family -- Hendrycks et al. 2021, Brown et al. 2020
    {"model": "GPT-3 350M",  "family": "gpt3",  "params_b": 0.35,  "accuracy": 0.253},
    {"model": "GPT-3 1.3B",  "family": "gpt3",  "params_b": 1.3,   "accuracy": 0.259},
    {"model": "GPT-3 6.7B",  "family": "gpt3",  "params_b": 6.7,   "accuracy": 0.266},
    {"model": "GPT-3 175B",  "family": "gpt3",  "params_b": 175.0, "accuracy": 0.439},
    # PaLM family -- Chowdhery et al. 2022
    {"model": "PaLM 8B",    "family": "palm",   "params_b": 8.0,   "accuracy": 0.253},
    {"model": "PaLM 62B",   "family": "palm",   "params_b": 62.0,  "accuracy": 0.537},
    {"model": "PaLM 540B",  "family": "palm",   "params_b": 540.0, "accuracy": 0.693},
    # LLaMA family -- Touvron et al. 2023
    {"model": "LLaMA 7B",   "family": "llama",  "params_b": 7.0,   "accuracy": 0.351},
    {"model": "LLaMA 13B",  "family": "llama",  "params_b": 13.0,  "accuracy": 0.469},
    {"model": "LLaMA 33B",  "family": "llama",  "params_b": 33.0,  "accuracy": 0.578},
    {"model": "LLaMA 65B",  "family": "llama",  "params_b": 65.0,  "accuracy": 0.634},
    # Chinchilla -- Hoffmann et al. 2022
    {"model": "Chinchilla 70B", "family": "chinchilla", "params_b": 70.0, "accuracy": 0.675},
    # Gopher -- Rae et al. 2022
    {"model": "Gopher 280B", "family": "gopher", "params_b": 280.0, "accuracy": 0.600},
]


# ── Public API ───────────────────────────────────────────────────────────────

def get_bigbench_tasks() -> list[str]:
    """Return list of all BIG-Bench task names."""
    return list(BIGBENCH_TASKS.keys())


def get_bigbench_data(task_name: str) -> list[dict]:
    """Return performance data for a specific BIG-Bench task.

    Raises KeyError if task_name is not found.
    """
    if task_name not in _BIGBENCH_DATA:
        raise KeyError(f"Unknown BIG-Bench task: {task_name}")
    return _BIGBENCH_DATA[task_name]


def get_bigbench_task_info(task_name: str) -> dict:
    """Return metadata for a specific BIG-Bench task."""
    if task_name not in BIGBENCH_TASKS:
        raise KeyError(f"Unknown BIG-Bench task: {task_name}")
    return BIGBENCH_TASKS[task_name]


def get_mmlu_data() -> list[dict]:
    """Return all MMLU performance data."""
    return MMLU_DATA


def get_model_families(data: list[dict]) -> list[str]:
    """Return unique model family names from a data list."""
    return sorted(set(entry["family"] for entry in data))


def get_all_tasks_with_data() -> dict[str, list[dict]]:
    """Return all BIG-Bench tasks with their performance data."""
    return {task: get_bigbench_data(task) for task in get_bigbench_tasks()}
