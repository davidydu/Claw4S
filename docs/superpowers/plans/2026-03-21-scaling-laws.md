# Scaling Laws Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a statistical analysis that verifies neural scaling laws using published Cerebras-GPT and Pythia data, demonstrating that loss scaling is robust while task scaling is unreliable.

**Architecture:** Six-module `src/` package — data (hardcoded published results), scaling_models (formulations + model selection utilities), fitting (NLS + parametric bootstrap), analysis (5-phase pipeline), plots (matplotlib figures), report (markdown generator). Thin `run.py` orchestrates; `validate.py` checks outputs.

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-scaling-laws-design.md`

---

## File Map

```
submissions/scaling-laws/
├── SKILL.md                    # Executable skill
├── run.py                      # Thin orchestrator (~15 lines)
├── validate.py                 # Results validator
├── requirements.txt            # Pinned deps
├── conftest.py                 # Pytest path config
├── src/
│   ├── __init__.py             # Empty
│   ├── data.py                 # Hardcoded Cerebras-GPT + Pythia data with citations
│   ├── scaling_models.py       # 3 formulations + AIC/BIC/adj-R² utilities
│   ├── fitting.py              # NLS wrapper + parametric bootstrap + convergence handling
│   ├── analysis.py             # 5-phase analysis pipeline
│   ├── plots.py                # 5 publication-quality matplotlib figures
│   └── report.py               # Markdown report generator
├── tests/
│   ├── __init__.py             # Empty
│   ├── test_data.py            # Data integrity checks
│   ├── test_scaling_models.py  # Formulation math + model selection
│   ├── test_fitting.py         # Fitting recovery on synthetic data
│   └── test_analysis.py        # Analysis pipeline on synthetic data
├── research_note/
│   └── main.tex
└── results/                    # gitignored
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `submissions/scaling-laws/requirements.txt`
- Create: `submissions/scaling-laws/conftest.py`
- Create: `submissions/scaling-laws/src/__init__.py`
- Create: `submissions/scaling-laws/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
numpy==2.2.4
scipy==1.15.2
matplotlib==3.10.1
pytest==8.3.5
```

- [ ] **Step 2: Create conftest.py**

```python
# conftest.py — ensures pytest can import from src/
```

- [ ] **Step 3: Create empty __init__.py files**

`src/__init__.py` and `tests/__init__.py` — both empty.

- [ ] **Step 4: Create venv and install deps**

Run:
```bash
cd submissions/scaling-laws
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: All packages install successfully.

- [ ] **Step 5: Verify imports**

Run:
```bash
.venv/bin/python -c "import numpy, scipy, matplotlib; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add submissions/scaling-laws/requirements.txt submissions/scaling-laws/conftest.py submissions/scaling-laws/src/__init__.py submissions/scaling-laws/tests/__init__.py
git commit -m "feat(scaling-laws): project scaffolding with pinned deps"
```

---

### Task 2: Data Module — Cerebras-GPT

**Files:**
- Create: `submissions/scaling-laws/src/data.py`
- Create: `submissions/scaling-laws/tests/test_data.py`

- [ ] **Step 1: Write failing test for Cerebras-GPT data**

```python
# tests/test_data.py
from src.data import CEREBRAS_GPT, get_family_data


def test_cerebras_gpt_has_seven_models():
    """Cerebras-GPT suite has exactly 7 model sizes."""
    assert len(CEREBRAS_GPT["models"]) == 7


def test_cerebras_gpt_has_required_fields():
    """Every model must have params, training_tokens, pile_test_loss, and benchmark scores."""
    required = {"params", "training_tokens", "pile_test_loss"}
    benchmarks = {"lambada_acc", "hellaswag_acc", "piqa_acc", "winogrande_acc",
                  "arc_easy_acc", "arc_challenge_acc", "openbookqa_acc"}
    for name, model in CEREBRAS_GPT["models"].items():
        for field in required | benchmarks:
            assert field in model, f"{name} missing {field}"


def test_cerebras_gpt_losses_decrease_with_scale():
    """Larger models should have lower training loss."""
    models = list(CEREBRAS_GPT["models"].values())
    losses = [m["pile_test_loss"] for m in models]
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1], "Losses should decrease with model size"


def test_cerebras_gpt_params_increase():
    """Model sizes should be in ascending order."""
    models = list(CEREBRAS_GPT["models"].values())
    params = [m["params"] for m in models]
    for i in range(len(params) - 1):
        assert params[i] < params[i + 1]


def test_cerebras_gpt_benchmarks_in_valid_range():
    """All benchmark accuracies should be in [0, 1]."""
    benchmark_keys = {"lambada_acc", "hellaswag_acc", "piqa_acc", "winogrande_acc",
                      "arc_easy_acc", "arc_challenge_acc", "openbookqa_acc"}
    for name, model in CEREBRAS_GPT["models"].items():
        for key in benchmark_keys:
            val = model[key]
            assert 0.0 <= val <= 1.0, f"{name}.{key} = {val} out of range"


def test_get_family_data_returns_arrays():
    """get_family_data should return numpy arrays of params and values."""
    import numpy as np
    params, losses = get_family_data(CEREBRAS_GPT, "pile_test_loss")
    assert isinstance(params, np.ndarray)
    assert isinstance(losses, np.ndarray)
    assert len(params) == 7
    assert len(losses) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Implement src/data.py with Cerebras-GPT data**

```python
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
            "training_tokens": 5.12e9,
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


def get_benchmark_keys(family: dict) -> list[str]:
    """Return list of benchmark keys present in a model family."""
    first_model = next(iter(family["models"].values()))
    exclude = {"params", "training_tokens", "pile_test_loss", "non_emb_params"}
    return [k for k in first_model if k not in exclude]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/scaling-laws/src/data.py submissions/scaling-laws/tests/test_data.py
git commit -m "feat(scaling-laws): data module with Cerebras-GPT published values"
```

---

### Task 3: Data Module — Pythia Benchmarks

**Files:**
- Modify: `submissions/scaling-laws/src/data.py`
- Modify: `submissions/scaling-laws/tests/test_data.py`

The Pythia benchmark values must be extracted from the EleutherAI GitHub repo JSON files at `evals/pythia-v1/pythia-{size}/zero-shot/{size}_step143000.json`. The 1B model uses `pythia-1b-bf16/zero-shot/1b-bf16_step143000.json`.

- [ ] **Step 1: Fetch Pythia benchmark values**

Use WebFetch or Bash to download the JSON evaluation files from the EleutherAI GitHub repo for all 8 model sizes. Extract the 0-shot accuracy values for: LAMBADA (acc), WinoGrande (acc), PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm).

Note: HellaSwag is NOT available in the official Pythia evals.

- [ ] **Step 2: Write failing tests for Pythia data**

Add to `tests/test_data.py`:

```python
from src.data import PYTHIA


def test_pythia_has_eight_models():
    """Pythia suite has exactly 8 model sizes."""
    assert len(PYTHIA["models"]) == 8


def test_pythia_has_required_fields():
    """Every Pythia model must have params, non_emb_params, and benchmark scores."""
    required = {"params", "non_emb_params"}
    benchmarks = {"lambada_acc", "winogrande_acc", "piqa_acc",
                  "arc_easy_acc", "arc_challenge_acc"}
    for name, model in PYTHIA["models"].items():
        for field in required | benchmarks:
            assert field in model, f"Pythia {name} missing {field}"


def test_pythia_no_hellaswag():
    """Pythia official evals do NOT include HellaSwag."""
    for name, model in PYTHIA["models"].items():
        assert "hellaswag_acc" not in model, f"Pythia {name} should not have hellaswag"


def test_pythia_non_emb_params_less_than_total():
    """Non-embedding params should be less than total params."""
    for name, model in PYTHIA["models"].items():
        assert model["non_emb_params"] < model["params"], f"Pythia {name}"


def test_overlapping_benchmarks():
    """Cerebras-GPT and Pythia should share at least 4 benchmark keys."""
    from src.data import CEREBRAS_GPT
    cgpt_keys = set(get_benchmark_keys(CEREBRAS_GPT))
    pythia_keys = set(get_benchmark_keys(PYTHIA))
    overlap = cgpt_keys & pythia_keys
    assert len(overlap) >= 4, f"Only {len(overlap)} overlapping benchmarks: {overlap}"
```

- [ ] **Step 3: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_data.py::test_pythia_has_eight_models -v`
Expected: FAIL — `ImportError: cannot import name 'PYTHIA'`

- [ ] **Step 4: Add PYTHIA dict to src/data.py**

Add the `PYTHIA` dict with architecture data from paper Table 1 and benchmark values extracted in Step 1. Structure:

```python
# ---------------------------------------------------------------------------
# Pythia (Biderman et al., 2023)
# Paper: arXiv 2304.01373
# Training: The Pile, ~300B tokens, same data + same order for all sizes
# Benchmarks: EleutherAI GitHub evals/pythia-v1/ (0-shot, step 143000)
# Architecture: Paper Table 1 + HuggingFace configs
# NOTE: Training losses are NOT published as exact values (only wandb figures)
# NOTE: HellaSwag is NOT in the official Pythia evaluation suite
# ---------------------------------------------------------------------------
PYTHIA: dict = {
    "name": "Pythia",
    "source": "Biderman et al., 2023 (arXiv 2304.01373); GitHub evals/pythia-v1/",
    "dataset": "The Pile",
    "training_recipe": "Fixed budget (~300B tokens for all sizes)",
    "training_tokens_per_model": 300e9,
    "models": {
        "70M": {
            "params": 70e6,
            "non_emb_params": 18_915_328,
            "lambada_acc": ...,  # from GitHub JSON
            "winogrande_acc": ...,
            "piqa_acc": ...,
            "arc_easy_acc": ...,
            "arc_challenge_acc": ...,
        },
        # ... remaining 7 sizes with extracted values
    },
}
```

Fill in the `...` placeholders with the exact values extracted in Step 1.

- [ ] **Step 5: Run all data tests**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add submissions/scaling-laws/src/data.py submissions/scaling-laws/tests/test_data.py
git commit -m "feat(scaling-laws): add Pythia benchmark data from EleutherAI evals"
```

---

### Task 4: Scaling Models + Model Selection

**Files:**
- Create: `submissions/scaling-laws/src/scaling_models.py`
- Create: `submissions/scaling-laws/tests/test_scaling_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scaling_models.py
import numpy as np
from src.scaling_models import (
    kaplan_loss, chinchilla_loss, corrected_loss,
    compute_aic, compute_bic, adjusted_r_squared,
    FORMULATIONS,
)


def test_kaplan_loss_decreases_with_n():
    """Kaplan power-law: loss should decrease as N increases."""
    n_values = np.array([1e8, 1e9, 1e10])
    losses = kaplan_loss(n_values, a=1.0, alpha=0.07, l_inf=1.5)
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1]


def test_kaplan_loss_approaches_l_inf():
    """As N → ∞, Kaplan loss should approach L_inf."""
    loss = kaplan_loss(np.array([1e15]), a=1.0, alpha=0.07, l_inf=1.5)
    assert abs(loss[0] - 1.5) < 0.01


def test_chinchilla_loss_with_fixed_d_equals_kaplan():
    """When D is constant, Chinchilla reduces to Kaplan + constant."""
    n = np.array([1e8, 1e9, 1e10])
    d = np.full_like(n, 300e9)
    chin = chinchilla_loss(n, d, a=1.0, alpha=0.07, b=1.0, beta=0.07, l_inf=1.5)
    # Should still decrease with N
    for i in range(len(chin) - 1):
        assert chin[i] > chin[i + 1]


def test_corrected_loss_correction_vanishes_at_large_n():
    """Finite-size correction should vanish for large N."""
    large_n = np.array([1e15])
    corrected = corrected_loss(large_n, a=1.0, alpha=0.07, c=5.0, gamma=0.1, l_inf=1.5)
    kaplan = kaplan_loss(large_n, a=1.0, alpha=0.07, l_inf=1.5)
    assert abs(corrected[0] - kaplan[0]) < 0.01


def test_formulations_registry():
    """FORMULATIONS dict should contain all three named formulations."""
    assert "kaplan" in FORMULATIONS
    assert "chinchilla" in FORMULATIONS
    assert "corrected" in FORMULATIONS


def test_aic_prefers_simpler_model_on_identical_fit():
    """Given equal RSS, AIC should prefer fewer parameters."""
    n, k1, k2, rss = 7, 3, 5, 0.01
    aic1 = compute_aic(n, k1, rss)
    aic2 = compute_aic(n, k2, rss)
    assert aic1 < aic2  # lower AIC = better


def test_bic_penalizes_params_more_than_aic():
    """BIC penalty grows with log(n), should penalize more than AIC for n >= 8."""
    n, k, rss = 10, 5, 0.01
    aic = compute_aic(n, k, rss)
    bic = compute_bic(n, k, rss)
    assert bic > aic  # BIC more conservative for n >= 8


def test_adjusted_r_squared_less_than_r_squared():
    """Adjusted R² should be less than or equal to R² for k > 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_pred = y + np.array([0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.05])
    adj_r2 = adjusted_r_squared(y, y_pred, k=3)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert adj_r2 <= r2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_scaling_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement src/scaling_models.py**

```python
"""Scaling law formulations and model selection utilities.

Three functional forms for L(N):
  - Kaplan power-law: L = a * N^(-alpha) + L_inf
  - Chinchilla two-term: L = a * N^(-alpha) + b * D^(-beta) + L_inf
  - Corrected power-law: L = a * N^(-alpha) * (1 + c * N^(-gamma)) + L_inf
"""
from __future__ import annotations

import numpy as np


# --- Scaling law formulations ---

def kaplan_loss(n: np.ndarray, a: float, alpha: float, l_inf: float) -> np.ndarray:
    """Kaplan power-law: L(N) = a * N^(-alpha) + L_inf."""
    return a * np.power(n, -alpha) + l_inf


def chinchilla_loss(
    n: np.ndarray, d: np.ndarray,
    a: float, alpha: float, b: float, beta: float, l_inf: float,
) -> np.ndarray:
    """Chinchilla two-term: L(N,D) = a * N^(-alpha) + b * D^(-beta) + L_inf."""
    return a * np.power(n, -alpha) + b * np.power(d, -beta) + l_inf


def corrected_loss(
    n: np.ndarray, a: float, alpha: float, c: float, gamma: float, l_inf: float,
) -> np.ndarray:
    """Power-law with finite-size correction:
    L(N) = a * N^(-alpha) * (1 + c * N^(-gamma)) + L_inf."""
    return a * np.power(n, -alpha) * (1.0 + c * np.power(n, -gamma)) + l_inf


# Registry of formulations for automated iteration
FORMULATIONS: dict[str, dict] = {
    "kaplan": {
        "func": kaplan_loss,
        "param_names": ["a", "alpha", "l_inf"],
        "n_params": 3,
        "needs_d": False,
    },
    "chinchilla": {
        "func": chinchilla_loss,
        "param_names": ["a", "alpha", "b", "beta", "l_inf"],
        "n_params": 5,
        "needs_d": True,
    },
    "corrected": {
        "func": corrected_loss,
        "param_names": ["a", "alpha", "c", "gamma", "l_inf"],
        "n_params": 5,
        "needs_d": False,
    },
}


# --- Model selection utilities ---

def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion. Lower is better.

    AIC = n * ln(RSS/n) + 2k
    """
    return n * np.log(rss / n) + 2 * k


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion. Lower is better.

    BIC = n * ln(RSS/n) + k * ln(n)
    """
    return n * np.log(rss / n) + k * np.log(n)


def adjusted_r_squared(y: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Adjusted R² that penalizes number of parameters k.

    adj_R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)
    """
    n = len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scaling_models.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/scaling-laws/src/scaling_models.py submissions/scaling-laws/tests/test_scaling_models.py
git commit -m "feat(scaling-laws): scaling law formulations and model selection utilities"
```

---

### Task 5: Fitting Module

**Files:**
- Create: `submissions/scaling-laws/src/fitting.py`
- Create: `submissions/scaling-laws/tests/test_fitting.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fitting.py
import numpy as np
from src.fitting import fit_scaling_law, parametric_bootstrap, FitResult


def test_fit_recovers_known_kaplan_params():
    """Fitting should recover parameters used to generate synthetic data."""
    np.random.seed(42)
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    true_a, true_alpha, true_linf = 5.0, 0.07, 1.5
    y = true_a * np.power(n, -true_alpha) + true_linf
    y += np.random.normal(0, 0.005, len(y))  # small noise

    result = fit_scaling_law("kaplan", n, y)
    assert isinstance(result, FitResult)
    assert abs(result.params["alpha"] - true_alpha) < 0.02
    assert abs(result.params["l_inf"] - true_linf) < 0.1
    assert result.adj_r_squared > 0.95


def test_fit_result_has_required_fields():
    """FitResult should contain params, residuals, adj_r_squared, aic, bic."""
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    y = 5.0 * np.power(n, -0.07) + 1.5

    result = fit_scaling_law("kaplan", n, y)
    assert hasattr(result, "params")
    assert hasattr(result, "residuals")
    assert hasattr(result, "adj_r_squared")
    assert hasattr(result, "aic")
    assert hasattr(result, "bic")
    assert hasattr(result, "converged")
    assert result.converged is True


def test_bootstrap_returns_ci():
    """Parametric bootstrap should return confidence intervals for each parameter."""
    np.random.seed(42)
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    y = 5.0 * np.power(n, -0.07) + 1.5 + np.random.normal(0, 0.01, 7)

    result = fit_scaling_law("kaplan", n, y)
    ci = parametric_bootstrap("kaplan", n, result, n_bootstrap=100, seed=42)
    assert "alpha" in ci
    assert len(ci["alpha"]) == 2  # (lower, upper)
    assert ci["alpha"][0] < ci["alpha"][1]  # lower < upper
    assert "convergence_rate" in ci


def test_fit_handles_convergence_failure_gracefully():
    """Fitting with nonsensical data should return converged=False, not raise."""
    n = np.array([1.0, 2.0, 3.0])  # too few / bad data
    y = np.array([100.0, 100.0, 100.0])  # constant — no scaling
    result = fit_scaling_law("kaplan", n, y)
    # Should return a result (possibly with converged=False), not raise
    assert isinstance(result, FitResult)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fitting.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement src/fitting.py**

Implement:
- `FitResult` dataclass with fields: `name`, `params` (dict), `param_values` (array), `residuals`, `y_pred`, `adj_r_squared`, `aic`, `bic`, `converged`
- `fit_scaling_law(name, n, y, d=None, n_restarts=10, seed=42) -> FitResult` — wraps `scipy.optimize.curve_fit` with multiple random restarts, catches `RuntimeError`, uses log-log space fitting
- `parametric_bootstrap(name, n, fit_result, n_bootstrap=1000, seed=42, d=None) -> dict` — generates synthetic data from fitted model + residual noise, refits each, returns 95% CIs and convergence rate

Key implementation details:
- Fitting in log-log space: transform L → log(L), N → log(N), fit, transform back
- Multiple restarts: sample initial guesses from reasonable ranges (a ∈ [0.1, 100], alpha ∈ [0.01, 0.5], l_inf ∈ [0.5, 3.0])
- Convergence handling: catch `RuntimeError` and `OptimizeWarning`, return `converged=False`
- Bootstrap: filter degenerate fits (alpha < 0 or > 1, l_inf < 0), report convergence rate

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fitting.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add submissions/scaling-laws/src/fitting.py submissions/scaling-laws/tests/test_fitting.py
git commit -m "feat(scaling-laws): NLS fitting with parametric bootstrap and convergence handling"
```

---

### Task 6: Analysis Module — Loss Scaling + Task Scaling

**Files:**
- Create: `submissions/scaling-laws/src/analysis.py`
- Create: `submissions/scaling-laws/tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analysis.py
import numpy as np
from src.analysis import (
    run_loss_scaling,
    run_task_scaling,
    run_cross_metric_correlation,
    run_extrapolation_risk,
    run_cross_family_transfer,
    run_full_analysis,
)


def test_run_loss_scaling_returns_expected_structure():
    """Loss scaling should return fits for all three formulations."""
    from src.data import CEREBRAS_GPT
    result = run_loss_scaling(CEREBRAS_GPT, n_bootstrap=50, seed=42)
    assert "kaplan" in result
    assert "chinchilla" in result
    assert "corrected" in result
    for name, fit in result.items():
        assert "params" in fit
        assert "adj_r_squared" in fit
        assert "aic" in fit
        assert "bic" in fit


def test_run_task_scaling_returns_per_task_results():
    """Task scaling should return results for each benchmark."""
    from src.data import CEREBRAS_GPT
    result = run_task_scaling(CEREBRAS_GPT, n_bootstrap=50, seed=42)
    assert len(result) >= 5  # at least 5 benchmarks
    for task_name, task_result in result.items():
        assert "bounded_power_law" in task_result
        assert "sigmoid" in task_result


def test_run_full_analysis_returns_all_phases():
    """Full analysis should return results for all 5 phases."""
    result = run_full_analysis(n_bootstrap=50, seed=42)
    assert "loss_scaling" in result
    assert "task_scaling" in result
    assert "cross_metric" in result
    assert "extrapolation" in result
    assert "cross_family" in result
    assert "metadata" in result
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_analysis.py::test_run_loss_scaling_returns_expected_structure -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement src/analysis.py**

Implement the 5-phase analysis pipeline:

1. `run_loss_scaling(family, n_bootstrap=1000, seed=42)` — fits 3 formulations to Cerebras-GPT losses, runs bootstrap, returns params + CIs + model selection
2. `run_task_scaling(family, n_bootstrap=1000, seed=42)` — for each benchmark, fits bounded power-law `acc(N) = 1 - a*N^(-alpha)` and sigmoid `acc(N) = L / (1 + exp(-k*(log(N) - x0)))`, runs bootstrap
3. `run_cross_metric_correlation(family)` — computes Δloss vs Δaccuracy, Pearson/Spearman correlations
4. `run_extrapolation_risk(family, n_train=4, n_bootstrap=1000, seed=42)` — fits on smallest `n_train` models, predicts rest, reports MAPE and prediction intervals
5. `run_cross_family_transfer(primary, secondary, n_bootstrap=1000, seed=42)` — fits task scaling on primary (Cerebras-GPT), predicts secondary (Pythia) benchmarks, reports transfer error
6. `run_full_analysis(n_bootstrap=1000, seed=42)` — orchestrates all 5 phases, prints `[1/5]` through `[5/5]` banners, saves `results/results.json`, returns full results dict with metadata (timestamp, versions, seed)

Each phase catches exceptions independently — a failure in one phase doesn't block others.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis.py -v`
Expected: All tests PASS (may take ~10-15 seconds due to bootstrap with n=50).

- [ ] **Step 5: Commit**

```bash
git add submissions/scaling-laws/src/analysis.py submissions/scaling-laws/tests/test_analysis.py
git commit -m "feat(scaling-laws): 5-phase analysis pipeline with cross-family transfer"
```

---

### Task 7: Plots Module

**Files:**
- Create: `submissions/scaling-laws/src/plots.py`

- [ ] **Step 1: Implement src/plots.py**

No TDD for plots — visual output is validated by existence checks in `validate.py`.

Implement `generate_all_plots(results: dict, output_dir: str = "results/figures")` that creates 5 PNG files:

1. `loss_scaling.png` — log-log plot of loss vs params with 3 fitted curves + CI bands. Cerebras-GPT data points as markers.
2. `task_scaling.png` — 2×4 subplot grid (7 tasks + 1 summary). Each subplot: data points + fitted bounded-power-law curve.
3. `residuals.png` — side-by-side: loss residuals (left) vs task residuals (right, averaged across tasks). Shows structured vs random residuals.
4. `model_selection.png` — grouped bar chart: AIC and BIC for each of the 3 loss formulations.
5. `extrapolation.png` — prediction intervals: fit on small models (filled markers), predict large models (open markers with error bars). Loss (left) vs task (right).

All plots use `matplotlib.pyplot`, `plt.savefig(dpi=150, bbox_inches='tight')`, and `plt.close()` after saving.

- [ ] **Step 2: Commit**

```bash
git add submissions/scaling-laws/src/plots.py
git commit -m "feat(scaling-laws): 5 publication-quality matplotlib figures"
```

---

### Task 8: Report Module

**Files:**
- Create: `submissions/scaling-laws/src/report.py`

- [ ] **Step 1: Implement src/report.py**

Implement `generate_report(results: dict) -> str` that produces a markdown report with sections:

1. **Summary** — one-paragraph overview of findings
2. **Loss Scaling Results** — table of fitted parameters + CIs for 3 formulations, model selection scores, best-fit adj-R²
3. **Task Scaling Results** — table per task: adj-R² for bounded power-law and sigmoid fits. Highlight tasks with poor scaling.
4. **Cross-Metric Correlation** — Pearson/Spearman correlations between Δloss and Δaccuracy
5. **Extrapolation Risk** — MAPE for loss vs task predictions, prediction interval widths
6. **Cross-Family Transfer** — transfer prediction error from Cerebras-GPT → Pythia
7. **Methodology** — brief description of statistical methods used
8. **Limitations** — small sample (n=7), HellaSwag excluded from Pythia, Chinchilla model identifiability

Also implement `save_report(report: str, path: str = "results/report.md")`.

- [ ] **Step 2: Commit**

```bash
git add submissions/scaling-laws/src/report.py
git commit -m "feat(scaling-laws): markdown report generator"
```

---

### Task 9: Integration — run.py + validate.py

**Files:**
- Create: `submissions/scaling-laws/run.py`
- Create: `submissions/scaling-laws/validate.py`

- [ ] **Step 1: Create run.py**

```python
"""Run the full scaling laws analysis.

Usage: .venv/bin/python run.py
"""
from src.analysis import run_full_analysis
from src.plots import generate_all_plots
from src.report import generate_report, save_report

results = run_full_analysis(n_bootstrap=1000, seed=42)
generate_all_plots(results)
report = generate_report(results)
save_report(report)
print(report)
```

- [ ] **Step 2: Create validate.py**

Checks:
- `results/results.json` exists and is valid JSON
- All 3 loss formulations present with finite params and CIs
- Bootstrap convergence rate ≥ 90% for each formulation
- Loss scaling best adj-R² > 0.90
- At least 2 tasks have adj-R² < 0.85 for bounded power-law
- All 5 figure PNGs exist in `results/figures/`
- `results/report.md` exists
- Kaplan alpha ∈ [0.02, 0.20]

Pattern: accumulate `errors = []`, print checks, exit(1) if errors.

- [ ] **Step 3: Run end-to-end**

```bash
cd submissions/scaling-laws
.venv/bin/python run.py
.venv/bin/python validate.py
```

Expected: run.py prints `[1/5]` through `[5/5]` banners and the report. validate.py prints checks and `Validation passed.`

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add submissions/scaling-laws/run.py submissions/scaling-laws/validate.py
git commit -m "feat(scaling-laws): run.py orchestrator and validate.py checker"
```

---

### Task 10: SKILL.md

**Files:**
- Create: `submissions/scaling-laws/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

Follow the same structure as `submissions/tokenizer-analysis/SKILL.md`:

```markdown
---
name: scaling-laws-verification
description: Verify neural scaling laws using published Cerebras-GPT and Pythia data. Fits Kaplan, Chinchilla, and corrected power-law formulations, compares loss scaling (robust) vs task scaling (unreliable), and quantifies extrapolation risk.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Scaling Laws Verification

[Steps: Prerequisites, Setup, Unit Tests, Run Analysis, Validate, Review Report, How to Extend]
```

Steps follow the exact pattern from tokenizer-analysis:
1. Prerequisites (Python 3.10+, internet not needed, ~2 min runtime)
2. Environment setup (venv + pip install)
3. Run unit tests (pytest)
4. Run analysis (run.py)
5. Validate results (validate.py)
6. Review report (cat results/report.md)
7. How to Extend section

- [ ] **Step 2: Commit**

```bash
git add submissions/scaling-laws/SKILL.md
git commit -m "feat(scaling-laws): executable SKILL.md for Claw agent"
```

---

### Task 11: Research Note

**Files:**
- Create: `submissions/scaling-laws/research_note/main.tex`

- [ ] **Step 1: Write LaTeX research note**

1-4 page paper using Claw4S LaTeX template. Sections:
1. Introduction — scaling laws background, thesis (loss robust, tasks not)
2. Data — Cerebras-GPT (7 sizes), Pythia (8 sizes), all public
3. Methods — 3 formulations, parametric bootstrap, AIC/BIC, breakpoint detection, extrapolation risk
4. Results — loss scaling confirms power-law (α ≈ X, adj-R² > Y), task scaling shows poor fit for Z/7 tasks, extrapolation risk N× larger for tasks
5. Discussion — implications for compute planning, emergent abilities debate
6. Conclusion + How to Extend

Authors: Yun Du, Lina Ji, Claw (the-mad-lobster)

- [ ] **Step 2: Commit**

```bash
git add submissions/scaling-laws/research_note/main.tex
git commit -m "feat(scaling-laws): research note LaTeX document"
```

---

### Task 12: E2E Cold-Start Validation

**Files:** None (validation only)

- [ ] **Step 1: Delete runtime artifacts**

```bash
cd submissions/scaling-laws
rm -rf .venv results __pycache__ .pytest_cache
```

- [ ] **Step 2: Run SKILL.md from scratch**

Follow SKILL.md exactly, step by step, as Claw would:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
.venv/bin/python run.py
.venv/bin/python validate.py
cat results/report.md
```

- [ ] **Step 3: Verify all steps pass**

Expected:
- All tests pass
- run.py completes in < 2 minutes
- validate.py prints "Validation passed."
- report.md contains all expected sections

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A submissions/scaling-laws/
git commit -m "fix(scaling-laws): E2E cold-start fixes"
```
