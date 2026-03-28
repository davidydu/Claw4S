# tests/test_data.py
from src.data import CEREBRAS_GPT, get_family_data, PYTHIA, get_benchmark_keys


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


def test_get_training_tokens_returns_array():
    """get_training_tokens should return token counts for Chinchilla fitting."""
    import numpy as np
    from src.data import get_training_tokens
    tokens = get_training_tokens(CEREBRAS_GPT)
    assert isinstance(tokens, np.ndarray)
    assert len(tokens) == 7
    assert tokens[0] < tokens[-1]  # should increase with model size


def test_pythia_has_eight_models():
    """Pythia suite has exactly 8 model sizes."""
    assert len(PYTHIA["models"]) == 8


def test_pythia_has_required_fields():
    """Every Pythia model must have params, non_emb_params, and benchmark scores."""
    required = {"params", "non_emb_params", "training_tokens"}
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
    cgpt_keys = set(get_benchmark_keys(CEREBRAS_GPT))
    pythia_keys = set(get_benchmark_keys(PYTHIA))
    overlap = cgpt_keys & pythia_keys
    assert len(overlap) >= 4, f"Only {len(overlap)} overlapping benchmarks: {overlap}"
