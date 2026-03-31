"""Tests for modular arithmetic data generation."""

import pytest
import torch

from src.data import generate_modular_addition_data, split_dataset


class TestGenerateModularAdditionData:
    """Tests for generate_modular_addition_data."""

    def test_correct_number_of_examples(self):
        """Should generate p^2 examples."""
        data = generate_modular_addition_data(p=7)
        assert len(data["inputs"]) == 49
        assert len(data["labels"]) == 49

    def test_labels_in_range(self):
        """All labels should be in [0, p-1]."""
        p = 11
        data = generate_modular_addition_data(p=p)
        assert data["labels"].min() >= 0
        assert data["labels"].max() < p

    def test_labels_correct(self):
        """Labels should equal (a + b) mod p."""
        p = 5
        data = generate_modular_addition_data(p=p)
        for i in range(len(data["inputs"])):
            a, b = data["inputs"][i]
            expected = (a.item() + b.item()) % p
            assert data["labels"][i].item() == expected

    def test_inputs_are_long_tensors(self):
        """Inputs and labels should be long (int64) tensors."""
        data = generate_modular_addition_data(p=5)
        assert data["inputs"].dtype == torch.long
        assert data["labels"].dtype == torch.long

    def test_input_shape(self):
        """Inputs should be (p^2, 2) shaped."""
        p = 7
        data = generate_modular_addition_data(p=p)
        assert data["inputs"].shape == (49, 2)

    def test_p_returned(self):
        """Should return p in the data dict."""
        data = generate_modular_addition_data(p=13)
        assert data["p"] == 13

    def test_invalid_p_raises(self):
        """Should raise ValueError for p < 2."""
        with pytest.raises(ValueError, match="must be >= 2"):
            generate_modular_addition_data(p=1)

    def test_default_p_is_97(self):
        """Default p should be 97."""
        data = generate_modular_addition_data()
        assert data["p"] == 97
        assert len(data["inputs"]) == 97 * 97


class TestSplitDataset:
    """Tests for split_dataset."""

    def test_split_sizes(self):
        """Train and test sizes should match fraction."""
        data = generate_modular_addition_data(p=7)
        train, test = split_dataset(data, train_fraction=0.5)
        total = len(data["inputs"])
        assert len(train["inputs"]) == int(total * 0.5)
        assert len(test["inputs"]) == total - int(total * 0.5)

    def test_no_overlap(self):
        """Train and test should have no overlapping examples."""
        data = generate_modular_addition_data(p=5)
        train, test = split_dataset(data, train_fraction=0.6)

        train_set = set()
        for i in range(len(train["inputs"])):
            pair = (train["inputs"][i][0].item(), train["inputs"][i][1].item())
            train_set.add(pair)

        for i in range(len(test["inputs"])):
            pair = (test["inputs"][i][0].item(), test["inputs"][i][1].item())
            assert pair not in train_set

    def test_deterministic_with_seed(self):
        """Same seed should produce same split."""
        data = generate_modular_addition_data(p=7)
        train1, _ = split_dataset(data, 0.5, seed=42)
        train2, _ = split_dataset(data, 0.5, seed=42)
        assert torch.equal(train1["inputs"], train2["inputs"])

    def test_different_seeds_differ(self):
        """Different seeds should produce different splits."""
        data = generate_modular_addition_data(p=7)
        train1, _ = split_dataset(data, 0.5, seed=42)
        train2, _ = split_dataset(data, 0.5, seed=99)
        assert not torch.equal(train1["inputs"], train2["inputs"])

    def test_invalid_fraction_raises(self):
        """Should raise for fraction outside (0, 1)."""
        data = generate_modular_addition_data(p=5)
        with pytest.raises(ValueError, match="must be in"):
            split_dataset(data, 0.0)
        with pytest.raises(ValueError, match="must be in"):
            split_dataset(data, 1.0)
