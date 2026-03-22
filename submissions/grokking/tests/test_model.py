"""Tests for the GrokkingMLP model."""

import torch

from src.model import GrokkingMLP


class TestGrokkingMLP:
    """Tests for the GrokkingMLP architecture."""

    def test_output_shape(self):
        """Output should be (batch_size, p)."""
        p, batch = 7, 10
        model = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)
        x = torch.randint(0, p, (batch, 2))
        out = model(x)
        assert out.shape == (batch, p)

    def test_parameter_count_h16(self):
        """Hidden dim 16 should be well under 100K params."""
        model = GrokkingMLP(p=97, embed_dim=16, hidden_dim=16)
        count = model.count_parameters()
        assert count < 100_000
        assert count > 0

    def test_parameter_count_h32(self):
        """Hidden dim 32 should be well under 100K params."""
        model = GrokkingMLP(p=97, embed_dim=16, hidden_dim=32)
        count = model.count_parameters()
        assert count < 100_000

    def test_parameter_count_h64(self):
        """Hidden dim 64 should be well under 100K params."""
        model = GrokkingMLP(p=97, embed_dim=16, hidden_dim=64)
        count = model.count_parameters()
        assert count < 100_000

    def test_deterministic_with_seed(self):
        """Same seed should produce same outputs."""
        p = 7
        x = torch.tensor([[1, 2], [3, 4]])

        torch.manual_seed(42)
        m1 = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)

        torch.manual_seed(42)
        m2 = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)

        with torch.no_grad():
            assert torch.equal(m1(x), m2(x))

    def test_forward_pass_no_error(self):
        """Forward pass should complete without errors."""
        model = GrokkingMLP(p=97, embed_dim=16, hidden_dim=64)
        x = torch.randint(0, 97, (32, 2))
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_stores_config(self):
        """Model should store its configuration."""
        model = GrokkingMLP(p=97, embed_dim=16, hidden_dim=64)
        assert model.p == 97
        assert model.embed_dim == 16
        assert model.hidden_dim == 64
