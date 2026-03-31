"""Tests for plotting helpers."""

from matplotlib.axes import Axes

from src.plots import plot_mlp_comparison


def test_plot_mlp_comparison_uses_passed_mlp_threshold(tmp_path, monkeypatch):
    """The MLP threshold marker should respect the configured threshold."""
    original_axvline = Axes.axvline
    captured_x = []

    def capture_axvline(self, x=0, *args, **kwargs):
        captured_x.append(x)
        return original_axvline(self, x=x, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", capture_axvline)

    plot_mlp_comparison(
        mlp_results=[
            {"width": 8, "param_ratio": 0.9, "train_loss": 0.5, "test_loss": 1.2},
            {"width": 16, "param_ratio": 1.8, "train_loss": 0.1, "test_loss": 0.9},
        ],
        rf_results=[
            {"width": 100, "train_loss": 0.1, "test_loss": 1.5},
            {"width": 200, "train_loss": 0.0, "test_loss": 10.0},
        ],
        n_train=200,
        mlp_interpolation_threshold=17,
        output_path=str(tmp_path / "mlp.png"),
    )

    assert 200 in captured_x
    assert 17 in captured_x
