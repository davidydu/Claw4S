import os
from pathlib import Path

import pytest

from run import _configure_plot_cache, build_configs


def test_build_configs_max_charts_limits_output_size():
    configs = build_configs(start_year=2000, end_year=2001, max_charts=25)
    assert len(configs) == 25


def test_build_configs_max_charts_rejects_non_positive():
    with pytest.raises(ValueError, match="max_charts"):
        build_configs(start_year=2000, end_year=2001, max_charts=0)


def test_configure_plot_cache_sets_writable_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    _configure_plot_cache(str(tmp_path))

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    mpl_config = os.environ.get("MPLCONFIGDIR")

    assert xdg_cache is not None
    assert mpl_config is not None
    assert Path(xdg_cache).is_dir()
    assert Path(mpl_config).is_dir()
    assert Path(mpl_config).is_relative_to(Path(xdg_cache))
