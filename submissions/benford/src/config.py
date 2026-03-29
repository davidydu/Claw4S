"""Experiment configuration helpers for run.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Resolved runtime configuration for the Benford experiments."""

    seed: int
    hidden_sizes: list[int]
    snapshot_epochs: list[int]
    mod_p: int
    sine_n: int
    epochs: int
    lr: float
    controls_n: int
    log_every: int
    make_plots: bool
    quick: bool


FULL_DEFAULTS = {
    "seed": 42,
    "hidden_sizes": [64, 128],
    "snapshot_epochs": [0, 100, 500, 1000, 2000, 5000],
    "mod_p": 97,
    "sine_n": 1000,
    "epochs": 5000,
    "lr": 1e-3,
    "controls_n": 10000,
    "log_every": 1000,
}

QUICK_DEFAULTS = {
    "seed": 42,
    "hidden_sizes": [64],
    "snapshot_epochs": [0, 50, 100, 200, 500],
    "mod_p": 97,
    "sine_n": 500,
    "epochs": 500,
    "lr": 1e-3,
    "controls_n": 2500,
    "log_every": 100,
}


def parse_int_list(raw, field_name):
    """Parse a comma-separated list of positive integers."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{field_name} must be a non-empty comma-separated list.")

    values = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must contain integers only, got '{part}'."
            ) from exc
        if value <= 0:
            raise ValueError(f"{field_name} values must be > 0, got {value}.")
        values.append(value)

    return values


def _normalize_snapshot_epochs(snapshot_epochs, epochs):
    """Ensure snapshot epochs are unique, sorted, and within [0, epochs]."""
    valid = {e for e in snapshot_epochs if 0 <= e <= epochs}
    valid.add(0)
    valid.add(epochs)
    return sorted(valid)


def resolve_run_config(
    *,
    quick=False,
    seed=None,
    hidden_sizes=None,
    snapshot_epochs=None,
    mod_p=None,
    sine_n=None,
    epochs=None,
    lr=None,
    controls_n=None,
    log_every=None,
    make_plots=True,
):
    """Resolve final runtime config from defaults plus overrides."""
    base = QUICK_DEFAULTS if quick else FULL_DEFAULTS

    resolved_epochs = base["epochs"] if epochs is None else int(epochs)
    if resolved_epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {resolved_epochs}.")

    resolved_hidden_sizes = (
        base["hidden_sizes"] if hidden_sizes is None else list(hidden_sizes)
    )
    if not resolved_hidden_sizes:
        raise ValueError("hidden_sizes must not be empty.")
    if any(h <= 0 for h in resolved_hidden_sizes):
        raise ValueError(f"hidden_sizes values must be > 0, got {resolved_hidden_sizes}.")

    raw_snapshots = base["snapshot_epochs"] if snapshot_epochs is None else list(snapshot_epochs)
    resolved_snapshots = _normalize_snapshot_epochs(raw_snapshots, resolved_epochs)

    resolved_mod_p = base["mod_p"] if mod_p is None else int(mod_p)
    resolved_sine_n = base["sine_n"] if sine_n is None else int(sine_n)
    resolved_seed = base["seed"] if seed is None else int(seed)
    resolved_lr = base["lr"] if lr is None else float(lr)
    resolved_controls_n = base["controls_n"] if controls_n is None else int(controls_n)
    resolved_log_every = base["log_every"] if log_every is None else int(log_every)

    if resolved_mod_p <= 1:
        raise ValueError(f"mod_p must be > 1, got {resolved_mod_p}.")
    if resolved_sine_n <= 0:
        raise ValueError(f"sine_n must be > 0, got {resolved_sine_n}.")
    if resolved_controls_n <= 0:
        raise ValueError(f"controls_n must be > 0, got {resolved_controls_n}.")
    if resolved_log_every <= 0:
        raise ValueError(f"log_every must be > 0, got {resolved_log_every}.")

    return RunConfig(
        seed=resolved_seed,
        hidden_sizes=resolved_hidden_sizes,
        snapshot_epochs=resolved_snapshots,
        mod_p=resolved_mod_p,
        sine_n=resolved_sine_n,
        epochs=resolved_epochs,
        lr=resolved_lr,
        controls_n=resolved_controls_n,
        log_every=resolved_log_every,
        make_plots=make_plots,
        quick=quick,
    )
