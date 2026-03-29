"""Tests for experiment orchestration and shock-node selection."""

import src.experiment as exp
from src.network import chain, fully_connected


def test_select_shock_node_random_uses_non_hub_when_available():
    adj = chain(20)
    # In a chain, only endpoints are non-hubs (degree 1).
    non_hubs = {0, 19}
    sampled = {exp.select_shock_node(adj, "random", seed=s) for s in [1, 2, 3, 4, 5]}
    assert sampled.issubset(non_hubs)


def test_select_shock_node_random_falls_back_when_all_nodes_are_hubs():
    adj = fully_connected(6)
    node = exp.select_shock_node(adj, "random", seed=42)
    assert node in set(range(6))


def test_run_experiment_subset_does_not_mutate_global_axes():
    original_topologies = list(exp.TOPOLOGY_NAMES)
    original_agents = list(exp.AGENT_TYPE_NAMES)

    results = exp.run_experiment(
        n_workers=1,
        topology_names=["ring"],
        agent_type_names=["fragile"],
        shock_magnitudes={"mild": 2.0},
        shock_locations=["hub", "random"],
        seeds=[42],
    )

    assert len(results) == 2
    assert {r["shock_location"] for r in results} == {"hub", "random"}
    assert exp.TOPOLOGY_NAMES == original_topologies
    assert exp.AGENT_TYPE_NAMES == original_agents
