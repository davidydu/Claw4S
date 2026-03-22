"""Tests for metrics aggregation and analysis."""

import math
from src.metrics import (
    aggregate_by_condition,
    topology_ranking,
    hub_vs_random_comparison,
    agent_type_comparison,
)


SAMPLE_RESULTS = [
    {"topology": "ring", "agent_type": "fragile", "shock_magnitude": 10.0,
     "shock_location": "random", "shock_node": 5, "seed": 42,
     "cascade_size": 0.8, "cascade_speed": 5.0, "recovery_time": 100.0,
     "systemic_risk": 1.2},
    {"topology": "ring", "agent_type": "fragile", "shock_magnitude": 10.0,
     "shock_location": "random", "shock_node": 5, "seed": 123,
     "cascade_size": 0.7, "cascade_speed": 6.0, "recovery_time": 90.0,
     "systemic_risk": 1.1},
    {"topology": "ring", "agent_type": "fragile", "shock_magnitude": 10.0,
     "shock_location": "hub", "shock_node": 0, "seed": 42,
     "cascade_size": 0.9, "cascade_speed": 3.0, "recovery_time": 120.0,
     "systemic_risk": 1.5},
    {"topology": "star", "agent_type": "robust", "shock_magnitude": 10.0,
     "shock_location": "random", "shock_node": 5, "seed": 42,
     "cascade_size": 0.2, "cascade_speed": 50.0, "recovery_time": 20.0,
     "systemic_risk": 0.3},
]


def test_aggregate_groups_correctly():
    agg = aggregate_by_condition(SAMPLE_RESULTS)
    # 3 unique conditions: ring/fragile/10/random, ring/fragile/10/hub, star/robust/10/random
    assert len(agg) == 3


def test_aggregate_mean():
    agg = aggregate_by_condition(SAMPLE_RESULTS)
    ring_random = [a for a in agg if a["topology"] == "ring" and a["shock_location"] == "random"][0]
    assert abs(ring_random["cascade_size_mean"] - 0.75) < 0.01
    assert ring_random["n_seeds"] == 2


def test_topology_ranking_order():
    agg = aggregate_by_condition(SAMPLE_RESULTS)
    ranked = topology_ranking(agg)
    # Ring should be riskier than star
    assert ranked[0]["topology"] == "ring"


def test_hub_vs_random():
    comp = hub_vs_random_comparison(aggregate_by_condition(SAMPLE_RESULTS))
    ring_comp = [c for c in comp if c["topology"] == "ring"][0]
    assert ring_comp["hub_cascade_size"] > ring_comp["random_cascade_size"]


def test_agent_type_comparison():
    comp = agent_type_comparison(aggregate_by_condition(SAMPLE_RESULTS))
    # Robust should be ranked first (lowest cascade)
    assert comp[0]["agent_type"] == "robust"
