"""Agent definitions for the reputation network simulation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Agent:
    """An agent in the reputation marketplace."""

    agent_id: int
    true_quality: float  # Ground-truth quality in [0, 1]
    is_sybil: bool = False
    sybil_controller: int = -1  # ID of the controller if Sybil
    account_age: int = 0  # Number of rounds since creation
    ratings_given: List[float] = field(default_factory=list)
    ratings_received: List[float] = field(default_factory=list)
    transaction_partners: List[int] = field(default_factory=list)
    reputation_score: float = 0.5  # Current reputation


def create_honest_agents(n: int, rng) -> List[Agent]:
    """Create n honest agents with random true qualities.

    Args:
        n: Number of honest agents.
        rng: Random generator with a ``uniform(low, high)`` method.

    Returns:
        List of honest Agent objects with IDs 0..n-1.
    """
    return [
        Agent(agent_id=i, true_quality=float(rng.uniform(0.2, 0.9)))
        for i in range(n)
    ]


def create_sybil_agents(
    k: int, start_id: int, controller_id: int
) -> List[Agent]:
    """Create k Sybil agents controlled by a single attacker.

    All Sybil agents share controller_id and have low true quality (0.1)
    but will try to manipulate the system.

    Args:
        k: Number of Sybil identities to create.
        start_id: Starting agent ID for Sybil agents.
        controller_id: ID of the controlling entity.

    Returns:
        List of Sybil Agent objects.
    """
    return [
        Agent(
            agent_id=start_id + i,
            true_quality=0.1,
            is_sybil=True,
            sybil_controller=controller_id,
        )
        for i in range(k)
    ]
