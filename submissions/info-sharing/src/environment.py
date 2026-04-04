"""Information sharing environment: hidden state, partial observations, payoffs."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the information-sharing environment.

    Parameters
    ----------
    n_agents : int
        Number of agents (default 4).
    state_dim : int
        Dimensionality of the hidden state vector (default 8).
    obs_fraction : float
        Fraction of state dimensions each agent privately observes (default 0.25).
    competition : float
        How much others' improved decisions hurt you. Range [0, 1].
        0 = pure cooperation, 1 = zero-sum.
    complementarity : float
        How much others' shared info improves your decision. Range [0, 1].
        0 = others' info is useless, 1 = maximally helpful.
    noise_std : float
        Observation noise standard deviation (default 0.1).
    """

    n_agents: int = 4
    state_dim: int = 8
    obs_fraction: float = 0.25
    competition: float = 0.5
    complementarity: float = 0.5
    noise_std: float = 0.1


class Environment:
    """Simulates a hidden-state environment with partial observations.

    Each round:
    1. A hidden state vector theta ~ N(0, I) is drawn.
    2. Each agent observes a noisy subset of dimensions.
    3. Agents choose disclosure levels (0-1).
    4. Shared info is pooled; agents make decisions (estimate theta).
    5. Payoffs = own accuracy - competition * others' accuracy gain from sharing.
    """

    def __init__(self, config: EnvConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        c = config

        # Pre-compute which dimensions each agent observes (fixed across rounds)
        n_obs = max(1, int(c.state_dim * c.obs_fraction))
        self.obs_masks: list[np.ndarray] = []
        for i in range(c.n_agents):
            # Deterministic assignment: agent i observes dims [i*n_obs : (i+1)*n_obs] mod state_dim
            dims = np.array([(i * n_obs + d) % c.state_dim for d in range(n_obs)])
            mask = np.zeros(c.state_dim, dtype=bool)
            mask[dims] = True
            self.obs_masks.append(mask)

    def step(
        self, disclosure_levels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run one round of the information-sharing game.

        Parameters
        ----------
        disclosure_levels : ndarray of shape (n_agents,)
            Each agent's disclosure level in [0, 1].

        Returns
        -------
        payoffs : ndarray of shape (n_agents,)
            Each agent's payoff this round.
        sharing_rates : ndarray of shape (n_agents,)
            Echo of disclosure_levels (for logging convenience).
        decision_errors : ndarray of shape (n_agents,)
            Each agent's squared error in estimating theta.
        group_welfare : float
            Sum of all payoffs.
        """
        c = self.config
        n = c.n_agents

        # 1. Draw hidden state
        theta = self.rng.standard_normal(c.state_dim)

        # 2. Each agent observes its subset with noise
        private_obs = np.zeros((n, c.state_dim))
        has_info = np.zeros((n, c.state_dim), dtype=bool)
        for i in range(n):
            mask = self.obs_masks[i]
            noise = self.rng.normal(0, c.noise_std, size=c.state_dim)
            private_obs[i] = theta + noise
            private_obs[i, ~mask] = 0.0  # no info on unobserved dims
            has_info[i] = mask.copy()

        # 3. Information sharing: agent i discloses disclosure_levels[i] fraction
        #    of its private observation (with noise proportional to 1 - disclosure)
        shared_pool = np.zeros((n, c.state_dim))
        shared_mask = np.zeros((n, c.state_dim), dtype=bool)
        for i in range(n):
            dl = float(disclosure_levels[i])
            mask = self.obs_masks[i]
            # Agent discloses: its observation + extra noise scaled by (1-dl)
            disclosure_noise = self.rng.normal(
                0, c.noise_std * (1.0 - dl + 1e-8), size=c.state_dim
            )
            shared_pool[i] = private_obs[i] + disclosure_noise * (1.0 - dl)
            shared_pool[i, ~mask] = 0.0
            # Only truly shared if dl > 0
            shared_mask[i] = mask & (dl > 0.01)

        # 4. Each agent combines own info with others' shared info
        #    Complementarity controls signal quality: low complementarity adds
        #    extra noise to received info (others' info is less useful).
        estimates = np.zeros((n, c.state_dim))
        for i in range(n):
            # Start with own private obs
            est = private_obs[i].copy()
            count = has_info[i].astype(float)

            for j in range(n):
                if j == i:
                    continue
                useful = shared_mask[j] & (~has_info[i])  # new dims from j
                if not np.any(useful):
                    continue
                # Complementarity noise: low complementarity => large extra noise
                compl_noise = self.rng.normal(
                    0,
                    c.noise_std * (1.0 - c.complementarity) * 3.0,
                    size=c.state_dim,
                )
                received = shared_pool[j] + compl_noise
                est[useful] += received[useful]
                count[useful] += 1.0

            # Average where we have multiple sources
            nonzero = count > 0
            est[nonzero] /= count[nonzero]
            estimates[i] = est

        # 5. Compute decision quality (negative MSE — higher is better)
        errors = np.array(
            [np.mean((estimates[i] - theta) ** 2) for i in range(n)]
        )

        # 6. Compute payoffs: own quality minus competitive cost
        #    Competitive cost = competition * (mean improvement of others due to sharing)
        baseline_errors = np.array(
            [np.mean((private_obs[i] - theta) ** 2 * has_info[i])
             + np.mean(theta ** 2 * (~has_info[i]))
             for i in range(n)]
        )
        improvements = np.maximum(baseline_errors - errors, 0.0)

        payoffs = np.zeros(n)
        for i in range(n):
            own_quality = -errors[i]
            others_gain = np.mean(
                [improvements[j] for j in range(n) if j != i]
            )
            payoffs[i] = own_quality - c.competition * others_gain

        group_welfare = float(np.sum(payoffs))
        return payoffs, disclosure_levels.copy(), errors, group_welfare
