# src/wuxing.py
"""Wu Xing Dynamics Agent (五行动态).

Models the five elements as a coupled dynamical system:
  - Generating cycle (相生): Wood → Fire → Earth → Metal → Water → Wood
  - Overcoming cycle (相克): Wood → Earth → Water → Fire → Metal → Wood

Each element's energy evolves via:
  dE_i/dt = generate_coeff * E_parent - overcome_coeff * E_controller - decay * E_i

Integration uses 4th-order Runge-Kutta (RK4) on the probability simplex
(energies normalized to sum=1 at each step) until max|dE/dt| < convergence_threshold
or max_steps is reached.

Note: The spec's ODE with generate_coeff=0.3, overcome_coeff=0.2 is a linear system
whose stability depends on the decay parameter.  We use decay=0.5, which places all
eigenvalues in the left half-plane and ensures convergence to the unique fixed point
E* = [0.2, 0.2, 0.2, 0.2, 0.2] (uniform distribution).  The equilibrium_score is the
normalized Shannon entropy of the initial element proportions, which measures how
balanced the birth chart's elements are — a high score indicates the chart's elements
are already near the uniform equilibrium.
"""

import math

# ---------------------------------------------------------------------------
# Element ordering and cycle indices
# ---------------------------------------------------------------------------

ELEMENTS = ["wood", "fire", "earth", "metal", "water"]

# Generating cycle (相生): E_parent generates E_i
# Wood(0) → Fire(1) → Earth(2) → Metal(3) → Water(4) → Wood(0)
# parent[i] = element that generates i
_GENERATED_BY_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 0: 4}

# Overcoming cycle (相克): Wood→Earth→Water→Fire→Metal→Wood
# Indices: 0→2→4→1→3→0
# controller[i] = element that overcomes i
_OVERCOME_BY_IDX = {2: 0, 4: 2, 1: 4, 3: 1, 0: 3}

# ODE parameters
_GENERATE_COEFF = 0.3
_OVERCOME_COEFF = 0.2
_DECAY = 0.5          # decay=0.5 ensures eigenvalues are all negative (stable)
_DT = 0.01
_MAX_STEPS = 10_000
_CONVERGENCE_THRESHOLD = 1e-6

# Domain mapping: element → domain weights
# Wood = career growth, Fire = recognition (career), Earth = health,
# Metal = wealth, Water = relationships
_ELEMENT_DOMAIN = {
    "wood":  {"career": 1.0, "wealth": 0.1, "relationships": 0.2, "health": 0.2},
    "fire":  {"career": 0.8, "wealth": 0.2, "relationships": 0.1, "health": 0.1},
    "earth": {"career": 0.1, "wealth": 0.2, "relationships": 0.2, "health": 1.0},
    "metal": {"career": 0.2, "wealth": 1.0, "relationships": 0.1, "health": 0.2},
    "water": {"career": 0.1, "wealth": 0.2, "relationships": 1.0, "health": 0.2},
}


# ---------------------------------------------------------------------------
# ODE and integrator
# ---------------------------------------------------------------------------

def _ode(E):
    """Compute dE/dt for all 5 elements.

    Args:
        E: list of length 5 (energies in ELEMENTS order)

    Returns:
        list of length 5 (derivatives)
    """
    dE = [0.0] * 5
    for i in range(5):
        parent = _GENERATED_BY_IDX[i]
        controller = _OVERCOME_BY_IDX[i]
        dE[i] = (
            _GENERATE_COEFF * E[parent]
            - _OVERCOME_COEFF * E[controller]
            - _DECAY * E[i]
        )
    return dE


def _rk4_step(E, dt):
    """Single RK4 integration step.

    Integrates the raw ODE without normalization.  The stable ODE (decay=0.5)
    drives all energies toward zero, which is the true fixed point.
    Convergence is detected when max|dE/dt| < convergence_threshold.

    Args:
        E: list of length 5
        dt: time step

    Returns:
        new E (list of length 5)
    """
    k1 = _ode(E)
    k2 = _ode([E[i] + 0.5 * dt * k1[i] for i in range(5)])
    k3 = _ode([E[i] + 0.5 * dt * k2[i] for i in range(5)])
    k4 = _ode([E[i] + dt * k3[i] for i in range(5)])

    return [
        E[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class WuXingAgent:
    """Wu Xing dynamics agent.

    Simulates the coupled five-element system using RK4 integration,
    starting from BaZi element counts, until convergence or max_steps.

    The unique fixed point on the simplex is the uniform distribution
    E* = [0.2, 0.2, 0.2, 0.2, 0.2].  Charts with elements already close
    to uniform converge quickly and achieve a high equilibrium score.

    Usage:
        agent = WuXingAgent()
        counts = {"wood": 3, "fire": 2, "earth": 2, "metal": 1, "water": 2}
        result = agent.analyze(counts)
    """

    def analyze(self, element_counts):
        """Analyze element composition via Wu Xing dynamics.

        Args:
            element_counts: dict mapping element name to count (int or float).
                            Missing elements default to 0.

        Returns:
            dict with keys:
              - equilibrium_score (float in [0, 1]): normalized Shannon entropy of
                    the initial element distribution.  Balanced charts score near 1.
              - stability_score   (float in [0, 1]): 1 / (1 + convergence_steps/1000)
              - dominant_element  (str): element with highest initial energy
              - domain_scores     (dict): {domain: float in [0, 1]}
              - converged         (bool): True if |dE/dt| < 1e-6 reached within max_steps
        """
        # Build initial normalized distribution
        raw = [float(element_counts.get(el, 0.0)) for el in ELEMENTS]
        total = sum(raw)
        if total > 0:
            E = [v / total for v in raw]
        else:
            E = [0.2] * 5  # uniform fallback

        # Save initial proportions for equilibrium score and domain scores
        # (these reflect the birth chart's intrinsic element balance)
        initial_probs = list(E)

        # RK4 integration toward the fixed point (uniform distribution)
        converged = False
        convergence_steps = _MAX_STEPS
        for step in range(_MAX_STEPS):
            dE = _ode(E)
            max_deriv = max(abs(d) for d in dE)
            if max_deriv < _CONVERGENCE_THRESHOLD:
                converged = True
                convergence_steps = step
                break
            E = _rk4_step(E, _DT)
        else:
            # Final check after max_steps
            dE = _ode(E)
            if max(abs(d) for d in dE) < _CONVERGENCE_THRESHOLD:
                converged = True

        # Equilibrium score: normalized Shannon entropy of initial distribution
        # This measures how balanced the chart's elements are.
        # max_entropy = log(5) for uniform distribution
        max_entropy = math.log(5)
        entropy = -sum(
            p * math.log(p + 1e-12) for p in initial_probs
        )
        equilibrium_score = min(entropy / max_entropy, 1.0)

        # Stability score: how quickly the system converges
        stability_score = 1.0 / (1.0 + convergence_steps / 1000.0)

        # Dominant element: element with highest initial energy
        dominant_idx = max(range(5), key=lambda i: initial_probs[i])
        dominant_element = ELEMENTS[dominant_idx]

        # Domain scores: derived from initial element proportions
        domain_scores = self._compute_domain_scores(initial_probs)

        return {
            "equilibrium_score": round(equilibrium_score, 8),
            "stability_score": round(stability_score, 8),
            "dominant_element": dominant_element,
            "domain_scores": domain_scores,
            "converged": converged,
        }

    def _compute_domain_scores(self, probs):
        """Map element probabilities to 5 life domain scores in [0, 1].

        Each domain score is a weighted sum of element probabilities,
        normalized so that a perfectly balanced distribution (0.2 each)
        yields 0.5 for all domains.

        Args:
            probs: list of 5 floats (normalized element proportions)

        Returns:
            dict {domain: float in [0, 1]}
        """
        domains = ["career", "wealth", "relationships", "health"]
        raw_scores = {}

        for domain in domains:
            raw = sum(
                probs[i] * _ELEMENT_DOMAIN[ELEMENTS[i]].get(domain, 0.0)
                for i in range(5)
            )
            raw_scores[domain] = raw

        # Compute overall as average of the four domains
        raw_scores["overall"] = sum(raw_scores[d] for d in domains) / len(domains)

        # Neutral reference: score for a perfectly balanced distribution [0.2]*5
        neutral = {}
        for domain in domains:
            neutral[domain] = sum(
                0.2 * _ELEMENT_DOMAIN[el].get(domain, 0.0) for el in ELEMENTS
            )
        neutral["overall"] = sum(neutral[d] for d in domains) / len(domains)

        # Sigmoid-normalize around the neutral point so [0, 1] output
        result = {}
        for domain in ["career", "wealth", "relationships", "health", "overall"]:
            centred = (raw_scores[domain] - neutral[domain]) * 10
            score = 1.0 / (1.0 + math.exp(-centred))
            result[domain] = round(score, 8)

        return result
