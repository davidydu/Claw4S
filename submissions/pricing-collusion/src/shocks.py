# src/shocks.py
"""Market shock injection for testing collusion robustness."""


class CostShock:
    """Increases one seller's marginal cost at a specified round."""

    def __init__(self, seller_id, cost_increase, trigger_round, total_rounds):
        self.seller_id = seller_id
        self.cost_increase = cost_increase
        self.trigger_round = trigger_round
        self.total_rounds = total_rounds
        self._triggered = False

    def should_trigger(self, current_round):
        if self._triggered:
            return False
        return current_round == self.trigger_round

    def apply(self, market):
        market.costs[self.seller_id] += self.cost_increase
        self._triggered = True


class DemandShock:
    """Shifts the market's price sensitivity (alpha) at a specified round."""

    def __init__(self, alpha_shift, trigger_round, total_rounds):
        self.alpha_shift = alpha_shift
        self.trigger_round = trigger_round
        self.total_rounds = total_rounds
        self._triggered = False

    def should_trigger(self, current_round):
        if self._triggered:
            return False
        return current_round == self.trigger_round

    def apply(self, market):
        market.alpha += self.alpha_shift
        self._triggered = True
