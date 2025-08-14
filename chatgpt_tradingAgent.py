from typing import Dict, List, Tuple, Callable
import pandas as pd

Option = str  # e.g., "Buy", "Sell", "Hold"
Ground = str  # e.g., "Momentum Up", "Pullback Shallow"
WeightType = str  # "J" or "R"

class DualWeightAgent:
    def __init__(self,
                 options: List[Option],
                 grounds: List[Ground],
                 weights: Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]]):
        """Initialize the agent.
        
        weights is a dict keyed by (ground, option_a, option_b) for ordered comparison (A vs B)
        Each value is a dict with keys 'J' and 'R'.
        Example key: ("Momentum Up", "Buy", "Sell") → {"J": 0.8, "R": 0.3}
        """
        self.options = options
        self.grounds = grounds
        self.weights = weights

    def sum_weights(self,
                    option_a: Option,
                    option_b: Option,
                    grounds_active: Dict[Ground, float],
                    weight_type: WeightType) -> float:
        total = 0.0
        for g, activation in grounds_active.items():
            if activation == 0:
                continue
            w = self.weights.get((g, option_a, option_b), {"J": 0.0, "R": 0.0})
            total += w.get(weight_type, 0.0) * activation
        return total

    def permitted_vs(self,
                     option_a: Option,
                     option_b: Option,
                     grounds_active: Dict[Ground, float]) -> bool:
        # Sum of J for A in (A vs B)
        J_a = self.sum_weights(option_a, option_b, grounds_active, "J")
        # Sum of R for B in (B vs A)
        R_b = self.sum_weights(option_b, option_a, grounds_active, "R")
        return J_a > R_b

    def get_permitted_options(self, grounds_active: Dict[Ground, float]) -> List[Option]:
        permitted = []
        for a in self.options:
            all_pass = True
            for b in self.options:
                if a == b:
                    continue
                if not self.permitted_vs(a, b, grounds_active):
                    all_pass = False
                    break
            if all_pass:
                permitted.append(a)
        return permitted

    def choose_action(self, grounds_active: Dict[Ground, float],
                      tie_breaker: Callable[[List[Option], Dict[Ground, float]], Option] = None) -> Option:
        permitted = self.get_permitted_options(grounds_active)
        if not permitted:
            return "Hold"  # default fallback
        if len(permitted) == 1:
            return permitted[0]
        if tie_breaker:
            return tie_breaker(permitted, grounds_active)
        # default tie-breaker: first in list
        return permitted[0]

# ============================
# Example usage
# ============================
if __name__ == "__main__":
    options = ["Buy", "Sell", "Hold"]
    grounds = ["Momentum Up", "Pullback Shallow", "RSI Oversold", "High Volume", "Breakout"]

    # Example: assign dummy weights for all (ground, A, B) combos
    weights = {}
    for g in grounds:
        for A in options:
            for B in options:
                if A == B:
                    continue
                # Dummy scheme: J higher for Buy in bullish grounds, R small
                if "Momentum" in g and A == "Buy":
                    J = 0.8
                    R = 0.3
                else:
                    J = 0.2
                    R = 0.1
                weights[(g, A, B)] = {"J": J, "R": R}

    agent = DualWeightAgent(options, grounds, weights)

    # Example active grounds for today (activation strengths)
    grounds_active = {
        "Momentum Up": 1.0,
        "Pullback Shallow": 0.5,
        "RSI Oversold": 0.0,
        "High Volume": 0.0,
        "Breakout": 0.0,
    }

    permitted = agent.get_permitted_options(grounds_active)
    action = agent.choose_action(grounds_active)
    print("Permitted options:", permitted)
    print("Chosen action:", action)
