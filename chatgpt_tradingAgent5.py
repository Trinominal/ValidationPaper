from typing import Dict, List, Tuple, Callable
import random
import matplotlib.pyplot as plt

Option = str
Ground = str
WeightType = str

class DualWeightAgent:
    def __init__(self,
                 options: List[Option],
                 grounds: List[Ground],
                 weights: Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]],
                 name: str = "Agent",
                 starting_equity: float = 10000.0):
        self.options = options
        self.grounds = grounds
        self.weights = weights
        self.name = name
        self.equity = starting_equity
        self.position = 0
        self.equity_history = []

    def sum_weights(self, option_a: Option, option_b: Option, grounds_active: Dict[Ground, float], weight_type: WeightType) -> float:
        return sum(
            self.weights.get((g, option_a, option_b), {"J": 0.0, "R": 0.0}).get(weight_type, 0.0) * activation
            for g, activation in grounds_active.items() if activation != 0
        )

    def permitted_vs(self, option_a: Option, option_b: Option, grounds_active: Dict[Ground, float]) -> bool:
        J_a = self.sum_weights(option_a, option_b, grounds_active, "J")
        R_b = self.sum_weights(option_b, option_a, grounds_active, "R")
        return J_a > R_b

    def get_permitted_options(self, grounds_active: Dict[Ground, float]) -> List[Option]:
        permitted = []
        for a in self.options:
            if all(self.permitted_vs(a, b, grounds_active) for b in self.options if a != b):
                permitted.append(a)
        return permitted

    def choose_action(self, grounds_active: Dict[Ground, float], tie_breaker: Callable[[List[Option], Dict[Ground, float]], Option] = None) -> Option:
        permitted = self.get_permitted_options(grounds_active)
        if not permitted:
            return random.choice(self.options)
        if len(permitted) == 1:
            return permitted[0]
        return tie_breaker(permitted, grounds_active) if tie_breaker else random.choice(permitted)

    def update_equity(self, action: Option, price: float):
        if action == "Buy" and self.position == 0:
            self.position = self.equity / price
            self.equity = 0.0
        elif action == "Sell" and self.position > 0:
            self.equity = self.position * price
            self.position = 0
        if self.position > 0:
            self.equity = self.position * price
        self.equity_history.append(self.equity)


def generate_weights(options: List[Option], grounds: List[Ground], profile: str) -> Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]]:
    weights = {}
    for g in grounds:
        for A in options:
            for B in options:
                if A == B:
                    continue
                base_J, base_R = 0.5, 0.5
                if profile == "aggressive":
                    if A == "Buy":
                        base_J += 0.3
                        base_R -= 0.2
                elif profile == "conservative":
                    if A == "Hold":
                        base_J += 0.2
                    if A in ["Buy", "Sell"]:
                        base_J += 0.1
                        base_R += 0.1
                elif profile == "balanced":
                    if A in ["Buy", "Sell"]:
                        base_J += 0.15
                        base_R -= 0.05
                weights[(g, A, B)] = {"J": max(0, base_J), "R": max(0, base_R)}
    return weights


def combine_agents(agents: List[DualWeightAgent]) -> DualWeightAgent:
    options, grounds = agents[0].options, agents[0].grounds
    combined_weights = {}
    for g in grounds:
        for A in options:
            for B in options:
                if A == B:
                    continue
                J_sum = sum(agent.weights[(g, A, B)]["J"] for agent in agents)
                R_sum = sum(agent.weights[(g, A, B)]["R"] for agent in agents)
                combined_weights[(g, A, B)] = {"J": J_sum, "R": R_sum}
    return DualWeightAgent(options, grounds, combined_weights, name="Collective")


def run_experiment(days: int, agents: List[DualWeightAgent], grounds: List[Ground]):
    price = 100.0
    for day in range(days):
        price *= (1 + random.uniform(-0.02, 0.02))
        grounds_active = {g: round(random.random(), 2) for g in grounds}
        for agent in agents:
            action = agent.choose_action(grounds_active)
            agent.update_equity(action, price)

def plot_equity(agents: List[DualWeightAgent]):
    for agent in agents:
        plt.plot(agent.equity_history, label=agent.name)
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.title("Equity Over Time")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    options = ["Buy", "Sell", "Hold"]
    grounds = ["Momentum Up", "Pullback Shallow", "RSI Oversold", "High Volume", "Breakout"]

    conservative = DualWeightAgent(options, grounds, generate_weights(options, grounds, "conservative"), name="Conservative")
    balanced = DualWeightAgent(options, grounds, generate_weights(options, grounds, "balanced"), name="Balanced")
    aggressive = DualWeightAgent(options, grounds, generate_weights(options, grounds, "aggressive"), name="Aggressive")
    collective = combine_agents([conservative, balanced, aggressive])

    agents = [conservative, balanced, aggressive, collective]
    run_experiment(100, agents, grounds)
    plot_equity(agents)
