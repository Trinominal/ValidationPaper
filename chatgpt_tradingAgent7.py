from typing import Dict, List, Tuple, Callable
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# TA indicators
def moving_average(series: pd.Series, window: int) -> float:
    return series.rolling(window=window).mean().iloc[-1]

def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs.iloc[-1]))

def stochastic_oscillator(series: pd.Series, k_window: int = 14) -> float:
    low_min = series.rolling(window=k_window).min().iloc[-1]
    high_max = series.rolling(window=k_window).max().iloc[-1]
    return 100 * ((series.iloc[-1] - low_min) / (high_max - low_min))

def macd(series: pd.Series) -> float:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return (ema12 - ema26).iloc[-1]

def equity_gain_if_sell(price: float, entry_price: float) -> float:
    if entry_price == 0:
        return 0.0
    return (price - entry_price) / entry_price

Option = str
Ground = str
WeightType = str

class DualWeightAgent:
    def __init__(self, options: List[Option], grounds: List[Ground], weights: Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]], name: str = "Agent", starting_equity: float = 10000.0):
        self.options = options
        self.grounds = grounds
        self.weights = weights
        self.name = name
        self.equity = starting_equity
        self.position = 0
        self.entry_price = 0.0
        self.equity_history = []
        self.actions_history = []

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
        return [a for a in self.options if all(self.permitted_vs(a, b, grounds_active) for b in self.options if a != b)]

    def choose_action(self, grounds_active: Dict[Ground, float]) -> Option:
        permitted = self.get_permitted_options(grounds_active)
        if not permitted:
            action = random.choice(self.options)
        else:
            action = random.choice(permitted)
        self.actions_history.append(action)
        return action

    def update_equity(self, action: Option, price: float):
        if action == "Buy" and self.position == 0:
            self.position = self.equity / price
            self.entry_price = price
            self.equity = 0.0
        elif action == "Sell" and self.position > 0:
            self.equity = self.position * price
            self.position = 0
            self.entry_price = 0.0
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

def run_experiment(prices: pd.Series, agents: List[DualWeightAgent]):
    for i in range(len(prices)):
        price = prices.iloc[i]
        grounds_active = {
            "MA20": moving_average(prices[:i+1], 20) if i >= 19 else 0,
            "RSI14": rsi(prices[:i+1], 14) if i >= 13 else 0,
            "Stoch14": stochastic_oscillator(prices[:i+1], 14) if i >= 13 else 0,
            "MACD": macd(prices[:i+1]) if i >= 26 else 0,
            "EquityGainSell": equity_gain_if_sell(price, agents[0].entry_price)
        }
        for agent in agents:
            action = agent.choose_action(grounds_active)
            agent.update_equity(action, price)

def plot_equity(agents: List[DualWeightAgent]):
    plt.figure(figsize=(10, 5))
    for agent in agents:
        plt.plot(agent.equity_history, label=agent.name)
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.title("Equity Over Time")
    plt.legend()
    plt.show()

def plot_market(prices: pd.Series):
    plt.figure(figsize=(10, 4))
    plt.plot(prices, label="Closing Price")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Market Closing Prices")
    plt.legend()
    plt.show()

def plot_action_bars(agents: List[DualWeightAgent]):
    collective_actions = agents[-1].actions_history
    fig, axs = plt.subplots(1, len(agents)-1, figsize=(15, 4), sharey=True)
    for idx, agent in enumerate(agents[:-1]):
        match_count = sum(a == c for a, c in zip(agent.actions_history, collective_actions))
        total = len(agent.actions_history)
        action_counts = pd.Series(agent.actions_history).value_counts()
        axs[idx].bar(action_counts.index, action_counts.values, color='skyblue')
        axs[idx].set_title(f"{agent.name}\nMatches with Collective: {match_count}/{total}")
        axs[idx].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    days = 200
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, days)))
    options = ["Buy", "Sell", "Hold"]
    grounds = ["MA20", "RSI14", "Stoch14", "MACD", "EquityGainSell"]

    conservative = DualWeightAgent(options, grounds, generate_weights(options, grounds, "conservative"), name="Conservative")
    balanced = DualWeightAgent(options, grounds, generate_weights(options, grounds, "balanced"), name="Balanced")
    aggressive = DualWeightAgent(options, grounds, generate_weights(options, grounds, "aggressive"), name="Aggressive")
    collective = combine_agents([conservative, balanced, aggressive])

    agents = [conservative, balanced, aggressive, collective]
    run_experiment(prices, agents)

    plot_market(prices)
    plot_equity(agents)
    plot_action_bars(agents)
