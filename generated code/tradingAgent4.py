import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Grounds (boolean indicators)
# -------------------------------
def compute_grounds(prices, day, portfolio, grounds):
    """Return boolean grounds for the given day and portfolio state"""
    gs = {g: False for g in grounds}

    if day < 20:  # not enough history for indicators
        return gs

    if "MA_crossover" in grounds: # moving average crossover
        window_short = 10
        window_long = 20
        ma_short = prices.iloc[day - window_short:day].mean()
        ma_long = prices.iloc[day - window_long:day].mean()
        ma_crossover = ma_short > ma_long
        gs["MA_crossover"] = ma_crossover # true means short term trend overtakes long term -> buy

    # RSI
    if "RSI_low" in grounds or "RSI_high" in grounds: # Relative Strength Index
        delta = prices.diff().iloc[1:day]
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
        if "RSI_low" in grounds:
            rsi_low = rsi < 30
            gs["RSI_low"] = rsi_low # true means oversold -> buy
        if "RSI_high" in grounds:
            rsi_high = rsi > 70
            gs["RSI_high"] = rsi_high # true means overbought -> sell

    # MACD (12 vs 26 EMA)
    if "MACD_positive" in grounds:  # moving average convergence/divergence
        ema12 = prices.ewm(span=12, adjust=False).mean().iloc[day]
        ema26 = prices.ewm(span=26, adjust=False).mean().iloc[day]
        macd_positive = ema12 > ema26
        gs["MACD_positive"] = macd_positive # true means upward momentum -> buy
        gs["MACD_negative"] = not macd_positive # true means downward momentum -> sell

    # Equity-based grounds
    current_price = prices.iloc[day]
    stock_value = portfolio["stock"] * current_price
    if "EquityGainSell" in grounds:
        equity_gain_sell = stock_value > portfolio["cost_basis"]
        gs["EquityGainSell"] = equity_gain_sell
    if "EquityGainBuy" in grounds:
        equity_gain_buy = portfolio["cash"] > current_price  # enough cash to buy at least 1
        gs["EquityGainBuy"] = equity_gain_buy

    return gs

# -------------------------------
# Agent and Collective Agent
# -------------------------------
class Agent:
    def __init__(self, name, style, grounds):
        self.name = name
        self.style = style
        self.grounds = grounds
        self.weights = self.generate_weights()
        self.portfolio = {"cash": 10000, "stock": 100, "cost_basis": 10000}
        self.equity = 10000 + 100*100
        self.history = []
        self.spent = []

    def generate_weights(self):
        weights = {}
        rng = np.random.default_rng()

        if self.style == "collective":
            return weights

        # Style biases
        style_bias = {
            "conservative": {"Buy": 1/6, "Sell": 1/6, "Hold": 4/6},
            "balanced": {"Buy": 1/3, "Sell": 1/3, "Hold": 1/3},
            "aggressive": {"Buy": 5/12, "Sell": 5/12, "Hold": 1/6},
            "collective": {"Buy": 0, "Sell": 0, "Hold": 0},
        }

        for g in self.grounds:
            for a in ["Buy", "Sell", "Hold"]:
                for b in ["Buy", "Sell", "Hold"]:
                    if a == b:
                        continue
                    # Randomized weights scaled by style
                    jw = rng.uniform(0.5, 2.0) * style_bias[self.style][a]
                    rw = rng.uniform(0.5, 2.0) * style_bias[self.style][b]
                    weights[(g, a, b)] = (jw, rw)
        return weights

    def decide(self, day, prices):
        grounds = compute_grounds(prices, day, self.portfolio, self.grounds)
        options = ["Buy", "Sell", "Hold"]
        permitted = {opt: True for opt in options}

        for a in options:
            for b in options:
                if a == b:
                    continue
                jw_sum = sum(
                    self.weights[(g, a, b)][0]
                    for g, val in grounds.items()
                    if val
                )
                rw_sum = sum(
                    self.weights[(g, b, a)][1]
                    for g, val in grounds.items()
                    if val
                )
                if jw_sum < rw_sum:
                    permitted[a] = False

        for opt in options:
            if permitted[opt]:
                return opt
        return "Hold"

    def update_portfolio(self, decision, price):
        if decision == "Buy" and self.portfolio["cash"] >= price:
            self.portfolio["stock"] += 1
            self.portfolio["cash"] -= price
            self.portfolio["cost_basis"] += price
        elif decision == "Sell" and self.portfolio["stock"] > 0:
            self.portfolio["cash"] += price
            self.portfolio["stock"] -= 1
            self.portfolio["cost_basis"] -= (
                self.portfolio["cost_basis"] / (self.portfolio["stock"] + 1)
            )
        # Update equity
        self.equity = ( 
            self.portfolio["cash"] + self.portfolio["stock"] * price
        )
        self.history.append(self.equity)

        self.spent.append(self.portfolio["cost_basis"])


class CollectiveAgent(Agent):
    def __init__(self, agents, grounds):
        super().__init__("Collective", style="collective", grounds=grounds)
        self.agents = agents
        self.weights = self.aggregate_weights(grounds)

    def aggregate_weights(self, grounds):
        agg = {}
        for g in grounds:
            for a in ["Buy", "Sell", "Hold"]:
                for b in ["Buy", "Sell", "Hold"]:
                    if a == b:
                        continue
                    jw_sum, rw_sum = 0.0, 0.0
                    for agent in self.agents:
                        jw, rw = agent.weights[(g, a, b)]
                        jw_sum += jw
                        rw_sum += rw
                    agg[(g, a, b)] = (jw_sum, rw_sum)
        return agg

# Maybe add that if no one would permit an option then that option is not considered. Not sure maybe given the reasons the option should be permitted?

# -------------------------------
# Experiment runner
# -------------------------------
def run_experiment(prices, grounds):
    conservative = Agent("Conservative", style="conservative", grounds=grounds)
    balanced = Agent("Balanced", style="balanced", grounds=grounds)
    aggressive = Agent("Aggressive", style="aggressive", grounds=grounds)
    collective = CollectiveAgent([conservative, balanced, aggressive], grounds)

    agents = [conservative, balanced, aggressive, collective]
    decisions = {a.name: [] for a in agents}
    warmup = 50

    for day in range(warmup, len(prices)):
        for agent in agents:
            choice = agent.decide(day, prices)
            agent.update_portfolio(choice, prices.iloc[day])
            decisions[agent.name].append(choice)

    return agents, decisions

# -------------------------------
# Plotting
# -------------------------------
def plot_results(prices, agents, decisions, warmup=50, show=True):
    # Build a clean x-axis (dates) after warmup and the price Series aligned to it
    idx_full = prices.index
    idx = idx_full[warmup:]           # x-axis for decisions/equity
    px  = prices.iloc[warmup:]        # prices aligned to idx

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax0, ax1, ax2 = axes

    # --- Price + decisions ---
    ax0.plot(idx_full, prices.values, label="Price")

    for name, decs in decisions.items():
        decs = list(decs)  # one decision per post-warmup day
        # positions relative to the post-warmup window
        buy_pos  = np.array([i for i, d in enumerate(decs) if d == "Buy"], dtype=int)
        sell_pos = np.array([i for i, d in enumerate(decs) if d == "Sell"], dtype=int)

        # Use idx[...] for x (dates) and px.iloc[...] for y (positional)
        if buy_pos.size:
            ax0.scatter(idx[buy_pos],  px.iloc[buy_pos],  marker="^", s=36, alpha=0.7, label=f"{name} Buy")
        if sell_pos.size:
            ax0.scatter(idx[sell_pos], px.iloc[sell_pos], marker="v", s=36, alpha=0.7, label=f"{name} Sell")

    ax0.set_title("Market Data and Decisions")
    ax0.legend()

    # --- Equity curves (length-safe) ---
    for agent in agents:
        hist = np.asarray(agent.history, dtype=float)
        # history should be post-warmup length; still guard for mismatches
        L = min(len(hist), len(idx))
        if L > 0:
            ax1.plot(idx[:L], hist[:L], label=agent.name)

    ax1.set_title("Equity Curves")
    ax1.legend()

    # --- Equity - Price curve ---
    for agent in agents:
        hist = np.asarray(agent.history, dtype=float)
        spent = np.asarray(agent.spent, dtype=float)
        # history should be post-warmup length; still guard for mismatches
        L = min(len(hist), len(idx))
        if L > 0:
            # not really proper to divide by 100 this should be the stock number at that time. I divided by 100 since that is what they start with. maybe should also be + money
            # how much value was gaind irrespective of the index. 
            # at start agent has 10000 + value(100 stock) we want to track the value gained irrespective of the passive change in value from the index.
            ax2.plot(idx[:L], hist[:L]-spent[:L], label=agent.name)
    # ax2.plot(idx_full, prices.values, label="Price")
    ax2.set_title("Equity - expenses")
    ax2.legend()

    plt.savefig("trading_agents_experiment.png", dpi=150)
    plt.tight_layout()
    if show:
        plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n_days = 50+356*2
    prices = pd.Series(np.cumsum(np.random.randn(n_days)) + 100)

    grounds = [
        "MA_crossover", "RSI_low", "RSI_high", "MACD_positive", "MACD_negative"
        , "EquityGainSell", "EquityGainBuy"
    ]

    agents, decisions = run_experiment(prices, grounds)
    plot_results(prices, agents, decisions)

    # Summary
    print("\nFinal equities:")
    for agent in agents:
        print(f"{agent.name}: {agent.equity:.2f}")
