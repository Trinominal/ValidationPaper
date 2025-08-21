import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utility indicators
# -----------------------------
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, span1=12, span2=26, signal=9):
    ema1 = prices.ewm(span=span1, adjust=False).mean()
    ema2 = prices.ewm(span=span2, adjust=False).mean()
    macd_line = ema1 - ema2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line})

# -----------------------------
# Grounds (booleans)
# -----------------------------
def compute_boolean_grounds(prices, agent, day):
    ma_short = prices.rolling(20).mean().iloc[day]
    ma_long = prices.rolling(50).mean().iloc[day]
    rsi = compute_rsi(prices).iloc[day]
    macd_vals = compute_macd(prices)
    macd_line, signal_line = macd_vals.iloc[day]

    price = prices.iloc[day]

    return {
        "MA_crossover": ma_short > ma_long,
        "RSI_low": rsi < 30,
        "RSI_high": rsi > 70,
        "MACD_signal": macd_line > signal_line,
        "EquityGainSell": agent.position > 0 and price > agent.entry_price,
        "EquityGainBuy": agent.position == 0 or price < agent.entry_price,
    }

# -----------------------------
# Agent + Weights
# -----------------------------
class Agent:
    def __init__(self, name, style="balanced"):
        self.name = name
        self.style = style
        self.equity = 10000
        self.position = 0
        self.entry_price = 0
        self.history = []

    def reset(self):
        self.equity = 10000
        self.position = 0
        self.entry_price = 0
        self.history = []

    def base_weight(self):
        """Base scale by style."""
        return {"conservative": 0.5, "balanced": 1.0, "aggressive": 2.0}[self.style]

    def decide(self, day, prices, all_agents=None):
        grounds = compute_boolean_grounds(prices, self, day)

        options = ["Buy", "Sell", "Hold"]
        permitted = []
        for a in options:
            allowed = True
            for b in options:
                if a == b:
                    continue
                if not self.compare_options(a, b, grounds):
                    allowed = False
                    break
            if allowed:
                permitted.append(a)

        choice = "Hold" if not permitted else np.random.choice(permitted)
        self.execute(choice, prices.iloc[day])
        return choice

    def compare_options(self, a, b, grounds):
        jw_a, rw_b = 0, 0
        for g, active in grounds.items():
            if not active:
                continue
            # base weight
            base = self.base_weight()
            jw = base * np.random.uniform(0.8, 1.2)
            rw = base * np.random.uniform(0.8, 1.2)

            # context adjustments
            if g == "RSI_low" and a == "Buy":
                jw *= 1.5 if self.style == "aggressive" else 0.8
            if g == "RSI_high" and a == "Sell":
                jw *= 1.5 if self.style == "conservative" else 0.8
            if g == "EquityGainSell" and a == "Sell":
                jw *= 1.5
            if g == "EquityGainBuy" and a == "Buy":
                jw *= 1.3 if self.style == "aggressive" else 1.0

            jw_a += jw
            rw_b += rw
        return jw_a > rw_b

    def execute(self, choice, price):
        if choice == "Buy" and self.position == 0:
            qty = self.equity // price
            if qty > 0:
                self.position = qty
                self.entry_price = price
                self.equity -= qty * price
        elif choice == "Sell" and self.position > 0:
            self.equity += self.position * price
            self.position = 0
        self.history.append(self.equity + self.position * price)

# -----------------------------
# Collective Agent
# -----------------------------
class CollectiveAgent(Agent):
    def __init__(self, agents):
        super().__init__("Collective", style="collective")
        self.agents = agents
        self.grounds = grounds
        self.weights = self.aggregate_weights()

    def decide(self, day, prices, last_votes=None):
        # DO NOT call other agents' decide() here; they already decided this day.
        if not last_votes:
            choice = "Hold"
        else:
            # majority vote
            choice = max(set(last_votes), key=last_votes.count)
        self.execute(choice, prices.iloc[day])
        return choice
    
    def aggregate_weights(self):
        agg = {}
        for g in self.grounds:
            for a in ["Buy", "Sell", "Hold"]:
                for b in ["Buy", "Sell", "Hold"]:
                    if a == b:
                        continue
                    jw_sum = 0
                    rw_sum = 0
                    for agent in self.agents:
                        jw, rw = agent.weights[(g, a, b)]
                        jw_sum += jw
                        rw_sum += rw
                    agg[(g, a, b)] = (jw_sum, rw_sum)
        return agg


# -----------------------------
# Experiment
# -----------------------------
def run_experiment(prices, grounds):
    conservative = Agent("Conservative", style="conservative", grounds=grounds)
    balanced = Agent("Balanced", style="balanced", grounds=grounds)
    aggressive = Agent("Aggressive", style="aggressive", grounds=grounds)

    collective = CollectiveAgent([conservative, balanced, aggressive], grounds)
    collectiveMV = CollectiveAgent([conservative, balanced, aggressive])

    agents = [conservative, balanced, aggressive, collective, collectiveMV]

    decisions = {a.name: [] for a in agents}
    warmup = 50

    for day in range(warmup, len(prices)):
        day_votes = []
        for agent in agents:
            if isinstance(agent, CollectiveAgent):
                choice = agent.decide(day, prices, last_votes=day_votes)
            else:
                choice = agent.decide(day, prices)
                day_votes.append(choice)
            decisions[agent.name].append(choice)

    return agents, decisions


# -----------------------------
# Plotting
# -----------------------------
def plot_results(prices, agents, decisions, warmup=50, show=True):
    import numpy as np
    import matplotlib.pyplot as plt

    idx_full = prices.index
    idx = idx_full[warmup:]
    px  = prices.iloc[warmup:]  # explicit positional slice

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax0, ax1 = axes

    # --- Price + decisions ---
    ax0.plot(idx_full, prices.values, label="Price")
    for name, decs in decisions.items():
        decs = list(decs)  # should be length len(px)
        buy_pos  = np.array([i for i, d in enumerate(decs) if d == "Buy"], dtype=int)
        sell_pos = np.array([i for i, d in enumerate(decs) if d == "Sell"], dtype=int)

        if buy_pos.size:
            ax0.scatter(idx[buy_pos],  px.iloc[buy_pos],  marker="^", s=36, alpha=0.7, label=f"{name} Buy")
        if sell_pos.size:
            ax0.scatter(idx[sell_pos], px.iloc[sell_pos], marker="v", s=36, alpha=0.7, label=f"{name} Sell")

    ax0.set_title("Market Data and Decisions")
    ax0.legend()

    # --- Equity curves (align lengths robustly) ---
    for agent in agents:
        hist = np.asarray(agent.history, dtype=float)
        L = min(len(hist), len(idx))
        if L > 0:
            ax1.plot(idx[:L], hist[:L], label=agent.name)

    ax1.set_title("Equity Curves")
    ax1.legend()

    plt.tight_layout()
    plt.show() if show else None


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=300)
    prices = pd.Series(np.cumsum(np.random.randn(300)) + 100, index=dates)
    grounds =  compute_boolean_grounds(prices, self, day)

    agents, decisions = run_experiment(prices, grounds)
    plot_results(prices, agents, decisions)

    for a in agents:
        print(f"{a.name}: final equity = {a.history[-1]:.2f}")
