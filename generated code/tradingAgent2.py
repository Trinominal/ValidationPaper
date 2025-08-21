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
# Utility functions for additional indicators
# -----------------------------
def compute_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(d_window).mean()
    return k_percent, d_percent

# -----------------------------
# Grounds (booleans)
# -----------------------------
def compute_boolean_grounds(prices, agent, day):
    # Moving averages
    ma_short = prices.rolling(20).mean().iloc[day]
    ma_long = prices.rolling(50).mean().iloc[day]
    ma_very_short = prices.rolling(10).mean().iloc[day]
    
    # RSI with multiple thresholds
    rsi = compute_rsi(prices).iloc[day]
    
    # MACD
    macd_vals = compute_macd(prices)
    macd_line, signal_line = macd_vals.iloc[day]
    macd_histogram = macd_line - signal_line
    
    # Bollinger Bands
    upper_band, lower_band = compute_bollinger_bands(prices)
    bb_upper = upper_band.iloc[day]
    bb_lower = lower_band.iloc[day]
    
    # Price momentum
    price = prices.iloc[day]
    price_5_ago = prices.iloc[day-5] if day >= 5 else price
    price_10_ago = prices.iloc[day-10] if day >= 10 else price
    
    # Volatility
    volatility = prices.rolling(20).std().iloc[day]
    avg_volatility = prices.rolling(50).std().mean()

    return {
        # Moving Average signals
        "MA_crossover": ma_short > ma_long,
        "MA_strong_uptrend": ma_very_short > ma_short > ma_long,
        "MA_strong_downtrend": ma_very_short < ma_short < ma_long,
        
        # RSI with multiple levels
        "RSI_oversold": rsi < 30,
        "RSI_very_oversold": rsi < 20,
        "RSI_oversold_mild": 30 <= rsi < 40,
        "RSI_overbought": rsi > 70,
        "RSI_very_overbought": rsi > 80,
        "RSI_overbought_mild": 60 < rsi <= 70,
        "RSI_neutral": 40 <= rsi <= 60,
        
        # MACD signals
        "MACD_bullish": macd_line > signal_line,
        "MACD_bearish": macd_line < signal_line,
        "MACD_momentum_up": macd_histogram > 0,
        "MACD_momentum_down": macd_histogram < 0,
        
        # Bollinger Bands
        "BB_squeeze": (bb_upper - bb_lower) < (bb_upper - bb_lower) * 0.8,  # Simplified squeeze
        "BB_breakout_up": price > bb_upper,
        "BB_breakout_down": price < bb_lower,
        "BB_middle_range": bb_lower < price < bb_upper,
        
        # Momentum signals
        "Price_momentum_up": price > price_5_ago * 1.02,  # 2% increase over 5 days
        "Price_momentum_down": price < price_5_ago * 0.98,  # 2% decrease over 5 days
        "Price_strong_momentum_up": price > price_10_ago * 1.05,  # 5% increase over 10 days
        "Price_strong_momentum_down": price < price_10_ago * 0.95,  # 5% decrease over 10 days
        
        # Volatility signals
        "High_volatility": volatility > avg_volatility * 1.5,
        "Low_volatility": volatility < avg_volatility * 0.7,
        
        # Position-based signals
        "EquityGainSell": agent.position > 0 and price > agent.entry_price,
        "EquityGainBuy": agent.position == 0 or price < agent.entry_price,
        "EquityLoss": agent.position > 0 and price < agent.entry_price * 0.95,  # 5% loss
        "EquityBigGain": agent.position > 0 and price > agent.entry_price * 1.1,  # 10% gain
    }

# -----------------------------
# Agent + Weights
# -----------------------------
class Agent:
    def __init__(self, name, style="balanced", options, grounds):
        self.name = name
        self.style = style
        self.equity = 10000
        self.position = 0
        self.entry_price = 0
        self.history = []
        self.last_weights = {option1: {option2: {ground: (0,0)
                           for ground in grounds}
                    for option2 in options}
                for option1 in options}

    def reset(self):
        self.equity = 10000
        self.position = 0
        self.entry_price = 0
        self.history = []

    def base_weight(self):
        """Base scale by style."""
        return {"conservative": 0.5, "balanced": 1.0, "aggressive": 2.0}[self.style]
    
    def get_option_weights(self, grounds):
        """return weights for this context of grounds"""
        # TODO agent_weight[option] = 
        return 0

    def decide(self, day, prices, all_agents=None):
        grounds = compute_boolean_grounds(prices, self, day)
        
        # Get weights for all options
        # option_weights = self.get_option_weights(grounds)

        options = ["Buy", "Sell", "Hold"]
        permitted = []
        for a in options:
            allowed = True
            for b in options:
                if a == b:
                    continue
                jw_a, rw_b = self.compare_options(a, b, grounds)
                self.last_weights[a,b] = (jw_a, rw_b)
                if jw_a < rw_b:
                    allowed = False
                    break
            if allowed:
                permitted.append(a)

        choice = "Hold" if not permitted else np.random.choice(permitted)
        self.execute(choice, prices.iloc[day])
        
        # Store the weights used for this decision
        # self.last_weights = option_weights
        
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

            # Enhanced context adjustments
            # RSI-based decisions
            if "RSI_very_oversold" in g and a == "Buy":
                jw *= 2.0 if self.style == "aggressive" else 1.5
            elif "RSI_oversold" in g and a == "Buy":
                jw *= 1.5 if self.style == "aggressive" else 1.2
            elif "RSI_oversold_mild" in g and a == "Buy":
                jw *= 1.2 if self.style == "aggressive" else 1.0
            elif "RSI_very_overbought" in g and a == "Sell":
                jw *= 2.0 if self.style == "conservative" else 1.5
            elif "RSI_overbought" in g and a == "Sell":
                jw *= 1.5 if self.style == "conservative" else 1.2
            elif "RSI_overbought_mild" in g and a == "Sell":
                jw *= 1.2 if self.style == "conservative" else 1.0
            
            # Moving average trends
            elif g == "MA_strong_uptrend" and a == "Buy":
                jw *= 1.8 if self.style == "aggressive" else 1.3
            elif g == "MA_strong_downtrend" and a == "Sell":
                jw *= 1.8 if self.style == "conservative" else 1.3
            elif g == "MA_crossover" and a == "Buy":
                jw *= 1.3
            
            # MACD signals
            elif g == "MACD_bullish" and a == "Buy":
                jw *= 1.4
            elif g == "MACD_bearish" and a == "Sell":
                jw *= 1.4
            elif g == "MACD_momentum_up" and a == "Buy":
                jw *= 1.2
            elif g == "MACD_momentum_down" and a == "Sell":
                jw *= 1.2
            
            # Bollinger Band signals
            elif g == "BB_breakout_down" and a == "Buy":
                jw *= 1.5 if self.style == "aggressive" else 0.8  # Contrarian vs trend following
            elif g == "BB_breakout_up" and a == "Sell":
                jw *= 1.5 if self.style == "conservative" else 0.8
            elif g == "BB_breakout_up" and a == "Buy":
                jw *= 1.3 if self.style == "aggressive" else 1.0  # Momentum following
            elif g == "BB_breakout_down" and a == "Sell":
                jw *= 1.3 if self.style == "conservative" else 1.0
            
            # Momentum signals
            elif g == "Price_strong_momentum_up" and a == "Buy":
                jw *= 1.6 if self.style == "aggressive" else 1.2
            elif g == "Price_strong_momentum_down" and a == "Sell":
                jw *= 1.6 if self.style == "conservative" else 1.2
            elif g == "Price_momentum_up" and a == "Buy":
                jw *= 1.3 if self.style == "aggressive" else 1.1
            elif g == "Price_momentum_down" and a == "Sell":
                jw *= 1.3 if self.style == "conservative" else 1.1
            
            # Volatility adjustments
            elif g == "High_volatility":
                if self.style == "conservative":
                    jw *= 0.7 if a != "Hold" else 1.3  # Conservative prefers to hold in high vol
                elif self.style == "aggressive":
                    jw *= 1.4  # Aggressive likes volatility opportunities
            elif g == "Low_volatility" and a != "Hold":
                jw *= 0.9 if self.style == "aggressive" else 1.1  # Less opportunity for aggressive
            
            # Position management
            elif g == "EquityBigGain" and a == "Sell":
                jw *= 2.0 if self.style == "conservative" else 1.5  # Take profits
            elif g == "EquityLoss" and a == "Sell":
                jw *= 1.8 if self.style == "conservative" else 1.2  # Cut losses
            elif g == "EquityGainSell" and a == "Sell":
                jw *= 1.5
            elif g == "EquityGainBuy" and a == "Buy":
                jw *= 1.3 if self.style == "aggressive" else 1.0

            jw_a += jw
            rw_b += rw
        return jw_a, rw_b

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
# Collective Agents
# -----------------------------
class MajorityVoteAgent(Agent):
    def __init__(self, agents):
        super().__init__("Majority Vote", style="collective")
        self.agents = agents

    def decide(self, day, prices, last_votes=None):
        if not last_votes:
            choice = "Hold"
        else:
            # majority vote
            choice = max(set(last_votes), key=last_votes.count)
        self.execute(choice, prices.iloc[day])
        return choice

class WeightAggregationAgent(Agent):
    def __init__(self, agents):
        super().__init__("Weight Aggregation", style="collective")
        self.agents = agents

    def decide(self, day, prices, agent_weights=None):
        if not agent_weights or len(agent_weights) != len(self.agents):
            choice = "Hold"
        else:
            # Aggregate weights from all individual agents
            aggregated_weights = {("Buy","Buy"): (0,0), "Sell": 0, "Hold": 0}
            
            for agent_weight in agent_weights:
                for a in ["Buy", "Sell", "Hold"]:
                    for b in ["Buy", "Sell", "Hold"]:
                        aggregated_weights[a,b] += agent_weight[a,b]
            
            # Choose option with highest aggregated weight
            choice = max(aggregated_weights, key=aggregated_weights.get)
            
        self.execute(choice, prices.iloc[day])
        return choice

# -----------------------------
# Experiment
# -----------------------------
def run_experiment(prices):
    conservative = Agent("Conservative", style="conservative")
    balanced = Agent("Balanced", style="balanced")
    aggressive = Agent("Aggressive", style="aggressive")

    majority_vote = MajorityVoteAgent([conservative, balanced, aggressive])
    weight_aggregation = WeightAggregationAgent([conservative, balanced, aggressive])

    agents = [conservative, balanced, aggressive, majority_vote, weight_aggregation]

    decisions = {a.name: [] for a in agents}
    warmup = 50

    for day in range(warmup, len(prices)):
        day_votes = []
        day_weights = []
        
        for agent in agents:
            if isinstance(agent, MajorityVoteAgent):
                choice = agent.decide(day, prices, last_votes=day_votes)
            elif isinstance(agent, WeightAggregationAgent):
                choice = agent.decide(day, prices, agent_weights=day_weights)
            else:
                choice = agent.decide(day, prices)
                day_votes.append(choice)
                # Store the weights this agent calculated
                day_weights.append(agent.last_weights)
            decisions[agent.name].append(choice)

    return agents, decisions


# -----------------------------
# Plotting
# -----------------------------
def plot_results(prices, agents, decisions, warmup=50, show=True):
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
    if show:
        plt.show()


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=300)
    prices = pd.Series(np.cumsum(np.random.randn(300)) + 100, index=dates)

    agents, decisions = run_experiment(prices)
    plot_results(prices, agents, decisions)

    for a in agents:
        print(f"{a.name}: final equity = {a.history[-1]:.2f}")