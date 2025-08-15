from typing import Dict, List, Tuple, Callable
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
    denom = (high_max - low_min)
    return 100 * ((series.iloc[-1] - low_min) / denom) if denom != 0 else 0.0


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
        self.equity_history: List[float] = []
        self.actions_history: List[str] = []

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
        if action == "Buy" and self.position == 0 and self.equity > 0:
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

# --- New Function: Equity Drawdown Ratio ---
def equity_drawdown_ratio(agent: DualWeightAgent) -> float:
    eq = np.array(agent.equity_history, dtype=float)
    if len(eq) == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    drawdowns = running_max - eq
    max_drawdown = drawdowns.max()
    total_gain = eq[-1] - eq[0]
    return float(max_drawdown / total_gain) if total_gain != 0 else 0.0



def generate_weights(options: List[Option], grounds: List[Ground], profile: str) -> Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]]:
    weights: Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]] = {}
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
    combined_weights: Dict[Tuple[Ground, Option, Option], Dict[WeightType, float]] = {}
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
    # For each day compute market-derived grounds (shared), then add agent-specific EquityGainSell
    for i in range(len(prices)):
        price = prices.iloc[i]
        base_grounds = {
            "MA20": moving_average(prices[:i+1], 20) if i >= 19 else 0,
            "RSI14": rsi(prices[:i+1], 14) if i >= 13 else 0,
            "Stoch14": stochastic_oscillator(prices[:i+1], 14) if i >= 13 else 0,
            "MACD": macd(prices[:i+1]) if i >= 26 else 0,
        }
        for agent in agents:
            # copy base grounds and add an agent-specific ground
            g = dict(base_grounds)
            g["EquityGainSell"] = equity_gain_if_sell(price, agent.entry_price)
            g["EquityDrawdownRatio"] = equity_drawdown_ratio(agent)
            action = agent.choose_action(g)
            agent.update_equity(action, price)


def plot_equity(agents: List[DualWeightAgent]):
    plt.figure(figsize=(10, 5))
    for agent in agents:
        label_final = f" (Final: {agent.equity_history[-1]:.2f})" if agent.equity_history else ""
        plt.plot(agent.equity_history, label=f"{agent.name}{label_final}")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.title("Equity Over Time")
    plt.legend()
    plt.show()
    # Print ranking
    ranking = sorted(agents, key=lambda a: a.equity_history[-1] if a.equity_history else a.equity, reverse=True)
    print("\nAgent Ranking (by final equity):")
    for idx, agent in enumerate(ranking, 1):
        final_val = agent.equity_history[-1] if agent.equity_history else agent.equity
        print(f"{idx}. {agent.name}: {final_val:.2f}")


def plot_market(prices: pd.Series, collective_actions: List[str]):
    plt.figure(figsize=(10, 4))
    plt.plot(prices, label="Closing Price")
    for idx, action in enumerate(collective_actions):
        if idx >= len(prices):
            break
        if action == "Buy":
            plt.scatter(idx, prices.iloc[idx], marker='^', color='green', s=40)
        elif action == "Sell":
            plt.scatter(idx, prices.iloc[idx], marker='v', color='red', s=40)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Market Closing Prices & Collective Agent Actions")
    plt.legend()
    plt.show()


def plot_action_bars(agents: List[DualWeightAgent]):
    collective_actions = agents[-1].actions_history
    n = len(agents)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    for idx, agent in enumerate(agents):
        match_count = sum(a == c for a, c in zip(agent.actions_history, collective_actions))
        total = len(agent.actions_history)
        action_counts = pd.Series(agent.actions_history).value_counts()
        # if only one axis returned turn into list
        ax = axs[idx] if n > 1 else axs
        ax.bar(action_counts.index.astype(str), action_counts.values)
        ax.set_title(f"{agent.name}\nMatches with Collective: {match_count}/{total}")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def _daily_returns(equity: List[float]) -> np.ndarray:
    eq = np.array(equity, dtype=float)
    if len(eq) < 2:
        return np.array([0.0])
    shifted = np.roll(eq, 1)
    shifted[0] = eq[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = (eq - shifted) / shifted
        rets[~np.isfinite(rets)] = 0.0
    return rets


def _max_drawdown(equity: List[float]) -> float:
    eq = np.array(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    drawdowns[np.isnan(drawdowns)] = 0.0
    return float(drawdowns.min())  # negative number


def evaluate_agents(agents: List[DualWeightAgent], initial_equity: float = 10000.0, trading_days_per_year: int = 252) -> pd.DataFrame:
    rows = []
    for agent in agents:
        eq = agent.equity_history
        if not eq:
            continue
        final_eq = eq[-1]
        total_return = (final_eq / initial_equity) - 1.0 if initial_equity > 0 else 0.0
        N = len(eq)
        years = max(N / trading_days_per_year, 1e-9)
        cagr = (final_eq / initial_equity) ** (1 / years) - 1 if initial_equity > 0 else 0.0
        rets = _daily_returns(eq)
        mean_daily = float(np.mean(rets))
        std_daily = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        vol_annual = std_daily * np.sqrt(trading_days_per_year)
        sharpe = (mean_daily / std_daily) * np.sqrt(trading_days_per_year) if std_daily > 0 else 0.0
        mdd = _max_drawdown(eq)
        trades = sum(1 for i in range(1, len(agent.actions_history)) if agent.actions_history[i] != agent.actions_history[i-1])
        rows.append({
            "Agent": agent.name,
            "Final Equity": final_eq,
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility (ann)": vol_annual,
            "Sharpe": sharpe,
            "Max Drawdown": mdd,
            "#Action Changes": trades
        })
    df = pd.DataFrame(rows).sort_values(by=["Final Equity"], ascending=False)
    return df


def pick_best_agent(df: pd.DataFrame, objective: str = "Final Equity") -> str:
    if df.empty:
        return ""
    if objective not in df.columns:
        raise ValueError(f"Objective '{objective}' not in results columns: {list(df.columns)}")
    if objective == "Max Drawdown":
        # Max Drawdown is negative (worse = more negative). pick the largest (closest to zero).
        best_row = df.sort_values(by="Max Drawdown", ascending=False).iloc[0]
    else:
        best_row = df.sort_values(by=objective, ascending=False).iloc[0]
    return best_row["Agent"]


def plot_final_equity_bar(df: pd.DataFrame):
    plt.figure(figsize=(8,4))
    plt.bar(df["Agent"], df["Final Equity"]) 
    plt.xlabel("Agent")
    plt.ylabel("Final Equity")
    plt.title("Final Equity by Agent")
    plt.show()

def plot_final_equity_distribution(df: pd.DataFrame):
    # Violin plot instead of bar plot to show distribution across experiments

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Agent", y="Final Equity", data=df, inner="quartile", palette="muted", hue="Agent")
    plt.title("Distribution of Final Equity Across Experiments")
    plt.xlabel("Agent")
    plt.ylabel("Final Equity")
    plt.show()


# if __name__ == "__main__":
#     np.random.seed(42)
#     days = 200
#     prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, days)))
#     options = ["Buy", "Sell", "Hold"]
#     grounds = ["MA20", "RSI14", "Stoch14", "MACD", "EquityGainSell"]

#     conservative = DualWeightAgent(options, grounds, generate_weights(options, grounds, "conservative"), name="Conservative")
#     balanced = DualWeightAgent(options, grounds, generate_weights(options, grounds, "balanced"), name="Balanced")
#     aggressive = DualWeightAgent(options, grounds, generate_weights(options, grounds, "aggressive"), name="Aggressive")
#     collective = combine_agents([conservative, balanced, aggressive])

#     agents = [conservative, balanced, aggressive, collective]
#     run_experiment(prices, agents)

#     # Plots
#     plot_market(prices, collective.actions_history)
#     plot_equity(agents)
#     plot_action_bars(agents)

#     # Evaluation + ranking
#     results = evaluate_agents(agents, initial_equity=10000.0, trading_days_per_year=252)
#     print("\n=== Agent Performance ===")
#     print(results.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

#     # Pick best by a chosen objective
#     objective = "Final Equity"  # or "Sharpe", "CAGR", "Max Drawdown"
#     best = pick_best_agent(results, objective=objective)
#     print(f"\nBest agent by {objective}: {best}")

#     plot_final_equity_bar(results)
def run_multiple_experiments(num_experiments: int = 10, days: int = 200, seed: int = 42):
    np.random.seed(seed)
    all_results = []

    for exp in range(num_experiments):
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, days)))
        options = ["Buy", "Sell", "Hold"]
        grounds = ["MA20", "RSI14", "Stoch14", "MACD", "EquityGainSell"]

        conservative = DualWeightAgent(options, grounds, generate_weights(options, grounds, "conservative"), name="Conservative")
        balanced = DualWeightAgent(options, grounds, generate_weights(options, grounds, "balanced"), name="Balanced")
        aggressive = DualWeightAgent(options, grounds, generate_weights(options, grounds, "aggressive"), name="Aggressive")
        collective = combine_agents([conservative, balanced, aggressive])

        agents = [conservative, balanced, aggressive, collective]
        run_experiment(prices, agents)

        # Evaluate and store results
        results = evaluate_agents(agents, initial_equity=10000.0, trading_days_per_year=252)
        results["Experiment"] = exp + 1
        all_results.append(results)

    # Combine all experiment results
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    combined_results = run_multiple_experiments(num_experiments=2000, days=200)

    # Show summary across experiments
    summary = combined_results.groupby("Agent").agg({
        "Final Equity": ["mean", "std"],
        "Total Return": ["mean", "std"],
        "CAGR": ["mean", "std"],
        "Sharpe": ["mean", "std"],
        "Max Drawdown": ["mean", "std"]
    })
    print("\n=== Summary Across Experiments ===")
    print(summary)

    # Optional: plot final equity distribution for each agent
    plot_final_equity_distribution(combined_results)
    # plt.figure(figsize=(8,5))
    # for agent_name in combined_results["Agent"].unique():
    #     eqs = combined_results[combined_results["Agent"] == agent_name]["Final Equity"]
    #     plt.hist(eqs, alpha=0.5, label=agent_name, bins=10)
    # plt.xlabel("Final Equity")
    # plt.ylabel("Frequency")
    # plt.title("Final Equity Distribution Across Experiments")
    # plt.legend()
    # plt.show()
