# class MultiAgentTradingSimulator:
#     """Enhanced backtesting simulator supporting multiple agent types and collective agents"""
    
#     def __init__(self, initial_capital: float = 100000):
#         self.initial_capital = initial_capital
        
#         # Create individual agents
#         self.agents = {
#             AgentType.RISK_TAKING: TradingAgent(agent_type=AgentType.RISK_TAKING),
#             AgentType.BALANCED: TradingAgent(agent_type=AgentType.BALANCED),
#             AgentType.CONSERVATIVE: TradingAgent(agent_type=AgentType.CONSERVATIVE)
#         }
        
#         # Create collective agents with different aggregation methods
#         self.collective_agents = {
#             'weighted_average': CollectiveTradingAgent('weighted_average'),
#             'maximum': CollectiveTradingAgent('maximum'),
#             'minimum': CollectiveTradingAgent('minimum'),
#             'simple_average': CollectiveTradingAgent('simple_average')
#         }
        
#         # Initialize portfolios
#         self.portfolios = {}
#         self.reset_portfolios()
    
#     def reset_portfolios(self):
#         """Reset all agent portfolios"""
#         for agent_name in list(self.agents.keys()) + list(self.collective_agents.keys()):
#             self.portfolios[agent_name] = {
#                 'capital': self.initial_capital,
#                 'position': 0.0,
#                 'trades': [],
#                 'portfolio_values': [],
#                 'decisions': []
#             }
    
#     def run_comparative_backtest(self, price_data: pd.DataFrame, transaction_cost: float = 0.001):
#         """Run backtest comparing all agent types and collective strategies"""
        
#         # Prepare data
#         sample_agent = list(self.agents.values())[0]
#         data = sample_agent.prepare_data(price_data)
        
#         start_idx = 50
#         all_results = {}
        
#         for i in range(start_idx, len(data)):
#             current_data = data.iloc[:i+1]
#             current_price = current_data['close'].iloc[-1]
            
#             # Test individual agents
#             for agent_type, agent in self.agents.items():
#                 self._execute_trade_for_agent(
#                     agent_type, agent, current_data, current_price, transaction_cost
#                 )
            
#             # Test collective agents
#             for collective_name, collective_agent in self.collective_agents.items():
#                 self._execute_collective_trade(
#                     collective_name, collective_agent, current_data, current_price, transaction_cost
#                 )
        
#         # Compile results
#         for agent_name, portfolio in self.portfolios.items():
#             if portfolio['portfolio_values']:
#                 final_value = portfolio['portfolio_values'][-1]
#                 total_return = (final_value - self.initial_capital) / self.initial_capital
                
#                 all_results[agent_name] = {
#                     'agent_type': agent_name,
#                     'final_return': total_return,
#                     'total_trades': len(portfolio['trades']),
#                     'final_portfolio_value': final_value,
#                     'portfolio_values': portfolio['portfolio_values'],
#                     'decisions': portfolio['decisions'],
#                     'trades': portfolio['trades']
#                 }
        
#         return all_results
    
#     def _execute_trade_for_agent(self, agent_name, agent, current_data, current_price, transaction_cost):
#         """Execute trade for individual agent"""
#         portfolio = self.portfolios[agent_name]
        
#         decision, explanation, debug_info = agent.make_decision(current_data)
        
#         old_position = portfolio['position']
        
#         # Agent-specific position sizing
#         if agent.get_agent_type() == AgentType.RISK_TAKING:
#             max_position = 0.95
#             trade_size = 0.15  # Larger trades for risk-taker
#         elif agent.get_agent_type() == AgentType.CONSERVATIVE:
#             max_position = 0.7
#             trade_size = 0.05  # Smaller trades for conservative
#         else:
#             max_position = 0.8
#             trade_size = 0.1  # Balanced trades
        
#         if decision == TradingAction.BUY and portfolio['position'] < max_position:
#             trade_amount = min(trade_size, max_position - portfolio['position'])
#             shares_to_buy = (trade_amount * portfolio['capital']) / current_price
#             cost = shares_to_buy * current_price * (1 + transaction_cost)
            
#             if portfolio['capital'] >= cost:
#                 portfolio['position'] += trade_amount
#                 portfolio['capital'] -= cost
#                 portfolio['trades'].append({
#                     'date': current_data.index[-1],
#                     'action': 'BUY',
#                     'price': current_price,
#                     'amount': trade_amount,
#                     'explanation': explanation
#                 })
                
#         elif decision == TradingAction.SELL and portfolio['position'] > -max_position:
#             trade_amount = min(trade_size, portfolio['position'] + max_position)
#             if trade_amount > 0:
#                 shares_to_sell = (trade_amount * portfolio['capital']) / current_price
#                 proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                
#                 portfolio['position'] -= trade_amount
#                 portfolio['capital'] += proceeds
#                 portfolio['trades'].append({
#                     'date': current_data.index[-1],
#                     'action': 'SELL',
#                     'price': current_price,
#                     'amount': trade_amount,
#                     'explanation': explanation
#                 })
        
#         # Calculate portfolio value
#         position_value = portfolio['position'] * portfolio['capital'] * current_price / self.initial_capital
#         portfolio_value = portfolio['capital'] + position_value
#         portfolio['portfolio_values'].append(portfolio_value)
        
#         portfolio['decisions'].append({
#             'date': current_data.index[-1],
#             'decision': decision.value,
#             'explanation': explanation,
#             'portfolio_value': portfolio_value,
#             'position': portfolio['position'],
#             'debug_info': debug_info
#         })
    
#     def analyze_agent_performance(self, results: Dict) -> Dict:
#         """Analyze and compare agent performance"""
#         analysis = {}
        
#         for agent_name, result in results.items():
#             portfolio_values = result['portfolio_values']
            
#             if len(portfolio_values) > 1:
#                 # Calculate metrics
#                 returns = np.diff(portfolio_values) / portfolio_values[:-1]
                
#                 analysis[agent_name] = {
#                     'total_return': result['final_return'],
#                     'annualized_return': result['final_return'] * (252 / len(portfolio_values)),  # Assuming daily data
#                     'volatility': np.std(returns) * np.sqrt(252),
#                     'sharpe_ratio': (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
#                     'max_drawdown': self._calculate_max_drawdown(portfolio_values),
#                     'total_trades': result['total_trades'],
#                     'win_rate': self._calculate_win_rate(result['trades']),
#                     'avg_trade_return': self._calculate_avg_trade_return(result['trades'])
#                 }
        
#         return analysis
    
#     def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
#         """Calculate maximum drawdown"""
#         peak = portfolio_values[0]
#         max_dd = 0
        
#         for value in portfolio_values:
#             if value > peak:
#                 peak = value
#             dd = (peak - value) / peak
#             if dd > max_dd:
#                 max_dd = dd
                
#         return max_dd
    
#     def _calculate_win_rate(self, trades: List[Dict]) -> float:
#         """Calculate win rate of trades"""
#         if len(trades) < 2:
#             return 0
        
#         wins = 0
#         for i in range(1, len(trades)):
#             prev_trade = trades[i-1]
#             curr_trade = trades[i]
            
#             if prev_trade['action'] == 'BUY' and curr_trade['action'] == 'SELL':
#                 if curr_trade['price'] > prev_trade['price']:
#                     wins += 1
#             elif prev_trade['action'] == 'SELL' and curr_trade['action'] == 'BUY':
#                 if prev_trade['price'] > curr_trade['price']:
#                     wins += 1
        
#         return wins / (len(trades) // 2) if len(trades) >= 2 else 0
    
#     def _calculate_avg_trade_return(self, trades: List[Dict]) -> float:
#         """Calculate average trade return"""
#         if len(trades) < 2:
#             return 0
        
#         trade_returns = []
#         for i in range(1, len(trades)):
#             prev_trade = trades[i-1]
#             curr_trade = trades[i]
            
#             if prev_trade['action'] == 'BUY' and curr_trade['action'] == 'SELL':
#                 trade_return = (curr_trade['price'] - prev_trade['price']) / prev_trade['price']
#                 trade_returns.append(trade_return)
#             elif prev_trade['action'] == 'SELL' and curr_trade['action'] == 'BUY':
#                 trade_return = (prev_trade['price'] - curr_trade['price']) / prev_trade['price']
#                 trade_returns.append(trade_return)
        
#         return np.mean(trade_returns) if trade_returns else 0

# # Example usage and testing
# if __name__ == "__main__":
#     print("Dual Scale Detachment Multi-Agent Trading System")
#     print("=" * 50)
    
#     # Generate sample data
#     print("Generating sample market data...")
#     sample_data = generate_sample_data(500)
    
#     # Run multi-agent backtest
#     print("Running multi-agent comparative backtest...")
#     simulator = MultiAgentTradingSimulator()
#     results = simulator.run_comparative_backtest(sample_data)
    
#     # Analyze performance
#     print("\nAnalyzing agent performance...")
#     performance_analysis = simulator.analyze_agent_performance(results)
    
#     # Display results
#     print(f"\n{'Agent Type':<20} {'Return':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<7} {'Win Rate':<9}")
#     print("=" * 70)
    
#     # Sort by return for better display
#     sorted_agents = sorted(performance_analysis.items(), key=lambda x: x[1]['total_return'], reverse=True)
    
#     for agent_name, metrics in sorted_agents:
#         agent_display = str(agent_name).replace('AgentType.', '').replace('_', ' ').title()
#         if len(agent_display) > 19:
#             agent_display = agent_display[:16] + "..."
            
#         print(f"{agent_display:<20} "
#               f"{metrics['total_return']:>8.2%} "
#               f"{metrics['sharpe_ratio']:>7.2f} "
#               f"{metrics['max_drawdown']:>7.2%} "
#               f"{metrics['total_trades']:>6d} "
#               f"{metrics['win_rate']:>8.2%}")
    
#     print("\n" + "=" * 70)
    
#     # Show detailed results for best and worst performers
#     best_agent = sorted_agents[0][0]
#     worst_agent = sorted_agents[-1][0]
    
#     print(f"\nBest Performer: {best_agent}")
#     print(f"Final Portfolio Value: ${results[best_agent]['final_portfolio_value']:,.2f}")
#     print(f"Last 3 trades:")
#     for trade in results[best_agent]['trades'][-3:]:
#         print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} at ${trade['price']:.2f}")
    
#     print(f"\nWorst Performer: {worst_agent}")  
#     print(f"Final Portfolio Value: ${results[worst_agent]['final_portfolio_value']:,.2f}")
#     print(f"Last 3 trades:")
#     for trade in results[worst_agent]['trades'][-3:]:
#         print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} at ${trade['price']:.2f}")
    
#     # Show collective agent consensus analysis
#     print(f"\nCollective Agent Analysis:")
#     print("-" * 30)
    
#     collective_results = {k: v for k, v in results.items() if k in simulator.collective_agents.keys()}
    
#     for collective_name, result in collective_results.items():
#         print(f"\n{collective_name.replace('_', ' ').title()} Collective:")
#         print(f"  Final Return: {result['final_return']:.2%}")
#         print(f"  Total Trades: {result['total_trades']}")
        
#         # Show last decision with consensus info
#         if result['decisions']:
#             last_decision = result['decisions'][-1]
#             if 'consensus_analysis' in last_decision['debug_info']:
#                 consensus = last_decision['debug_info']['consensus_analysis']
#                 print(f"  Last Decision: {last_decision['decision']} ({consensus['type']} consensus)")
                
#                 if 'individual_decisions' in last_decision['debug_info']:
#                     individual_decs = last_decision['debug_info']['individual_decisions']
#                     print("  Individual agent votes:")
#                     for agent_type, info in individual_decs.items():
#                         agent_name = str(agent_type).replace('AgentType.', '')
#                         print(f"    {agent_name}: {info['decision'].value}")
    
#     # Performance comparison summary
#     print(f"\n\nPerformance Summary:")
#     print(f"Individual Agents Average Return: {np.mean([performance_analysis[agent]['total_return'] for agent in [AgentType.RISK_TAKING, AgentType.BALANCED, AgentType.CONSERVATIVE]]):,.2%}")
    
#     collective_avg = np.mean([performance_analysis[agent]['total_return'] for agent in simulator.collective_agents.keys()])
#     print(f"Collective Agents Average Return: {collective_avg:,.2%}")
    
#     best_collective = max([(name, performance_analysis[name]['total_return']) for name in simulator.collective_agents.keys()], key=lambda x: x[1])
#     print(f"Best Collective Strategy: {best_collective[0]} ({best_collective[1]:.2%})")
    
#     print(f"\nDual Scale Detachment successfully implemented with:")
#     print(f"✓ Individual agent types (Risk-taking, Balanced, Conservative)")
#     print(f"✓ Collective agents with multiple aggregation methods")
#     print(f"✓ Normative reasoning with justifying/requiring weight distinction") 
#     print(f"✓ Context-sensitive reason generation")
#     print(f"✓ Comprehensive backtesting and performance analysis") / self.initial_capital
#         portfolio_value = portfolio['capital'] + position_value
#         portfolio['portfolio_values'].append(portfolio_value)
        
#         portfolio['decisions'].append({
#             'date': current_data.index[-1],
#             'decision': decision.value,
#             'explanation': explanation,
#             'portfolio_value': portfolio_value,
#             'position': portfolio['position'],
#             'debug_info': debug_info
#         })
    
#     def _execute_collective_trade(self, collective_name, collective_agent, current_data, current_price, transaction_cost):
#         """Execute trade for collective agent"""
#         portfolio = self.portfolios[collective_name]
        
#         decision, explanation, debug_info = collective_agent.make_collective_decision(current_data)
        
#         # Collective agents use balanced position sizing
#         max_position = 0.85
#         trade_size = 0.1
        
#         if decision == TradingAction.BUY and portfolio['position'] < max_position:
#             trade_amount = min(trade_size, max_position - portfolio['position'])
#             shares_to_buy = (trade_amount * portfolio['capital']) / current_price
#             cost = shares_to_buy * current_price * (1 + transaction_cost)
            
#             if portfolio['capital'] >= cost:
#                 portfolio['position'] += trade_amount
#                 portfolio['capital'] -= cost
#                 portfolio['trades'].append({
#                     'date': current_data.index[-1],
#                     'action': 'BUY',
#                     'price': current_price,
#                     'amount': trade_amount,
#                     'explanation': explanation
#                 })
                
#         elif decision == TradingAction.SELL and portfolio['position'] > -max_position:
#             trade_amount = min(trade_size, portfolio['position'] + max_position)
#             if trade_amount > 0:
#                 shares_to_sell = (trade_amount * portfolio['capital']) / current_price
#                 proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                
#                 portfolio['position'] -= trade_amount
#                 portfolio['capital'] += proceeds
#                 portfolio['trades'].append({
#                     'date': current_data.index[-1],
#                     'action': 'SELL',
#                     'price': current_price,
#                     'amount': trade_amount,
#                     'explanation': explanation
#                 })
        
#         # Calculate portfolio value
#         position_value = portfolio['position'] * portfolio['capital'] * current_price

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
from dataclasses import dataclass
# Create working version without external dependencies for testing
def simple_rsi(prices, period=14):
    """Simple RSI calculation without TA-Lib"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.convolve(gains, np.ones(period), 'valid') / period
    avg_losses = np.convolve(losses, np.ones(period), 'valid') / period
    
    rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Pad with NaN to match original length
    return np.concatenate([np.full(len(prices) - len(rsi), np.nan), rsi])

def simple_sma(prices, period):
    """Simple moving average"""
    return pd.Series(prices).rolling(window=period).mean().values

def simple_macd(prices, fast=12, slow=26, signal=9):
    """Simple MACD calculation"""
    exp_fast = pd.Series(prices).ewm(span=fast).mean()
    exp_slow = pd.Series(prices).ewm(span=slow).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line.values, signal_line.values
from abc import ABC, abstractmethod

class AgentType(Enum):
    RISK_TAKING = "risk_taking"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"

class ScaleValue(Enum):
    POSITIVE = "+"
    NEGATIVE = "-"
    NEUTRAL = "0"

class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingReason:
    """Represents a reason for a trading action based on market grounds"""
    ground: str  # e.g., "RSI_oversold", "MA_crossover"
    action: TradingAction
    justifying_weight: float  # How good at making action permissible
    requiring_weight: float   # How good at making action required
    
class MarketContext:
    """Represents market conditions that affect reason weights"""
    def __init__(self, volatility: float, volume: float, trend: str, market_hours: bool):
        self.volatility = volatility
        self.volume = volume  
        self.trend = trend  # "bull", "bear", "sideways"
        self.market_hours = market_hours
        
class ReasonGenerator(ABC):
    """Abstract base class for generating trading reasons"""
    
    @abstractmethod
    def get_agent_type(self) -> AgentType:
        pass
    
    @abstractmethod
    def get_risk_multiplier(self) -> float:
        pass
    
    @abstractmethod
    def get_weight_adjustment(self, base_justifying: float, base_requiring: float) -> Tuple[float, float]:
        pass
    
    def generate_reasons(self, data: pd.DataFrame, context: MarketContext) -> List[TradingReason]:
        """Generate reasons based on technical indicators with agent-specific adjustments"""
        reasons = []
        latest = data.iloc[-1]
        
        # RSI-based reasons
        rsi = latest['RSI']
        if rsi < 30:  # Oversold
            base_jw = 3.0 * self._volatility_modifier(context)
            base_rw = 1.5 * self._volume_modifier(context)
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="RSI_oversold",
                action=TradingAction.BUY,
                justifying_weight=jw,
                requiring_weight=rw
            ))
        elif rsi > 70:  # Overbought
            base_jw = 3.0 * self._volatility_modifier(context)
            base_rw = 1.5 * self._volume_modifier(context)
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="RSI_overbought", 
                action=TradingAction.SELL,
                justifying_weight=jw,
                requiring_weight=rw
            ))
            
        # Moving average crossover reasons
        if latest['MA_short'] > latest['MA_long'] and data.iloc[-2]['MA_short'] <= data.iloc[-2]['MA_long']:
            base_jw = 2.5
            base_rw = 2.0 * self._trend_modifier(context)
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="MA_golden_cross",
                action=TradingAction.BUY,
                justifying_weight=jw,
                requiring_weight=rw
            ))
        elif latest['MA_short'] < latest['MA_long'] and data.iloc[-2]['MA_short'] >= data.iloc[-2]['MA_long']:
            base_jw = 2.5
            base_rw = 2.0 * self._trend_modifier(context)
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="MA_death_cross",
                action=TradingAction.SELL,
                justifying_weight=jw,
                requiring_weight=rw
            ))
            
        # MACD reasons
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD'] > 0:
            base_jw = 2.0
            base_rw = 1.0
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="MACD_bullish",
                action=TradingAction.BUY,
                justifying_weight=jw,
                requiring_weight=rw
            ))
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD'] < 0:
            base_jw = 2.0
            base_rw = 1.0
            jw, rw = self.get_weight_adjustment(base_jw, base_rw)
            
            reasons.append(TradingReason(
                ground="MACD_bearish",
                action=TradingAction.SELL,
                justifying_weight=jw,
                requiring_weight=rw
            ))
            
        # Volume-based reasons
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        if latest['volume'] > 1.5 * avg_volume:
            price_change = (latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']
            if price_change > 0.01:  # 1% up
                base_jw = 1.5
                base_rw = 2.5
                jw, rw = self.get_weight_adjustment(base_jw, base_rw)
                
                reasons.append(TradingReason(
                    ground="high_volume_up",
                    action=TradingAction.BUY,
                    justifying_weight=jw,
                    requiring_weight=rw
                ))
            elif price_change < -0.01:  # 1% down
                base_jw = 1.5
                base_rw = 2.5
                jw, rw = self.get_weight_adjustment(base_jw, base_rw)
                
                reasons.append(TradingReason(
                    ground="high_volume_down", 
                    action=TradingAction.SELL,
                    justifying_weight=jw,
                    requiring_weight=rw
                ))
        
        return reasons
    
    def _volatility_modifier(self, context: MarketContext) -> float:
        """Higher volatility increases justifying weights (more permissive)"""
        base_modifier = 1.0 + (context.volatility - 0.02) * 5
        return base_modifier * self.get_risk_multiplier()
    
    def _volume_modifier(self, context: MarketContext) -> float:
        """Higher volume increases requiring weights (more obligation)"""
        return 1.0 + context.volume * 0.1
    
    def _trend_modifier(self, context: MarketContext) -> float:
        """Trend alignment increases requiring weights"""
        base_modifier = 1.0
        if context.trend == "bull":
            base_modifier = 1.5  
        elif context.trend == "bear": 
            base_modifier = 1.5  
        return base_modifier

class RiskTakingReasonGenerator(ReasonGenerator):
    """Risk-taking agent: Higher weights, more aggressive trading"""
    
    def get_agent_type(self) -> AgentType:
        return AgentType.RISK_TAKING
    
    def get_risk_multiplier(self) -> float:
        return 1.5  # 50% higher risk tolerance
    
    def get_weight_adjustment(self, base_justifying: float, base_requiring: float) -> Tuple[float, float]:
        # Risk-taking: Boost both weights, especially requiring weights (more obligation to act)
        return (base_justifying * 1.3, base_requiring * 1.6)

class BalancedReasonGenerator(ReasonGenerator):
    """Balanced agent: Standard weights"""
    
    def get_agent_type(self) -> AgentType:
        return AgentType.BALANCED
    
    def get_risk_multiplier(self) -> float:
        return 1.0  # Neutral risk tolerance
    
    def get_weight_adjustment(self, base_justifying: float, base_requiring: float) -> Tuple[float, float]:
        # Balanced: Use base weights as-is
        return (base_justifying, base_requiring)

class ConservativeReasonGenerator(ReasonGenerator):
    """Conservative agent: Lower weights, more cautious"""
    
    def get_agent_type(self) -> AgentType:
        return AgentType.CONSERVATIVE
    
    def get_risk_multiplier(self) -> float:
        return 0.7  # 30% lower risk tolerance
    
    def get_weight_adjustment(self, base_justifying: float, base_requiring: float) -> Tuple[float, float]:
        # Conservative: Reduce requiring weights significantly, slightly reduce justifying weights
        return (base_justifying * 0.9, base_requiring * 0.6)

class ReasonGenerator:
    """Generates trading reasons from market data"""
    
    def generate_reasons(self, data: pd.DataFrame, context: MarketContext) -> List[TradingReason]:
        """Generate reasons based on technical indicators"""
        reasons = []
        latest = data.iloc[-1]
        
        # RSI-based reasons
        rsi = latest['RSI']
        if rsi < 30:  # Oversold
            reasons.append(TradingReason(
                ground="RSI_oversold",
                action=TradingAction.BUY,
                justifying_weight=3.0 * self._volatility_modifier(context),
                requiring_weight=1.5 * self._volume_modifier(context)
            ))
        elif rsi > 70:  # Overbought
            reasons.append(TradingReason(
                ground="RSI_overbought", 
                action=TradingAction.SELL,
                justifying_weight=3.0 * self._volatility_modifier(context),
                requiring_weight=1.5 * self._volume_modifier(context)
            ))
            
        # Moving average crossover reasons
        if latest['MA_short'] > latest['MA_long'] and data.iloc[-2]['MA_short'] <= data.iloc[-2]['MA_long']:
            reasons.append(TradingReason(
                ground="MA_golden_cross",
                action=TradingAction.BUY,
                justifying_weight=2.5,
                requiring_weight=2.0 * self._trend_modifier(context)
            ))
        elif latest['MA_short'] < latest['MA_long'] and data.iloc[-2]['MA_short'] >= data.iloc[-2]['MA_long']:
            reasons.append(TradingReason(
                ground="MA_death_cross",
                action=TradingAction.SELL, 
                justifying_weight=2.5,
                requiring_weight=2.0 * self._trend_modifier(context)
            ))
            
        # MACD reasons
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD'] > 0:
            reasons.append(TradingReason(
                ground="MACD_bullish",
                action=TradingAction.BUY,
                justifying_weight=2.0,
                requiring_weight=1.0
            ))
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD'] < 0:
            reasons.append(TradingReason(
                ground="MACD_bearish",
                action=TradingAction.SELL,
                justifying_weight=2.0, 
                requiring_weight=1.0
            ))
            
        # Volume-based reasons
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        if latest['volume'] > 1.5 * avg_volume:
            # High volume can support either direction based on price movement
            price_change = (latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']
            if price_change > 0.01:  # 1% up
                reasons.append(TradingReason(
                    ground="high_volume_up",
                    action=TradingAction.BUY,
                    justifying_weight=1.5,
                    requiring_weight=2.5
                ))
            elif price_change < -0.01:  # 1% down
                reasons.append(TradingReason(
                    ground="high_volume_down", 
                    action=TradingAction.SELL,
                    justifying_weight=1.5,
                    requiring_weight=2.5
                ))
        
        return reasons
    
    def _volatility_modifier(self, context: MarketContext) -> float:
        """Higher volatility increases justifying weights (more permissive)"""
        return 1.0 + (context.volatility - 0.02) * 5  # Assuming 2% base volatility
    
    def _volume_modifier(self, context: MarketContext) -> float:
        """Higher volume increases requiring weights (more obligation)"""
        return 1.0 + context.volume * 0.1
    
    def _trend_modifier(self, context: MarketContext) -> float:
        """Trend alignment increases requiring weights"""
        if context.trend == "bull":
            return 1.5  # More obligation to buy in bull market
        elif context.trend == "bear": 
            return 1.5  # More obligation to sell in bear market
        return 1.0

class DualScaleDetachment:
    """Implements dual scale detachment for trading decisions"""
    
    def evaluate_options(self, reasons: List[TradingReason], 
                        option1: TradingAction, option2: TradingAction) -> Tuple[ScaleValue, ScaleValue]:
        """Compare two trading options using dual scale model"""
        
        # Separate reasons by action
        reasons_o1 = [r for r in reasons if r.action == option1]
        reasons_o2 = [r for r in reasons if r.action == option2]
        
        # Calculate aggregate weights
        jw_o1 = sum(r.justifying_weight for r in reasons_o1)
        rw_o1 = sum(r.requiring_weight for r in reasons_o1)
        jw_o2 = sum(r.justifying_weight for r in reasons_o2) 
        rw_o2 = sum(r.requiring_weight for r in reasons_o2)
        
        # Permission scale: JW(o1) vs RW(o2)
        if jw_o1 > rw_o2:
            v1 = ScaleValue.POSITIVE
        elif jw_o1 < rw_o2:
            v1 = ScaleValue.NEGATIVE
        else:
            v1 = ScaleValue.NEUTRAL
            
        # Commitment scale: JW(o2) vs RW(o1) 
        if jw_o2 > rw_o1:
            v2 = ScaleValue.POSITIVE
        elif jw_o2 < rw_o1:
            v2 = ScaleValue.NEGATIVE
        else:
            v2 = ScaleValue.NEUTRAL
            
        return (v1, v2)

class CollectiveReasonGenerator:
    """Collective agent that combines reasons from multiple individual agents"""
    
    def __init__(self, agent_generators: List[ReasonGenerator], 
                 aggregation_method: str = "weighted_average"):
        self.agent_generators = agent_generators
        self.aggregation_method = aggregation_method
        
        # Define weights for different agent types in collective decisions
        self.agent_weights = {
            AgentType.RISK_TAKING: 0.25,    # Lower weight for risk-taking
            AgentType.BALANCED: 0.50,       # Highest weight for balanced
            AgentType.CONSERVATIVE: 0.25    # Lower weight for conservative
        }
    
    def get_agent_type(self) -> AgentType:
        return AgentType.BALANCED  # Collective defaults to balanced
    
    def generate_reasons(self, data: pd.DataFrame, context: MarketContext) -> List[TradingReason]:
        """Aggregate reasons from all individual agents"""
        
        # Collect reasons from all agents
        all_agent_reasons = {}
        for agent in self.agent_generators:
            agent_type = agent.get_agent_type()
            agent_reasons = agent.generate_reasons(data, context)
            all_agent_reasons[agent_type] = agent_reasons
        
        # Aggregate reasons by ground and action
        aggregated_reasons = self._aggregate_reasons(all_agent_reasons)
        
        return aggregated_reasons
    
    def _aggregate_reasons(self, all_agent_reasons: Dict[AgentType, List[TradingReason]]) -> List[TradingReason]:
        """Aggregate reasons from multiple agents using specified method"""
        
        # Group reasons by (ground, action) pairs
        reason_groups = {}
        
        for agent_type, reasons in all_agent_reasons.items():
            agent_weight = self.agent_weights[agent_type]
            
            for reason in reasons:
                key = (reason.ground, reason.action)
                
                if key not in reason_groups:
                    reason_groups[key] = {
                        'ground': reason.ground,
                        'action': reason.action,
                        'justifying_weights': [],
                        'requiring_weights': [],
                        'agent_weights': []
                    }
                
                reason_groups[key]['justifying_weights'].append(reason.justifying_weight)
                reason_groups[key]['requiring_weights'].append(reason.requiring_weight)
                reason_groups[key]['agent_weights'].append(agent_weight)
        
        # Aggregate weights for each reason group
        collective_reasons = []
        
        for key, group in reason_groups.items():
            if self.aggregation_method == "weighted_average":
                # Weighted average based on agent type importance
                total_weight = sum(group['agent_weights'])
                
                agg_jw = sum(jw * aw for jw, aw in zip(group['justifying_weights'], group['agent_weights']))
                agg_rw = sum(rw * aw for rw, aw in zip(group['requiring_weights'], group['agent_weights']))
                
                agg_jw /= total_weight
                agg_rw /= total_weight
                
            elif self.aggregation_method == "maximum":
                # Take maximum weights (most aggressive)
                agg_jw = max(group['justifying_weights'])
                agg_rw = max(group['requiring_weights'])
                
            elif self.aggregation_method == "minimum":
                # Take minimum weights (most conservative)
                agg_jw = min(group['justifying_weights'])
                agg_rw = min(group['requiring_weights'])
                
            elif self.aggregation_method == "simple_average":
                # Simple arithmetic mean
                agg_jw = np.mean(group['justifying_weights'])
                agg_rw = np.mean(group['requiring_weights'])
                
            else:  # Default to weighted average
                total_weight = sum(group['agent_weights'])
                agg_jw = sum(jw * aw for jw, aw in zip(group['justifying_weights'], group['agent_weights'])) / total_weight
                agg_rw = sum(rw * aw for rw, aw in zip(group['requiring_weights'], group['agent_weights'])) / total_weight
            
            collective_reasons.append(TradingReason(
                ground=f"collective_{group['ground']}",
                action=group['action'],
                justifying_weight=agg_jw,
                requiring_weight=agg_rw
            ))
        
        return collective_reasons
    
    def get_individual_decisions(self, data: pd.DataFrame) -> Dict[AgentType, Dict]:
        """Get individual decisions from each agent type for analysis"""
        individual_results = {}
        
        for agent_generator in self.agent_generators:
            # Create temporary agent with this generator
            temp_agent = TradingAgent(reason_generator=agent_generator)
            
            # Get decision
            decision, explanation, debug_info = temp_agent.make_decision(data)
            
            individual_results[agent_generator.get_agent_type()] = {
                'decision': decision,
                'explanation': explanation,
                'debug_info': debug_info
            }
        
        return individual_results

class TradingAgent:
    """Main trading agent using dual scale detachment"""
    
    def __init__(self, reason_generator: ReasonGenerator = None, agent_type: AgentType = None):
        # Initialize reason generator based on agent type or provided generator
        if reason_generator is not None:
            self.reason_generator = reason_generator
        elif agent_type is not None:
            if agent_type == AgentType.RISK_TAKING:
                self.reason_generator = RiskTakingReasonGenerator()
            elif agent_type == AgentType.CONSERVATIVE:
                self.reason_generator = ConservativeReasonGenerator()
            else:
                self.reason_generator = BalancedReasonGenerator()
        else:
            self.reason_generator = BalancedReasonGenerator()  # Default to balanced
            
        self.detachment = DualScaleDetachment()
        self.position_size = 0.0
        self.cash = 100000.0
        self.portfolio_value = 100000.0
        self.agent_type = getattr(self.reason_generator, 'get_agent_type', lambda: AgentType.BALANCED)()
    
    def get_agent_type(self) -> AgentType:
        return self.agent_type
        
    def prepare_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data using simple implementations"""
        data = price_data.copy()
        
        # Technical indicators using simple implementations
        data['RSI'] = simple_rsi(data['close'].values, timeperiod=14)
        data['MA_short'] = simple_sma(data['close'].values, 10)
        data['MA_long'] = simple_sma(data['close'].values, 30)
        
        macd, macd_signal = simple_macd(data['close'].values)
        data['MACD'] = macd
        data['MACD_signal'] = macd_signal
        
        # Volatility (20-day rolling std)
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        return data.dropna()
    
    def create_market_context(self, data: pd.DataFrame) -> MarketContext:
        """Create market context from recent data"""
        latest = data.iloc[-1]
        
        # Use default volatility if NaN
        volatility = latest.get('volatility', 0.02)
        if pd.isna(volatility):
            volatility = 0.02
        
        # Calculate trend from 50-period slope
        if len(data) >= 50:
            recent_prices = data['close'].iloc[-50:].values
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            
            if slope > latest['close'] * 0.001:  # 0.1% slope threshold
                trend = "bull"
            elif slope < -latest['close'] * 0.001:
                trend = "bear" 
            else:
                trend = "sideways"
        else:
            trend = "sideways"
            
        # Normalize volume (assume average volume = 1.0)
        if len(data) >= 20:
            avg_vol = data['volume'].rolling(20).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol <= 0:
                normalized_volume = 1.0
            else:
                normalized_volume = latest['volume'] / avg_vol
        else:
            normalized_volume = 1.0
        
        return MarketContext(
            volatility=volatility,
            volume=normalized_volume,
            trend=trend,
            market_hours=True  # Simplified
        )
    
    def make_decision(self, data: pd.DataFrame) -> Tuple[TradingAction, str, Dict]:
        """Make trading decision using dual scale detachment"""
        
        # Create market context
        context = self.create_market_context(data)
        
        # Generate reasons
        reasons = self.reason_generator.generate_reasons(data, context)
        
        if not reasons:
            return TradingAction.HOLD, "No reasons generated", {}
        
        # Evaluate all pairwise comparisons
        actions = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD]
        results = {}
        
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i != j:
                    v1, v2 = self.detachment.evaluate_options(reasons, action1, action2)
                    results[f"{action1.value}_vs_{action2.value}"] = (v1, v2)
        
        # Decision logic based on dual scale results
        decision, explanation = self._interpret_results(results, reasons)
        
        debug_info = {
            'reasons': [(r.ground, r.action.value, r.justifying_weight, r.requiring_weight) 
                       for r in reasons],
            'context': {
                'volatility': context.volatility,
                'volume': context.volume, 
                'trend': context.trend
            },
            'scale_results': {k: (v1.value, v2.value) for k, (v1, v2) in results.items()}
        }
        
        return decision, explanation, debug_info
    
    def _interpret_results(self, results: Dict, reasons: List[TradingReason]) -> Tuple[TradingAction, str]:
        """Interpret dual scale results to make final decision"""
        
        # Count wins for each action
        action_scores = {action: 0 for action in TradingAction}
        
        for key, (v1, v2) in results.items():
            action1_str, action2_str = key.split('_vs_')
            action1 = TradingAction(action1_str)
            action2 = TradingAction(action2_str)
            
            # v1 positive means action1 is permissible vs action2
            # v2 negative means action2 is not permissible vs action1
            if v1 == ScaleValue.POSITIVE:
                action_scores[action1] += 2
            elif v1 == ScaleValue.NEUTRAL:
                action_scores[action1] += 1
                
            if v2 == ScaleValue.NEGATIVE:
                action_scores[action1] += 1
        
        # Find best action
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Check for ties or weak signals
        sorted_scores = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] == sorted_scores[1][1]:
            return TradingAction.HOLD, "Tie between top actions, holding"
        
        # Agent-specific decision thresholds
        if hasattr(self.reason_generator, 'get_agent_type'):
            agent_type = self.reason_generator.get_agent_type()
            if agent_type == AgentType.RISK_TAKING:
                min_strength_threshold = 1.5  # Lower threshold for risk-taker
                min_score_threshold = 2
            elif agent_type == AgentType.CONSERVATIVE:
                min_strength_threshold = 3.0  # Higher threshold for conservative
                min_score_threshold = 4
            else:
                min_strength_threshold = 2.0  # Balanced
                min_score_threshold = 3
        else:
            min_strength_threshold = 2.0
            min_score_threshold = 3
        
        if sorted_scores[0][1] < min_score_threshold:
            return TradingAction.HOLD, "Weak signal, holding position"
            
        # Check reason strength
        action_reasons = [r for r in reasons if r.action == best_action]
        if not action_reasons:
            return TradingAction.HOLD, "No supporting reasons for best action"
            
        avg_strength = np.mean([r.justifying_weight + r.requiring_weight for r in action_reasons])
        if avg_strength < min_strength_threshold:
            return TradingAction.HOLD, "Insufficient reason strength"
        
        explanation = f"{best_action.value.upper()} signal with {len(action_reasons)} reasons, avg strength {avg_strength:.2f}"
        return best_action, explanation

class CollectiveTradingAgent(TradingAgent):
    """Collective trading agent that combines multiple individual agents"""
    
    def __init__(self, aggregation_method: str = "weighted_average"):
        # Create individual agent generators
        individual_generators = [
            RiskTakingReasonGenerator(),
            BalancedReasonGenerator(), 
            ConservativeReasonGenerator()
        ]
        
        # Create collective reason generator
        collective_generator = CollectiveReasonGenerator(individual_generators, aggregation_method)
        
        # Initialize as TradingAgent with collective generator
        super().__init__(reason_generator=collective_generator)
        self.individual_agents = {
            AgentType.RISK_TAKING: TradingAgent(agent_type=AgentType.RISK_TAKING),
            AgentType.BALANCED: TradingAgent(agent_type=AgentType.BALANCED),
            AgentType.CONSERVATIVE: TradingAgent(agent_type=AgentType.CONSERVATIVE)
        }
        self.aggregation_method = aggregation_method
    
    def make_collective_decision(self, data: pd.DataFrame) -> Tuple[TradingAction, str, Dict]:
        """Make collective decision with detailed individual agent analysis"""
        
        # Get individual decisions
        individual_decisions = {}
        for agent_type, agent in self.individual_agents.items():
            decision, explanation, debug_info = agent.make_decision(data)
            individual_decisions[agent_type] = {
                'decision': decision,
                'explanation': explanation,
                'debug_info': debug_info
            }
        
        # Make collective decision
        collective_decision, collective_explanation, collective_debug = self.make_decision(data)
        
        # Enhanced debug info with individual agent details
        enhanced_debug = collective_debug.copy()
        enhanced_debug['individual_decisions'] = individual_decisions
        enhanced_debug['aggregation_method'] = self.aggregation_method
        
        # Consensus analysis
        decisions = [info['decision'] for info in individual_decisions.values()]
        consensus = self._analyze_consensus(decisions)
        enhanced_debug['consensus_analysis'] = consensus
        
        # Enhanced explanation
        consensus_str = f"Consensus: {consensus['type']}"
        if consensus['type'] == 'unanimous':
            enhanced_explanation = f"COLLECTIVE {collective_decision.value.upper()} - {consensus_str}, all agents agree"
        elif consensus['type'] == 'majority':
            enhanced_explanation = f"COLLECTIVE {collective_decision.value.upper()} - {consensus_str} ({consensus['majority_count']}/3)"
        else:
            enhanced_explanation = f"COLLECTIVE {collective_decision.value.upper()} - {consensus_str}, aggregation decides"
        
        return collective_decision, enhanced_explanation, enhanced_debug
    
    def _analyze_consensus(self, decisions: List[TradingAction]) -> Dict:
        """Analyze consensus among individual agents"""
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        max_count = max(decision_counts.values())
        majority_decisions = [d for d, c in decision_counts.items() if c == max_count]
        
        if len(majority_decisions) == 1 and max_count == len(decisions):
            return {'type': 'unanimous', 'decision': majority_decisions[0], 'count': max_count}
        elif len(majority_decisions) == 1 and max_count >= 2:
            return {'type': 'majority', 'decision': majority_decisions[0], 'majority_count': max_count}
        else:
            return {'type': 'split', 'decisions': decision_counts}

# Enhanced Trading Simulator with Multi-Agent Support
class MultiAgentTradingSimulator:
    """Simple backtesting simulator"""
    
    def __init__(self, initial_capital: float = 100000):
        self.agent = TradingAgent()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0
        self.trades = []
        
    def run_backtest(self, price_data: pd.DataFrame, transaction_cost: float = 0.001):
        """Run backtest on historical data"""
        
        # Prepare data
        data = self.agent.prepare_data(price_data)
        
        portfolio_values = []
        decisions = []
        
        # Need sufficient data for indicators
        start_idx = 50
        
        for i in range(start_idx, len(data)):
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            # Make decision
            decision, explanation, debug_info = self.agent.make_decision(current_data)
            
            # Execute trade
            old_position = self.position
            
            if decision == TradingAction.BUY and self.position < 0.95:
                # Buy signal - increase position
                trade_amount = min(0.1, 0.95 - self.position)  # 10% of capital or to max position
                shares_to_buy = (trade_amount * self.capital) / current_price
                cost = shares_to_buy * current_price * (1 + transaction_cost)
                
                if self.capital >= cost:
                    self.position += trade_amount
                    self.capital -= cost
                    self.trades.append({
                        'date': current_data.index[-1],
                        'action': 'BUY',
                        'price': current_price,
                        'amount': trade_amount,
                        'explanation': explanation
                    })
                    
            elif decision == TradingAction.SELL and self.position > -0.95:
                # Sell signal - decrease position
                trade_amount = min(0.1, self.position + 0.95)  # 10% of capital or to min position
                if trade_amount > 0:
                    shares_to_sell = (trade_amount * self.capital) / current_price
                    proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                    
                    self.position -= trade_amount
                    self.capital += proceeds
                    self.trades.append({
                        'date': current_data.index[-1],
                        'action': 'SELL', 
                        'price': current_price,
                        'amount': trade_amount,
                        'explanation': explanation
                    })
            
            # Calculate portfolio value
            position_value = self.position * self.capital * current_price / self.initial_capital
            portfolio_value = self.capital + position_value
            portfolio_values.append(portfolio_value)
            
            decisions.append({
                'date': current_data.index[-1],
                'decision': decision.value,
                'explanation': explanation,
                'portfolio_value': portfolio_value,
                'position': self.position,
                'debug_info': debug_info
            })
        
        return {
            'portfolio_values': portfolio_values,
            'decisions': decisions,
            'trades': self.trades,
            'final_return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trades)
        }

# Example data generation for testing
def generate_sample_data(days: int = 1000) -> pd.DataFrame:
    """Generate sample price data for testing"""
    
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    
    # Random walk with trend
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # 0.05% daily return, 2% volatility
    
    # Add some trend and cycles
    trend = np.linspace(0, 0.001, days)  # Slight upward trend
    cycle = 0.0002 * np.sin(np.linspace(0, 4*np.pi, days))  # Some cyclical behavior
    returns += trend + cycle
    
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
    
    # Generate volume (correlated with volatility)
    volume = np.random.lognormal(12, 0.5, days)  # Log-normal volume
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, days)),
        'high': prices * (1 + np.abs(np.random.normal(0.001, 0.002, days))),
        'low': prices * (1 - np.abs(np.random.normal(0.001, 0.002, days))),
        'close': prices,
        'volume': volume
    }, index=dates)

if __name__ == "__main__":
    # Example usage
    print("Dual Scale Detachment Trading Agent")
    print("=" * 40)
    
    # Generate sample data
    print("Generating sample market data...")
    sample_data = generate_sample_data(500)
    
    # Run backtest
    print("Running backtest...")
    simulator = TradingSimulator()
    results = simulator.run_backtest(sample_data)
    
    # Print results
    print(f"\nBacktest Results:")
    print(f"Final Return: {results['final_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Final Portfolio Value: ${results['portfolio_values'][-1]:,.2f}")
    
    print(f"\nLast 5 Decisions:")
    for decision in results['decisions'][-5:]:
        print(f"{decision['date'].strftime('%Y-%m-%d')}: {decision['decision']} - {decision['explanation']}")
    
    print(f"\nLast 3 Trades:")
    for trade in results['trades'][-3:]:
        print(f"{trade['date'].strftime('%Y-%m-%d')}: {trade['action']} at ${trade['price']:.2f} - {trade['explanation']}")