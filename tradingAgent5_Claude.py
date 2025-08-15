import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
from dataclasses import dataclass
import talib
from abc import ABC, abstractmethod

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

class TradingAgent:
    """Main trading agent using dual scale detachment"""
    
    def __init__(self):
        self.reason_generator = ReasonGenerator()
        self.detachment = DualScaleDetachment()
        self.position_size = 0.0  # Current position (-1 to 1, negative = short)
        self.cash = 100000.0
        self.portfolio_value = 100000.0
        
    def prepare_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        data = price_data.copy()
        
        # Technical indicators
        data['RSI'] = talib.RSI(data['close'].values, timeperiod=14)
        data['MA_short'] = talib.SMA(data['close'].values, timeperiod=10)
        data['MA_long'] = talib.SMA(data['close'].values, timeperiod=30)
        
        macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
        data['MACD'] = macd
        data['MACD_signal'] = macd_signal
        
        # Volatility (20-day rolling std)
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        return data.dropna()
    
    def create_market_context(self, data: pd.DataFrame) -> MarketContext:
        """Create market context from recent data"""
        latest = data.iloc[-1]
        
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
        avg_vol = data['volume'].rolling(20).mean().iloc[-1]
        normalized_volume = latest['volume'] / avg_vol if avg_vol > 0 else 1.0
        
        return MarketContext(
            volatility=latest['volatility'],
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
        
        if sorted_scores[0][1] < 3:  # Weak signal threshold
            return TradingAction.HOLD, "Weak signal, holding position"
            
        # Check reason strength
        action_reasons = [r for r in reasons if r.action == best_action]
        if not action_reasons:
            return TradingAction.HOLD, "No supporting reasons for best action"
            
        avg_strength = np.mean([r.justifying_weight + r.requiring_weight for r in action_reasons])
        if avg_strength < 2.0:  # Minimum strength threshold
            return TradingAction.HOLD, "Insufficient reason strength"
        
        explanation = f"{best_action.value.upper()} signal with {len(action_reasons)} reasons, avg strength {avg_strength:.2f}"
        return best_action, explanation

# Example usage and backtesting framework
class TradingSimulator:
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