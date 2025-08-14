import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

# Define the complete set of OPTIONS (decisions the agent can make)
class TradingOption(Enum):
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    STOP_TRADING = "stop_trading"

# Define the set of GROUNDS (reasons for decisions)
class Ground(Enum):
    # Market Technical Grounds
    BULLISH_SIGNAL = "bullish_signal"
    BEARISH_SIGNAL = "bearish_signal" 
    OVERSOLD_CONDITION = "oversold_condition"
    OVERBOUGHT_CONDITION = "overbought_condition"
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    
    # Risk Management Grounds  
    PROFIT_TARGET_REACHED = "profit_target_reached"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    
    # Portfolio Grounds
    DIVERSIFICATION_NEEDED = "diversification_needed"
    CONCENTRATION_RISK = "concentration_risk"
    GOOD_RISK_REWARD = "good_risk_reward"
    POOR_RISK_REWARD = "poor_risk_reward"

@dataclass
class Decision:
    """A decision with its supporting grounds"""
    option: TradingOption
    grounds: List[Ground]
    confidence: float  # 0.0 to 1.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TraderPersonality(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    AGGRESSIVE = "aggressive"

class GroundsBasedAgent:
    def __init__(self, initial_balance=10000, personality=TraderPersonality.BALANCED):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.personality = personality
        self.trade_history = []
        self.decision_history = []  # Track all decisions with grounds
        
        # Personality-based thresholds
        self._set_personality_parameters()
        
    def _set_personality_parameters(self):
        """Set decision thresholds based on personality"""
        params = {
            TraderPersonality.CONSERVATIVE: {
                'position_size': 0.05,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'volatility_threshold': 0.15,
                'confidence_threshold': 0.8,
                'drawdown_limit': 0.10,
                'profit_target': 0.05,
                'stop_loss': 0.02
            },
            TraderPersonality.BALANCED: {
                'position_size': 0.10,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volatility_threshold': 0.25,
                'confidence_threshold': 0.6,
                'drawdown_limit': 0.20,
                'profit_target': 0.10,
                'stop_loss': 0.05
            },
            TraderPersonality.AGGRESSIVE: {
                'position_size': 0.25,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'volatility_threshold': 0.40,
                'confidence_threshold': 0.4,
                'drawdown_limit': 0.35,
                'profit_target': 0.20,
                'stop_loss': 0.08
            }
        }
        self.params = params[self.personality]
        
    def assess_grounds(self, market_data: pd.Series, symbol: str) -> List[Tuple[Ground, float]]:
        """
        Assess all possible grounds and their strength (0.0 to 1.0)
        Returns list of (Ground, strength) tuples
        """
        grounds_assessment = []
        
        # Get current market indicators
        current_price = market_data['Close']
        sma_5 = market_data.get('SMA_5', 0)
        sma_20 = market_data.get('SMA_20', 0)
        rsi = market_data.get('RSI', 50)
        volatility = market_data.get('Volatility', 0)
        
        # Technical Analysis Grounds
        if sma_5 > sma_20 and sma_5 > 0:
            strength = min((sma_5 - sma_20) / sma_20 * 10, 1.0)  # Normalize
            grounds_assessment.append((Ground.BULLISH_SIGNAL, strength))
        elif sma_5 < sma_20 and sma_5 > 0:
            strength = min((sma_20 - sma_5) / sma_20 * 10, 1.0)
            grounds_assessment.append((Ground.BEARISH_SIGNAL, strength))
            
        if rsi < self.params['rsi_oversold']:
            strength = (self.params['rsi_oversold'] - rsi) / self.params['rsi_oversold']
            grounds_assessment.append((Ground.OVERSOLD_CONDITION, strength))
        elif rsi > self.params['rsi_overbought']:
            strength = (rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought'])
            grounds_assessment.append((Ground.OVERBOUGHT_CONDITION, strength))
            
        # Volatility Grounds
        if volatility > self.params['volatility_threshold']:
            strength = min(volatility / self.params['volatility_threshold'] - 1, 1.0)
            grounds_assessment.append((Ground.HIGH_VOLATILITY, strength))
        elif volatility < self.params['volatility_threshold'] * 0.5:
            strength = 1 - (volatility / (self.params['volatility_threshold'] * 0.5))
            grounds_assessment.append((Ground.LOW_VOLATILITY, strength))
            
        # Risk Management Grounds
        current_position = self.positions.get(symbol, 0)
        if current_position > 0:
            # Check profit/loss conditions
            last_buy_price = self._get_last_entry_price(symbol)
            if last_buy_price:
                price_change = (current_price - last_buy_price) / last_buy_price
                
                if price_change >= self.params['profit_target']:
                    strength = min(price_change / self.params['profit_target'], 1.0)
                    grounds_assessment.append((Ground.PROFIT_TARGET_REACHED, strength))
                elif price_change <= -self.params['stop_loss']:
                    strength = min(abs(price_change) / self.params['stop_loss'], 1.0)
                    grounds_assessment.append((Ground.STOP_LOSS_TRIGGERED, strength))
        
        # Portfolio-level grounds
        portfolio_value = self._calculate_portfolio_value({symbol: current_price})
        drawdown = self._calculate_drawdown()
        
        if drawdown > self.params['drawdown_limit']:
            strength = min(drawdown / self.params['drawdown_limit'], 1.0)
            grounds_assessment.append((Ground.EXCESSIVE_DRAWDOWN, strength))
            
        if self.balance < self.initial_balance * 0.1:  # Less than 10% cash
            grounds_assessment.append((Ground.INSUFFICIENT_CAPITAL, 1.0))
            
        # Position concentration
        position_count = sum(1 for qty in self.positions.values() if qty > 0)
        if position_count >= 5:  # Arbitrary limit
            grounds_assessment.append((Ground.POSITION_LIMIT_REACHED, 1.0))
            
        return grounds_assessment
    
    def generate_decision_options(self, grounds: List[Tuple[Ground, float]], symbol: str) -> List[Decision]:
        """
        Generate possible decisions based on assessed grounds
        """
        decisions = []
        current_position = self.positions.get(symbol, 0)
        
        # Convert grounds to dict for easier access
        grounds_dict = {ground: strength for ground, strength in grounds}
        
        # BUY Decision Logic
        buy_grounds = []
        buy_confidence = 0.0
        
        if Ground.BULLISH_SIGNAL in grounds_dict:
            buy_grounds.append(Ground.BULLISH_SIGNAL)
            buy_confidence += grounds_dict[Ground.BULLISH_SIGNAL] * 0.4
            
        if Ground.OVERSOLD_CONDITION in grounds_dict:
            buy_grounds.append(Ground.OVERSOLD_CONDITION)
            buy_confidence += grounds_dict[Ground.OVERSOLD_CONDITION] * 0.3
            
        if Ground.LOW_VOLATILITY in grounds_dict and self.personality == TraderPersonality.CONSERVATIVE:
            buy_grounds.append(Ground.LOW_VOLATILITY)
            buy_confidence += grounds_dict[Ground.LOW_VOLATILITY] * 0.2
        elif Ground.HIGH_VOLATILITY in grounds_dict and self.personality == TraderPersonality.AGGRESSIVE:
            buy_grounds.append(Ground.HIGH_VOLATILITY)
            buy_confidence += grounds_dict[Ground.HIGH_VOLATILITY] * 0.2
            
        # Reduce confidence based on negative grounds
        if Ground.INSUFFICIENT_CAPITAL in grounds_dict:
            buy_confidence *= 0.1
        if Ground.POSITION_LIMIT_REACHED in grounds_dict:
            buy_confidence *= 0.1
        if Ground.EXCESSIVE_DRAWDOWN in grounds_dict:
            buy_confidence *= 0.5
            
        if buy_grounds and buy_confidence > self.params['confidence_threshold'] and current_position == 0:
            decisions.append(Decision(
                option=TradingOption.BUY,
                grounds=buy_grounds,
                confidence=min(buy_confidence, 1.0),
                metadata={'symbol': symbol}
            ))
        
        # SELL Decision Logic  
        sell_grounds = []
        sell_confidence = 0.0
        
        if Ground.BEARISH_SIGNAL in grounds_dict:
            sell_grounds.append(Ground.BEARISH_SIGNAL)
            sell_confidence += grounds_dict[Ground.BEARISH_SIGNAL] * 0.4
            
        if Ground.OVERBOUGHT_CONDITION in grounds_dict:
            sell_grounds.append(Ground.OVERBOUGHT_CONDITION)
            sell_confidence += grounds_dict[Ground.OVERBOUGHT_CONDITION] * 0.3
            
        if Ground.PROFIT_TARGET_REACHED in grounds_dict:
            sell_grounds.append(Ground.PROFIT_TARGET_REACHED)
            sell_confidence += grounds_dict[Ground.PROFIT_TARGET_REACHED] * 0.5
            
        if Ground.STOP_LOSS_TRIGGERED in grounds_dict:
            sell_grounds.append(Ground.STOP_LOSS_TRIGGERED)
            sell_confidence = 1.0  # Always sell on stop loss
            
        if sell_grounds and current_position > 0:
            decisions.append(Decision(
                option=TradingOption.SELL,
                grounds=sell_grounds, 
                confidence=min(sell_confidence, 1.0),
                metadata={'symbol': symbol}
            ))
        
        # HOLD Decision (always an option)
        hold_grounds = []
        hold_confidence = 1.0 - max([d.confidence for d in decisions], default=0)
        
        if not buy_grounds and not sell_grounds:
            hold_grounds.append(Ground.WEAK_TREND)
            hold_confidence = 0.8
            
        decisions.append(Decision(
            option=TradingOption.HOLD,
            grounds=hold_grounds if hold_grounds else [Ground.WEAK_TREND],
            confidence=hold_confidence,
            metadata={'symbol': symbol}
        ))
        
        # STOP_TRADING Decision
        if Ground.EXCESSIVE_DRAWDOWN in grounds_dict:
            decisions.append(Decision(
                option=TradingOption.STOP_TRADING,
                grounds=[Ground.EXCESSIVE_DRAWDOWN],
                confidence=grounds_dict[Ground.EXCESSIVE_DRAWDOWN],
                metadata={'reason': 'Risk limit exceeded'}
            ))
        
        return decisions
    
    def select_best_decision(self, decisions: List[Decision]) -> Decision:
        """
        Select the best decision from available options
        Can implement different selection strategies here
        """
        if not decisions:
            return Decision(TradingOption.HOLD, [], 0.0)
            
        # For now, select highest confidence decision
        # Could implement more sophisticated selection logic
        best_decision = max(decisions, key=lambda d: d.confidence)
        
        # Override with critical decisions
        critical_decisions = [d for d in decisions if d.option == TradingOption.STOP_TRADING]
        if critical_decisions:
            best_decision = critical_decisions[0]
            
        return best_decision
    
    def execute_decision(self, decision: Decision, market_data: pd.Series) -> bool:
        """Execute the selected decision"""
        symbol = decision.metadata.get('symbol', 'UNKNOWN')
        price = market_data['Close']
        
        success = False
        
        if decision.option == TradingOption.BUY:
            success = self._execute_buy(symbol, price, decision)
        elif decision.option == TradingOption.SELL:
            success = self._execute_sell(symbol, price, decision)
        elif decision.option == TradingOption.HOLD:
            success = True  # Always successful
        elif decision.option == TradingOption.STOP_TRADING:
            print(f"🛑 TRADING HALTED: {decision.grounds}")
            success = True
            
        # Log the decision
        self.decision_history.append({
            'timestamp': market_data.name,
            'option': decision.option.value,
            'grounds': [g.value for g in decision.grounds],
            'confidence': decision.confidence,
            'executed': success
        })
        
        return success
    
    def _execute_buy(self, symbol: str, price: float, decision: Decision) -> bool:
        """Execute buy order"""
        position_value = self.balance * self.params['position_size']
        shares = position_value / price
        cost = shares * price
        
        if cost <= self.balance:
            self.balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            
            self.trade_history.append({
                'timestamp': decision.metadata.get('timestamp', 'unknown'),
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'grounds': [g.value for g in decision.grounds],
                'confidence': decision.confidence
            })
            
            print(f"📈 BUY {symbol}: {shares:.2f} shares @ ${price:.2f}")
            print(f"   Grounds: {[g.value for g in decision.grounds]}")
            print(f"   Confidence: {decision.confidence:.2%}")
            return True
        return False
    
    def _execute_sell(self, symbol: str, price: float, decision: Decision) -> bool:
        """Execute sell order"""
        if symbol in self.positions and self.positions[symbol] > 0:
            shares = self.positions[symbol]
            revenue = shares * price
            
            self.balance += revenue
            self.positions[symbol] = 0
            
            self.trade_history.append({
                'timestamp': decision.metadata.get('timestamp', 'unknown'),
                'action': 'SELL',
                'symbol': symbol, 
                'shares': shares,
                'price': price,
                'grounds': [g.value for g in decision.grounds],
                'confidence': decision.confidence
            })
            
            print(f"📉 SELL {symbol}: {shares:.2f} shares @ ${price:.2f}")
            print(f"   Grounds: {[g.value for g in decision.grounds]}")
            print(f"   Confidence: {decision.confidence:.2%}")
            return True
        return False
    
    def _get_last_entry_price(self, symbol: str) -> Optional[float]:
        """Get the last buy price for a symbol"""
        buy_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['action'] == 'BUY']
        return buy_trades[-1]['price'] if buy_trades else None
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        value = self.balance
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                value += shares * current_prices[symbol]
        return value
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        # Simplified drawdown calculation
        current_value = self.balance + sum(self.positions.values()) * 100  # Rough estimate
        peak_value = max(self.initial_balance, current_value)
        return (peak_value - current_value) / peak_value if peak_value > 0 else 0
    
    def make_decision(self, market_data: pd.Series, symbol: str) -> Decision:
        """
        Main decision-making process:
        1. Assess all grounds
        2. Generate possible decisions
        3. Select best decision
        4. Execute decision
        """
        # Step 1: Assess grounds
        grounds = self.assess_grounds(market_data, symbol)
        
        # Step 2: Generate decision options
        decisions = self.generate_decision_options(grounds, symbol)
        
        # Step 3: Select best decision
        best_decision = self.select_best_decision(decisions)
        
        # Step 4: Execute decision
        best_decision.metadata['timestamp'] = market_data.name
        self.execute_decision(best_decision, market_data)
        
        return best_decision
    
    def get_decision_summary(self) -> pd.DataFrame:
        """Get summary of all decisions made"""
        return pd.DataFrame(self.decision_history)
    
    def print_reasoning_trace(self, limit: int = 5):
        """Print recent decision reasoning"""
        print(f"\n🧠 Recent Decision Reasoning (Last {limit}):")
        print("=" * 60)
        
        recent_decisions = self.decision_history[-limit:]
        for i, decision in enumerate(recent_decisions, 1):
            print(f"{i}. {decision['option'].upper()}")
            print(f"   Grounds: {decision['grounds']}")
            print(f"   Confidence: {decision['confidence']:.2%}")
            print(f"   Executed: {'✅' if decision['executed'] else '❌'}")
            print()

# Helper function to add technical indicators
def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to market data"""
    data = data.copy()
    
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    return data

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    prices = [100]
    for _ in range(len(dates)-1):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Create agents with different personalities
    conservative_agent = GroundsBasedAgent(10000, TraderPersonality.CONSERVATIVE)
    aggressive_agent = GroundsBasedAgent(10000, TraderPersonality.AGGRESSIVE)
    
    print("🤖 Grounds-Based Trading Simulation")
    print("=" * 50)
    
    # Run simulation for last 30 days
    for date in data.index[-30:]:
        market_data = data.loc[date]
        
        print(f"\n📅 {date.strftime('%Y-%m-%d')} - Price: ${market_data['Close']:.2f}")
        print("-" * 40)
        
        # Conservative agent decision
        print("🛡️ CONSERVATIVE AGENT:")
        conservative_decision = conservative_agent.make_decision(market_data, 'SAMPLE')
        
        print("\n🚀 AGGRESSIVE AGENT:")  
        aggressive_decision = aggressive_agent.make_decision(market_data, 'SAMPLE')
    
    # Show decision reasoning
    print("\n" + "="*60)
    print("CONSERVATIVE AGENT REASONING:")
    conservative_agent.print_reasoning_trace()
    
    print("AGGRESSIVE AGENT REASONING:")
    aggressive_agent.print_reasoning_trace()
