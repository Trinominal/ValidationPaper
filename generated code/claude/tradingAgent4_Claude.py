import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

# TRADING OPTIONS (O) - Complete set of decisions
class TradingOption(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    STOP_TRADING = "stop_trading"

# GROUNDS (G) - Facts that serve as reasons
class Ground(Enum):
    # Technical Analysis Grounds
    BULLISH_CROSSOVER = "bullish_crossover"
    BEARISH_CROSSOVER = "bearish_crossover"
    OVERSOLD_RSI = "oversold_rsi"
    OVERBOUGHT_RSI = "overbought_rsi"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    
    # Risk Management Grounds
    PROFIT_TARGET_HIT = "profit_target_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    
    # Portfolio Grounds
    GOOD_RISK_REWARD_RATIO = "good_risk_reward_ratio"
    POOR_RISK_REWARD_RATIO = "poor_risk_reward_ratio"
    CONCENTRATION_RISK = "concentration_risk"
    DIVERSIFICATION_OPPORTUNITY = "diversification_opportunity"

# DEONTIC STATUS VALUES (V)
class DeonticValue(Enum):
    OBLIGATORY = "+"      # Must do this action
    FORBIDDEN = "-"       # Must not do this action  
    PERMISSIBLE = "0"     # May do this action

@dataclass
class Reason:
    """A reason connecting a ground to an option"""
    ground: Ground
    option: TradingOption
    justifying_weight: float  # How good at making option permissible
    requiring_weight: float   # How good at making option obligatory
    
    def __str__(self):
        return f"({self.ground.value} → {self.option.value}: JW={self.justifying_weight:.2f}, RW={self.requiring_weight:.2f})"

class TraderPersonality(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class DualScaleTradingAgent:
    """Trading agent using dual scale detachment for deontic reasoning"""
    
    def __init__(self, initial_balance=10000, personality=TraderPersonality.BALANCED):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.personality = personality
        self.trade_history = []
        self.decision_history = []
        
        # Personality affects how we assign weights to reasons
        self._set_personality_parameters()
        
    def _set_personality_parameters(self):
        """Set weight assignment parameters based on personality"""
        params = {
            TraderPersonality.CONSERVATIVE: {
                'risk_aversion': 0.8,          # High weight on risk grounds
                'trend_sensitivity': 0.6,      # Moderate trend following
                'volatility_penalty': 0.9,     # Strongly avoid volatility
                'profit_taking_urgency': 0.9,  # Quick to take profits
                'stop_loss_urgency': 0.95,     # Very quick to cut losses
            },
            TraderPersonality.BALANCED: {
                'risk_aversion': 0.5,
                'trend_sensitivity': 0.7,
                'volatility_penalty': 0.5,
                'profit_taking_urgency': 0.6,
                'stop_loss_urgency': 0.8,
            },
            TraderPersonality.AGGRESSIVE: {
                'risk_aversion': 0.2,          # Low weight on risk grounds
                'trend_sensitivity': 0.9,      # Strong trend following
                'volatility_penalty': 0.1,     # Embrace volatility
                'profit_taking_urgency': 0.3,  # Hold for bigger gains
                'stop_loss_urgency': 0.6,      # Wider stop losses
            }
        }
        self.params = params[self.personality]
    
    def assess_market_grounds(self, market_data: pd.Series, symbol: str) -> List[Tuple[Ground, float]]:
        """Assess which grounds (facts) exist and their strength"""
        grounds_present = []
        
        # Technical indicators
        current_price = market_data['Close']
        sma_5 = market_data.get('SMA_5', 0)
        sma_20 = market_data.get('SMA_20', 0)
        rsi = market_data.get('RSI', 50)
        volatility = market_data.get('Volatility', 0)
        
        # Previous values for crossover detection
        prev_sma_5 = market_data.get('SMA_5_prev', sma_5)
        prev_sma_20 = market_data.get('SMA_20_prev', sma_20)
        
        # Technical Analysis Grounds
        if sma_5 > sma_20 and prev_sma_5 <= prev_sma_20:
            strength = min((sma_5 - sma_20) / sma_20 * 20, 1.0)
            grounds_present.append((Ground.BULLISH_CROSSOVER, strength))
        elif sma_5 < sma_20 and prev_sma_5 >= prev_sma_20:
            strength = min((sma_20 - sma_5) / sma_20 * 20, 1.0)
            grounds_present.append((Ground.BEARISH_CROSSOVER, strength))
            
        if rsi < 30:
            strength = (30 - rsi) / 30
            grounds_present.append((Ground.OVERSOLD_RSI, strength))
        elif rsi > 70:
            strength = (rsi - 70) / 30
            grounds_present.append((Ground.OVERBOUGHT_RSI, strength))
            
        # Volatility grounds
        if volatility > 0.03:  # High volatility threshold
            strength = min(volatility / 0.05, 1.0)
            grounds_present.append((Ground.HIGH_VOLATILITY, strength))
        elif volatility < 0.01:  # Low volatility threshold
            strength = 1 - (volatility / 0.01)
            grounds_present.append((Ground.LOW_VOLATILITY, strength))
            
        # Risk Management Grounds
        current_position = self.positions.get(symbol, 0)
        if current_position > 0:
            last_buy_price = self._get_last_entry_price(symbol)
            if last_buy_price:
                price_change = (current_price - last_buy_price) / last_buy_price
                
                if price_change >= 0.05:  # 5% profit target
                    strength = min(price_change / 0.10, 1.0)
                    grounds_present.append((Ground.PROFIT_TARGET_HIT, strength))
                elif price_change <= -0.03:  # 3% stop loss
                    strength = min(abs(price_change) / 0.05, 1.0)
                    grounds_present.append((Ground.STOP_LOSS_HIT, strength))
        
        # Portfolio level grounds
        portfolio_value = self._calculate_portfolio_value({symbol: current_price})
        drawdown = self._calculate_drawdown()
        
        if drawdown > 0.15:  # 15% drawdown
            strength = min(drawdown / 0.25, 1.0)
            grounds_present.append((Ground.EXCESSIVE_DRAWDOWN, strength))
            
        if self.balance < self.initial_balance * 0.2:
            strength = 1 - (self.balance / (self.initial_balance * 0.2))
            grounds_present.append((Ground.INSUFFICIENT_CAPITAL, strength))
            
        return grounds_present
    
    def generate_reasons(self, grounds: List[Tuple[Ground, float]], symbol: str) -> List[Reason]:
        """Convert grounds into reasons (ground → option with weights)"""
        reasons = []
        
        for ground, strength in grounds:
            # Generate reasons based on ground type and personality
            if ground == Ground.BULLISH_CROSSOVER:
                # Strong bullish signal supports buying
                jw = strength * self.params['trend_sensitivity']  # Justifies buying
                rw = strength * self.params['trend_sensitivity'] * 0.8  # May require buying
                reasons.append(Reason(ground, TradingOption.BUY, jw, rw))
                
                # Also supports not selling if we have position
                if self.positions.get(symbol, 0) > 0:
                    reasons.append(Reason(ground, TradingOption.SELL, 0.0, strength * 0.3))  # Against selling
                    
            elif ground == Ground.BEARISH_CROSSOVER:
                # Bearish signal supports selling
                jw = strength * self.params['trend_sensitivity']
                rw = strength * self.params['trend_sensitivity'] * 0.9
                reasons.append(Reason(ground, TradingOption.SELL, jw, rw))
                
                # Against buying
                reasons.append(Reason(ground, TradingOption.BUY, 0.0, strength * 0.8))
                
            elif ground == Ground.OVERSOLD_RSI:
                # Oversold supports buying (mean reversion)
                jw = strength * (1 - self.params['trend_sensitivity'] * 0.3)  # Counter-trend
                rw = strength * 0.5  # Moderate obligation
                reasons.append(Reason(ground, TradingOption.BUY, jw, rw))
                
            elif ground == Ground.OVERBOUGHT_RSI:
                # Overbought supports selling
                jw = strength * (1 - self.params['trend_sensitivity'] * 0.3)
                rw = strength * 0.6
                reasons.append(Reason(ground, TradingOption.SELL, jw, rw))
                
            elif ground == Ground.HIGH_VOLATILITY:
                # High volatility affects based on personality
                if self.params['volatility_penalty'] > 0.7:  # Conservative
                    # Against any action (prefer holding)
                    penalty = strength * self.params['volatility_penalty']
                    reasons.append(Reason(ground, TradingOption.BUY, 0.0, penalty))
                    reasons.append(Reason(ground, TradingOption.SELL, 0.0, penalty * 0.5))
                else:  # Aggressive - opportunity!
                    bonus = strength * (1 - self.params['volatility_penalty'])
                    reasons.append(Reason(ground, TradingOption.BUY, bonus, bonus * 0.3))
                    
            elif ground == Ground.PROFIT_TARGET_HIT:
                # Strong obligation to sell at profit
                jw = strength * self.params['profit_taking_urgency']
                rw = strength * self.params['profit_taking_urgency']
                reasons.append(Reason(ground, TradingOption.SELL, jw, rw))
                
            elif ground == Ground.STOP_LOSS_HIT:
                # Very strong obligation to sell at loss
                jw = strength * self.params['stop_loss_urgency']
                rw = strength * self.params['stop_loss_urgency']
                reasons.append(Reason(ground, TradingOption.SELL, jw, rw))
                
            elif ground == Ground.EXCESSIVE_DRAWDOWN:
                # Stop all trading
                jw = strength
                rw = strength
                reasons.append(Reason(ground, TradingOption.STOP_TRADING, jw, rw))
                # Against new positions
                reasons.append(Reason(ground, TradingOption.BUY, 0.0, strength))
                
            elif ground == Ground.INSUFFICIENT_CAPITAL:
                # Can't buy more
                reasons.append(Reason(ground, TradingOption.BUY, 0.0, strength))
                
        return reasons
    
    def dual_scale_detachment(self, reasons: List[Reason], option1: TradingOption, option2: TradingOption) -> Tuple[DeonticValue, DeonticValue]:
        """
        Apply dual scale detachment to determine deontic status of two options
        
        For option1: Compare justifying weights supporting option1 vs requiring weights supporting option2
        For option2: Compare justifying weights supporting option2 vs requiring weights supporting option1
        """
        # Separate reasons by which option they support
        reasons_for_o1 = [r for r in reasons if r.option == option1]
        reasons_for_o2 = [r for r in reasons if r.option == option2]
        
        # Calculate aggregate weights
        jw_o1 = sum(r.justifying_weight for r in reasons_for_o1)  # Justifying weight for option1
        rw_o1 = sum(r.requiring_weight for r in reasons_for_o1)   # Requiring weight for option1
        jw_o2 = sum(r.justifying_weight for r in reasons_for_o2)  # Justifying weight for option2  
        rw_o2 = sum(r.requiring_weight for r in reasons_for_o2)   # Requiring weight for option2
        
        # Permission scale for option1: JW(o1) vs RW(o2)
        if jw_o1 > rw_o2:
            v1 = DeonticValue.OBLIGATORY if rw_o1 > jw_o2 else DeonticValue.PERMISSIBLE
        elif jw_o1 < rw_o2:
            v1 = DeonticValue.FORBIDDEN
        else:
            v1 = DeonticValue.PERMISSIBLE
            
        # Permission scale for option2: JW(o2) vs RW(o1) 
        if jw_o2 > rw_o1:
            v2 = DeonticValue.OBLIGATORY if rw_o2 > jw_o1 else DeonticValue.PERMISSIBLE
        elif jw_o2 < rw_o1:
            v2 = DeonticValue.FORBIDDEN
        else:
            v2 = DeonticValue.PERMISSIBLE
            
        return v1, v2
    
    def determine_best_action(self, market_data: pd.Series, symbol: str) -> Tuple[TradingOption, str]:
        """
        Main decision process using dual scale detachment:
        1. Assess grounds (facts about market)
        2. Generate reasons (grounds → options with dual weights)  
        3. Compare options pairwise using dual scale detachment
        4. Select action based on deontic status
        """
        # Step 1: Assess what grounds exist
        grounds = self.assess_market_grounds(market_data, symbol)
        
        # Step 2: Generate reasons from grounds
        reasons = self.generate_reasons(grounds, symbol)
        
        # Step 3: Pairwise comparisons using dual scale detachment
        # Primary comparison: BUY vs SELL
        buy_status, sell_status = self.dual_scale_detachment(reasons, TradingOption.BUY, TradingOption.SELL)
        
        # Secondary comparisons
        hold_vs_buy = self.dual_scale_detachment(reasons, TradingOption.HOLD, TradingOption.BUY)
        hold_vs_sell = self.dual_scale_detachment(reasons, TradingOption.HOLD, TradingOption.SELL)
        
        # Check for stop trading
        stop_reasons = [r for r in reasons if r.option == TradingOption.STOP_TRADING]
        if stop_reasons and sum(r.requiring_weight for r in stop_reasons) > 0.8:
            return TradingOption.STOP_TRADING, "Excessive risk - halting trading"
        
        # Step 4: Select action based on deontic status
        reasoning = f"Grounds: {[g.value for g, _ in grounds]}\n"
        reasoning += f"BUY: {buy_status.value}, SELL: {sell_status.value}\n"
        reasoning += f"Active reasons: {len(reasons)}"
        
        current_position = self.positions.get(symbol, 0)
        
        # Decision logic based on deontic status
        if buy_status == DeonticValue.OBLIGATORY:
            return TradingOption.BUY, f"BUY obligatory. {reasoning}"
        elif sell_status == DeonticValue.OBLIGATORY and current_position > 0:
            return TradingOption.SELL, f"SELL obligatory. {reasoning}"
        elif buy_status == DeonticValue.PERMISSIBLE and sell_status == DeonticValue.FORBIDDEN:
            return TradingOption.BUY, f"BUY permissible, SELL forbidden. {reasoning}"
        elif sell_status == DeonticValue.PERMISSIBLE and buy_status == DeonticValue.FORBIDDEN and current_position > 0:
            return TradingOption.SELL, f"SELL permissible, BUY forbidden. {reasoning}"
        else:
            return TradingOption.HOLD, f"No clear obligation. {reasoning}"
    
    def execute_decision(self, action: TradingOption, market_data: pd.Series, symbol: str, reasoning: str) -> bool:
        """Execute the decided action"""
        price = market_data['Close']
        success = False
        
        if action == TradingOption.BUY:
            success = self._execute_buy(symbol, price, reasoning)
        elif action == TradingOption.SELL:
            success = self._execute_sell(symbol, price, reasoning)
        elif action == TradingOption.HOLD:
            success = True
        elif action == TradingOption.STOP_TRADING:
            print(f"🛑 TRADING HALTED: {reasoning}")
            success = True
            
        # Log decision with deontic reasoning
        self.decision_history.append({
            'timestamp': market_data.name,
            'action': action.value,
            'reasoning': reasoning,
            'executed': success,
            'personality': self.personality.value
        })
        
        return success
    
    def _execute_buy(self, symbol: str, price: float, reasoning: str) -> bool:
        """Execute buy order"""
        position_value = self.balance * 0.1  # 10% of balance
        shares = position_value / price
        cost = shares * price
        
        if cost <= self.balance:
            self.balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            
            print(f"📈 BUY {symbol}: {shares:.2f} shares @ ${price:.2f}")
            print(f"   Deontic Reasoning: {reasoning.split('.')[0]}")
            
            self.trade_history.append({
                'timestamp': 'current',
                'action': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'reasoning': reasoning
            })
            return True
        return False
    
    def _execute_sell(self, symbol: str, price: float, reasoning: str) -> bool:
        """Execute sell order"""
        if symbol in self.positions and self.positions[symbol] > 0:
            shares = self.positions[symbol]
            revenue = shares * price
            
            # Calculate P&L
            avg_cost = self._get_average_cost(symbol)
            pnl_pct = ((price - avg_cost) / avg_cost) if avg_cost else 0
            pnl_symbol = "📈" if pnl_pct > 0 else "📉"
            
            self.balance += revenue
            self.positions[symbol] = 0
            
            print(f"{pnl_symbol} SELL {symbol}: {shares:.2f} shares @ ${price:.2f} (P&L: {pnl_pct:.2%})")
            print(f"   Deontic Reasoning: {reasoning.split('.')[0]}")
            
            self.trade_history.append({
                'timestamp': 'current',
                'action': 'SELL',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'reasoning': reasoning,
                'pnl_pct': pnl_pct
            })
            return True
        return False
    
    def _get_last_entry_price(self, symbol: str) -> Optional[float]:
        """Get last buy price for symbol"""
        buy_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['action'] == 'BUY']
        return buy_trades[-1]['price'] if buy_trades else None
    
    def _get_average_cost(self, symbol: str) -> Optional[float]:
        """Calculate average cost basis"""
        buy_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['action'] == 'BUY']
        if not buy_trades:
            return None
        total_cost = sum(t['shares'] * t['price'] for t in buy_trades)
        total_shares = sum(t['shares'] for t in buy_trades)
        return total_cost / total_shares if total_shares > 0 else None
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        value = self.balance
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                value += shares * current_prices[symbol]
        return value
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        current_value = self.balance + sum(self.positions.values()) * 100  # Rough estimate
        peak_value = max(self.initial_balance, current_value)
        return (peak_value - current_value) / peak_value if peak_value > 0 else 0

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators with previous values for crossover detection"""
    data = data.copy()
    
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    
    # Previous values for crossover detection
    data['SMA_5_prev'] = data['SMA_5'].shift(1)
    data['SMA_20_prev'] = data['SMA_20'].shift(1)
    
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
    print("🧠 Dual Scale Detachment Trading Agent")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    prices = [100]
    for _ in range(len(dates)-1):
        change = np.random.normal(0.001, 0.025)  # More volatile
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Create agents with different personalities
    conservative_agent = DualScaleTradingAgent(10000, TraderPersonality.CONSERVATIVE)
    aggressive_agent = DualScaleTradingAgent(10000, TraderPersonality.AGGRESSIVE)
    
    # Run simulation for last 20 days
    for date in data.index[-20:]:
        market_data = data.loc[date]
        
        print(f"\n📅 {date.strftime('%Y-%m-%d')} - Price: ${market_data['Close']:.2f}")
        print("-" * 50)
        
        # Conservative agent
        print("🛡️ CONSERVATIVE AGENT (Dual Scale Detachment):")
        action, reasoning = conservative_agent.determine_best_action(market_data, 'SAMPLE')
        conservative_agent.execute_decision(action, market_data, 'SAMPLE', reasoning)
        
        print("\n🚀 AGGRESSIVE AGENT (Dual Scale Detachment):")
        action, reasoning = aggressive_agent.determine_best_action(market_data, 'SAMPLE')  
        aggressive_agent.execute_decision(action, market_data, 'SAMPLE', reasoning)
    
    # Show final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"Conservative Agent: ${conservative_agent.balance:.2f}")
    print(f"Aggressive Agent: ${aggressive_agent.balance:.2f}")
    print(f"Conservative Trades: {len(conservative_agent.trade_history)}")
    print(f"Aggressive Trades: {len(aggressive_agent.trade_history)}")