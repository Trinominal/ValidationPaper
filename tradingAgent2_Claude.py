import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import time

class TraderType(Enum):
    RISK_AVERSE = "risk_averse"
    BALANCED = "balanced"
    RISK_SEEKING = "risk_seeking"
    AGGRESSIVE = "aggressive"

class TradingAgent:
    def __init__(self, initial_balance=10000, commission=0.001, trader_type=TraderType.BALANCED):
        """
        Initialize the trading agent with different personality types
        
        Args:
            initial_balance: Starting cash amount
            commission: Commission rate (0.001 = 0.1%)
            trader_type: Risk profile of the trader
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.commission = commission
        self.trader_type = trader_type
        self.trade_history = []
        self.portfolio_value_history = []
        
        # Set trader-specific parameters
        self._set_trader_parameters()
        
    def _set_trader_parameters(self):
        """Set parameters based on trader type"""
        trader_configs = {
            TraderType.RISK_AVERSE: {
                'position_size_pct': 0.05,  # 5% of balance per trade
                'max_positions': 3,         # Max 3 positions at once
                'stop_loss_pct': 0.02,      # 2% stop loss
                'take_profit_pct': 0.04,    # 4% take profit
                'rsi_oversold': 25,         # Very oversold before buying
                'rsi_overbought': 75,       # Sell earlier when overbought
                'volatility_threshold': 0.15, # Avoid high volatility stocks
                'min_sma_separation': 0.02,  # Need clear trend signal
                'drawdown_limit': 0.10,     # Stop trading if 10% drawdown
            },
            TraderType.BALANCED: {
                'position_size_pct': 0.10,  # 10% of balance per trade
                'max_positions': 5,
                'stop_loss_pct': 0.05,      # 5% stop loss
                'take_profit_pct': 0.10,    # 10% take profit
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volatility_threshold': 0.25,
                'min_sma_separation': 0.01,
                'drawdown_limit': 0.20,
            },
            TraderType.RISK_SEEKING: {
                'position_size_pct': 0.20,  # 20% of balance per trade
                'max_positions': 8,
                'stop_loss_pct': 0.08,      # 8% stop loss (wider)
                'take_profit_pct': 0.20,    # 20% take profit
                'rsi_oversold': 35,         # Less strict oversold
                'rsi_overbought': 65,       # Hold longer before selling
                'volatility_threshold': 0.40, # Accept higher volatility
                'min_sma_separation': 0.005, # React to smaller signals
                'drawdown_limit': 0.30,
            },
            TraderType.AGGRESSIVE: {
                'position_size_pct': 0.35,  # 35% of balance per trade
                'max_positions': 10,
                'stop_loss_pct': 0.12,      # 12% stop loss (very wide)
                'take_profit_pct': 0.30,    # 30% take profit
                'rsi_oversold': 40,         # Buy on any dip
                'rsi_overbought': 60,       # Hold through momentum
                'volatility_threshold': 0.60, # Love volatility
                'min_sma_separation': 0.001, # React to any signal
                'drawdown_limit': 0.50,     # High risk tolerance
            }
        }
        
        self.config = trader_configs[self.trader_type]
        print(f"Initialized {self.trader_type.value} trader with:")
        print(f"- Position size: {self.config['position_size_pct']*100:.1f}% per trade")
        print(f"- Max positions: {self.config['max_positions']}")
        print(f"- Stop loss: {self.config['stop_loss_pct']*100:.1f}%")
        print(f"- Take profit: {self.config['take_profit_pct']*100:.1f}%")
        
    def calculate_sma(self, prices, window):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volatility(self, prices, window=20):
        """Calculate rolling volatility"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std()
    
    def calculate_drawdown(self):
        """Calculate current drawdown from peak"""
        if not self.portfolio_value_history:
            return 0
        
        values = [entry['portfolio_value'] for entry in self.portfolio_value_history]
        peak = max(values)
        current = values[-1]
        return (peak - current) / peak
    
    def check_risk_limits(self):
        """Check if current drawdown exceeds trader's limit"""
        current_drawdown = self.calculate_drawdown()
        if current_drawdown > self.config['drawdown_limit']:
            print(f"⚠️ Drawdown limit exceeded: {current_drawdown:.2%} > {self.config['drawdown_limit']:.2%}")
            return False
        return True
    
    def get_active_positions_count(self):
        """Count active positions"""
        return sum(1 for qty in self.positions.values() if qty > 0)
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on trader type"""
        # Calculate technical indicators
        data['SMA_5'] = self.calculate_sma(data['Close'], 5)
        data['SMA_20'] = self.calculate_sma(data['Close'], 20)
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Volatility'] = self.calculate_volatility(data['Close'])
        
        # Calculate SMA separation (trend strength)
        data['SMA_Separation'] = abs(data['SMA_5'] - data['SMA_20']) / data['SMA_20']
        
        data['Signal'] = 0
        
        # Risk-adjusted buy conditions
        base_buy_condition = (
            (data['SMA_5'] > data['SMA_20']) & 
            (data['SMA_5'].shift(1) <= data['SMA_20'].shift(1))
        )
        
        # Apply trader-specific filters
        rsi_buy_condition = data['RSI'] < self.config['rsi_oversold']
        volatility_condition = data['Volatility'] < self.config['volatility_threshold']
        trend_strength_condition = data['SMA_Separation'] > self.config['min_sma_separation']
        
        # Risk averse traders need all conditions
        if self.trader_type == TraderType.RISK_AVERSE:
            buy_condition = (base_buy_condition & rsi_buy_condition & 
                           volatility_condition & trend_strength_condition)
        # Balanced traders need most conditions
        elif self.trader_type == TraderType.BALANCED:
            buy_condition = base_buy_condition & (rsi_buy_condition | trend_strength_condition)
        # Risk seeking traders are more flexible
        elif self.trader_type == TraderType.RISK_SEEKING:
            buy_condition = base_buy_condition & (data['RSI'] < self.config['rsi_overbought'])
        # Aggressive traders buy on any upward signal
        else:  # AGGRESSIVE
            buy_condition = (base_buy_condition | 
                           (data['RSI'] < self.config['rsi_oversold']))
        
        # Sell conditions (trader-specific)
        base_sell_condition = (
            (data['SMA_5'] < data['SMA_20']) & 
            (data['SMA_5'].shift(1) >= data['SMA_20'].shift(1))
        )
        
        rsi_sell_condition = data['RSI'] > self.config['rsi_overbought']
        sell_condition = base_sell_condition | rsi_sell_condition
        
        data.loc[buy_condition, 'Signal'] = 1
        data.loc[sell_condition, 'Signal'] = -1
        
        return data
    
    def calculate_position_size(self, price):
        """Calculate position size based on trader type and current balance"""
        base_size = (self.balance * self.config['position_size_pct']) / price
        
        # Adjust for current risk exposure
        current_positions = self.get_active_positions_count()
        if current_positions >= self.config['max_positions']:
            return 0
        
        # Risk averse traders reduce size as positions increase
        if self.trader_type == TraderType.RISK_AVERSE:
            size_multiplier = 1 - (current_positions / self.config['max_positions']) * 0.5
            base_size *= size_multiplier
        
        return base_size
    
    def check_stop_loss_take_profit(self, symbol, current_price, entry_price):
        """Check if stop loss or take profit should trigger"""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return 0
        
        price_change = (current_price - entry_price) / entry_price
        
        # Stop loss check
        if price_change <= -self.config['stop_loss_pct']:
            print(f"🛑 Stop loss triggered for {symbol}: {price_change:.2%}")
            return -1
        
        # Take profit check
        if price_change >= self.config['take_profit_pct']:
            print(f"🎯 Take profit triggered for {symbol}: {price_change:.2%}")
            return -1
        
        return 0
    
    def execute_trade(self, symbol, signal, price, timestamp):
        """Execute buy or sell order with trader-specific logic"""
        # Check risk limits before trading
        if signal == 1 and not self.check_risk_limits():
            print("❌ Trade blocked due to risk limits")
            return
        
        if signal == 1:  # Buy signal
            position_size = self.calculate_position_size(price)
            
            if position_size > 0 and self.balance > 100:
                cost = position_size * price
                commission_cost = cost * self.commission
                
                if cost + commission_cost <= self.balance:
                    self.balance -= (cost + commission_cost)
                    self.positions[symbol] = self.positions.get(symbol, 0) + position_size
                    
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_size,
                        'price': price,
                        'value': cost,
                        'commission': commission_cost,
                        'entry_price': price,
                        'trader_type': self.trader_type.value
                    }
                    self.trade_history.append(trade)
                    print(f"📈 BUY ({self.trader_type.value}): {position_size:.4f} shares of {symbol} at ${price:.2f}")
        
        elif signal == -1:  # Sell signal
            if symbol in self.positions and self.positions[symbol] > 0:
                position_size = self.positions[symbol]
                revenue = position_size * price
                commission_cost = revenue * self.commission
                
                self.balance += (revenue - commission_cost)
                
                # Calculate P&L for this trade
                entry_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['action'] == 'BUY']
                if entry_trades:
                    avg_entry_price = sum(t['price'] * t['quantity'] for t in entry_trades) / sum(t['quantity'] for t in entry_trades)
                    pnl_pct = (price - avg_entry_price) / avg_entry_price
                    pnl_symbol = "📈" if pnl_pct > 0 else "📉"
                    print(f"{pnl_symbol} SELL ({self.trader_type.value}): {position_size:.4f} shares of {symbol} at ${price:.2f} (P&L: {pnl_pct:.2%})")
                
                self.positions[symbol] = 0
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position_size,
                    'price': price,
                    'value': revenue,
                    'commission': commission_cost,
                    'trader_type': self.trader_type.value
                }
                self.trade_history.append(trade)
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.balance
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity > 0:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
    
    def run_backtest(self, data, symbol):
        """Run backtest on historical data"""
        print(f"\n🚀 Starting backtest for {symbol} with {self.trader_type.value} strategy")
        print(f"Initial balance: ${self.initial_balance:.2f}")
        print("-" * 60)
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Execute trades based on signals
        for i, row in data.iterrows():
            # Check for stop loss/take profit on existing positions
            if symbol in self.positions and self.positions[symbol] > 0:
                # Find last entry price
                buy_trades = [t for t in self.trade_history if t['symbol'] == symbol and t['action'] == 'BUY']
                if buy_trades:
                    last_entry = buy_trades[-1]['price']
                    sl_tp_signal = self.check_stop_loss_take_profit(symbol, row['Close'], last_entry)
                    if sl_tp_signal != 0:
                        self.execute_trade(symbol, sl_tp_signal, row['Close'], i)
            
            # Execute regular signals
            if not pd.isna(row['Signal']) and row['Signal'] != 0:
                self.execute_trade(symbol, row['Signal'], row['Close'], i)
            
            # Track portfolio value
            current_prices = {symbol: row['Close']}
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.portfolio_value_history.append({
                'timestamp': i,
                'portfolio_value': portfolio_value
            })
        
        # Final portfolio value
        final_value = self.calculate_portfolio_value({symbol: data['Close'].iloc[-1]})
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print("-" * 60)
        print(f"🏁 Final portfolio value: ${final_value:.2f}")
        print(f"📊 Total return: {total_return:.2f}%")
        print(f"📈 Number of trades: {len(self.trade_history)}")
        print(f"⚖️ Max drawdown: {max([0] + [self.calculate_drawdown()]):.2%}")
        
        return data
    
    def get_performance_stats(self):
        """Calculate trader-specific performance statistics"""
        if not self.portfolio_value_history:
            return {}
        
        values = [entry['portfolio_value'] for entry in self.portfolio_value_history]
        returns = np.diff(values) / values[:-1]
        
        # Calculate win rate
        profitable_trades = 0
        total_trades = 0
        for i in range(len(self.trade_history)):
            if self.trade_history[i]['action'] == 'SELL':
                # Find corresponding buy
                symbol = self.trade_history[i]['symbol']
                sell_price = self.trade_history[i]['price']
                
                # Find most recent buy for this symbol
                for j in range(i-1, -1, -1):
                    if (self.trade_history[j]['symbol'] == symbol and 
                        self.trade_history[j]['action'] == 'BUY'):
                        buy_price = self.trade_history[j]['price']
                        if sell_price > buy_price:
                            profitable_trades += 1
                        total_trades += 1
                        break
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate maximum drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        stats = {
            'Trader Type': self.trader_type.value.replace('_', ' ').title(),
            'Total Return': ((values[-1] - self.initial_balance) / self.initial_balance) * 100,
            'Max Portfolio Value': max(values),
            'Min Portfolio Value': min(values),
            'Max Drawdown': max_drawdown * 100,
            'Volatility': np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0,
            'Sharpe Ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'Win Rate': win_rate,
            'Number of Trades': len(self.trade_history),
            'Avg Position Size': self.config['position_size_pct'] * 100
        }
        
        return stats

# Function to compare different trader types
def compare_trader_types(data, symbol, initial_balance=10000):
    """Compare performance of different trader types"""
    results = {}
    
    for trader_type in TraderType:
        print(f"\n{'='*70}")
        agent = TradingAgent(initial_balance=initial_balance, trader_type=trader_type)
        agent.run_backtest(data.copy(), symbol)
        results[trader_type.value] = agent.get_performance_stats()
    
    # Print comparison
    print(f"\n{'='*70}")
    print("📊 TRADER TYPE COMPARISON")
    print('='*70)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(2))
    
    return results

# Example usage with sample data
def generate_sample_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate more volatile realistic price movement
    initial_price = 100
    trend = 0.0003  # Slight upward trend
    volatility = 0.025  # Daily volatility
    
    prices = [initial_price]
    for i in range(1, len(dates)):
        # Add some market cycles
        cycle_factor = np.sin(i * 2 * np.pi / 60) * 0.001  # 60-day cycle
        daily_return = trend + cycle_factor + np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + daily_return))
    
    # Add some realistic high/low spreads
    highs = [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices]
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    data.set_index('Date', inplace=True)
    return data

if __name__ == "__main__":
    # Create sample data
    sample_data = generate_sample_data()
    
    # Compare all trader types
    results = compare_trader_types(sample_data, 'SAMPLE', initial_balance=10000)