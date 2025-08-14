import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class BasicTradingAgent:
    def __init__(self, initial_balance=10000, commission=0.001):
        """
        Initialize the trading agent
        
        Args:
            initial_balance: Starting cash amount
            commission: Commission rate (0.001 = 0.1%)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.commission = commission
        self.trade_history = []
        self.portfolio_value_history = []
        
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
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on simple moving average crossover
        
        Strategy: Buy when short MA crosses above long MA, sell when it crosses below
        """
        # Calculate moving averages
        data['SMA_5'] = self.calculate_sma(data['Close'], 5)
        data['SMA_20'] = self.calculate_sma(data['Close'], 20)
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Generate signals
        data['Signal'] = 0
        data['Position'] = 0
        
        # Buy signal: SMA_5 crosses above SMA_20 and RSI < 70 (not overbought)
        buy_condition = (
            (data['SMA_5'] > data['SMA_20']) & 
            (data['SMA_5'].shift(1) <= data['SMA_20'].shift(1)) &
            (data['RSI'] < 70)
        )
        
        # Sell signal: SMA_5 crosses below SMA_20 or RSI > 80 (overbought)
        sell_condition = (
            (data['SMA_5'] < data['SMA_20']) & 
            (data['SMA_5'].shift(1) >= data['SMA_20'].shift(1))
        ) | (data['RSI'] > 80)
        
        data.loc[buy_condition, 'Signal'] = 1
        data.loc[sell_condition, 'Signal'] = -1
        
        return data
    
    def execute_trade(self, symbol, signal, price, timestamp):
        """Execute buy or sell order"""
        if signal == 1:  # Buy signal
            if self.balance > 100:  # Minimum trade amount
                # Calculate position size (use 10% of available balance)
                position_size = (self.balance * 0.1) / price
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
                        'commission': commission_cost
                    }
                    self.trade_history.append(trade)
                    print(f"BUY: {position_size:.4f} shares of {symbol} at ${price:.2f}")
        
        elif signal == -1:  # Sell signal
            if symbol in self.positions and self.positions[symbol] > 0:
                position_size = self.positions[symbol]
                revenue = position_size * price
                commission_cost = revenue * self.commission
                
                self.balance += (revenue - commission_cost)
                self.positions[symbol] = 0
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position_size,
                    'price': price,
                    'value': revenue,
                    'commission': commission_cost
                }
                self.trade_history.append(trade)
                print(f"SELL: {position_size:.4f} shares of {symbol} at ${price:.2f}")
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.balance
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity > 0:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
    
    def run_backtest(self, data, symbol):
        """Run backtest on historical data"""
        print(f"Starting backtest for {symbol}")
        print(f"Initial balance: ${self.initial_balance:.2f}")
        print("-" * 50)
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Execute trades based on signals
        for i, row in data.iterrows():
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
        
        print("-" * 50)
        print(f"Final portfolio value: ${final_value:.2f}")
        print(f"Total return: {total_return:.2f}%")
        print(f"Number of trades: {len(self.trade_history)}")
        
        return data
    
    def get_performance_stats(self):
        """Calculate basic performance statistics"""
        if not self.portfolio_value_history:
            return {}
        
        values = [entry['portfolio_value'] for entry in self.portfolio_value_history]
        returns = np.diff(values) / values[:-1]
        
        stats = {
            'Total Return': ((values[-1] - self.initial_balance) / self.initial_balance) * 100,
            'Max Portfolio Value': max(values),
            'Min Portfolio Value': min(values),
            'Volatility': np.std(returns) * np.sqrt(252) * 100,  # Annualized
            'Sharpe Ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'Number of Trades': len(self.trade_history)
        }
        
        return stats

# Example usage with sample data
def generate_sample_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate realistic price movement
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Small daily returns with volatility
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    data.set_index('Date', inplace=True)
    return data

if __name__ == "__main__":
    # Create sample data
    sample_data = generate_sample_data()
    
    # Initialize and run the trading agent
    agent = BasicTradingAgent(initial_balance=10000, commission=0.001)
    
    # Run backtest
    results = agent.run_backtest(sample_data, 'SAMPLE')
    
    # Print performance statistics
    stats = agent.get_performance_stats()
    print("\nPerformance Statistics:")
    print("-" * 30)
    for key, value in stats.items():
        if 'Return' in key or 'Volatility' in key:
            print(f"{key}: {value:.2f}%")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")