import pandas as pd
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class OHLCV:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class Tick:
    def __init__(self, timestamp, symbol, side, amount, price, exchange):
        self.timestamp = timestamp
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.price = price
        self.exchange = exchange

class DataSample:
    def __init__(self):
        pass

    def generate(self):
        #generates 1 data sample
        pass

    def build(self):
        #builds a dataframe of data samples
        pass

    def plot(self):
        #visualizes our dataset
        pass

class DataSampleOHLCV(DataSample):
    def __init__(self):
        self.ts_counter = 0
        self.last_price = random.randrange(19000, 21000)

    def generate(self):
        timestamp = datetime.now() + timedelta(minutes = self.ts_counter)
        self.ts_counter += 1
        open = random.randrange(self.last_price - 10, self.last_price + 10)
        high = random.randrange(self.last_price, self.last_price + 10)
        low = random.randrange(self.last_price - 10, self.last_price)
        close = random.randrange(self.last_price - 10, self.last_price + 11)
        self.last_price = close
        volume = random.randrange(10, 300)
        ohlcv = OHLCV(timestamp, open, high, low, close, volume)
        return ohlcv

    def build(self, n = 3):
        list_of_ohlcvs = []
        for i in range(0, n):
            ohlcv = self.generate()
            list_of_ohlcvs.append([ohlcv.timestamp, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume])
        df = pd.DataFrame(list_of_ohlcvs, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

    def plot(self, df):
        df.close.plot(figsize = (12, 8))
        plt.title('OHLCV: {} samples'.format(len(df)), fontsize = 15)
        plt.show()

class DataSampleTicks(DataSample):
    def __init__(self):
        self.last_price = random.randrange(19000, 21000)

    def generate(self):
        timestamp = time.time()
        symbol = 'BTC/USD'
        side = random.choice(['BUY', 'SELL'])
        amount = random.randrange(0, 10)
        price = random.randrange(self.last_price - 10, self.last_price + 11)
        self.last_price = price
        exchange = random.choice(['Binance', 'FTX', 'Bybit', 'dYdX'])
        tick = Tick(timestamp, symbol, side, amount, price, exchange)
        return tick

    def build(self, n = 3):
        list_of_ticks = []
        for i in range(0, n):
            tick = self.generate()
            list_of_ticks.append([tick.timestamp, tick.symbol, tick.side, tick.amount, tick.price, tick.exchange])
        df = pd.DataFrame(list_of_ticks, columns = ['timestamp', 'symbol', 'side', 'amount', 'price', 'exchange'])
        return df

    def plot(self, df):
        df.price.plot(figsize = (12, 8))
        plt.title('Ticks: {} samples'.format(len(df)), fontsize = 15)
        plt.show()

class DataGenerator:
    def __init__(self):
        self.ticks = DataSampleTicks()
        self.ohlcv = DataSampleOHLCV()

    def run_ticks(self):
        data = self.ticks.build(n = 10000)
        self.ticks.plot(data)
        return data

    def run_ohlcv(self):
        data = self.ohlcv.build(n = 1000)
        self.ohlcv.plot(data)
        return data

if __name__ == '__main__':
    app = DataGenerator()
    df1 = app.run_ticks()
    df1.to_csv('out_ticks.csv', index=False) # Uncomment to save the ticks data to a CSV file
    # print(df1)
    # print(df1.head())
    # print(df1.info())
    # print(df1.describe())
    df2 = app.run_ohlcv()
    df2.to_csv('out_ohlcv.csv', index=False) # Uncomment to save the OHLCV data to a CSV file
    # print(df2)
    # print(df2.head())
    # print(df2.info())
    # print(df2.describe())

