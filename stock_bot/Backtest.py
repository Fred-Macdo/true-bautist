import pandas as pd
import numpy as np
from typing import Dict, Any
import yaml
from .Strategy import Strategy
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
import numpy as np

class Backtest:
    """
    Backtest class to evaluate a trading strategy using the Alpaca API for both historical data and paper trading.
    """
    def __init__(self, strategy: 'Strategy'):
        self.strategy = strategy
        self.api = TradingClient(strategy.get_keys()['api_key'], strategy.get_keys()['api_secret'], paper=True)
        self.positions = []
        self.account_balance = 100000  # Default starting balance for paper trading
        self.trades = pd.DataFrame(columns=['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'position_size', 'pnl'])
        self.equity = pd.Series(dtype=float)

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data using the Alpaca API and apply indicators."""
        config = self.strategy.get_config()
        symbol_data = []
        for symbol in config['symbols']:
            bars = self.api.get_bars(
                symbol,
                config['timeframe'],
                start=config['start_date'],
                end=config['end_date']
            ).df
            bars['symbol'] = symbol
            symbol_data.append(bars)

        historical_data = pd.concat(symbol_data)
        historical_data.sort_index(inplace=True)
        historical_data = self.strategy.calculate_indicators(historical_data)
        return historical_data

    def get_open_positions(self):
        """Retrieve all open positions using the Alpaca API."""
        positions = self.api.list_positions()
        return pd.DataFrame([{**position._raw} for position in positions])

    def execute_trade(self, symbol: str, qty: float, side: str):
        """Execute a trade (buy/sell) on the Alpaca paper trading account."""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            print(f"{side.capitalize()} order placed: {order}")
        except Exception as e:
            print(f"Error placing {side} order for {symbol}: {e}")

    def run_backtest(self):
        """Run the backtest using historical data and paper trading on Alpaca."""
        historical_data = self.fetch_historical_data()
        for index, row in historical_data.iterrows():
            symbol = row['symbol']
            if self.strategy._check_entry_conditions(row):
                # Calculate position size and execute buy order
                position_size = self.strategy._calculate_position_size(row, self.account_balance)
                if position_size > 0:
                    self.execute_trade(symbol, position_size, side='buy')

                    self.trades = pd.concat([
                        self.trades,
                        pd.DataFrame.from_records([{
                            'symbol': symbol,
                            'entry_date': index,
                            'entry_price': row['close'],
                            'position_size': position_size,
                            'exit_date': None,
                            'exit_price': None,
                            'pnl': None
                        }])
                    ])

            open_positions = self.get_open_positions()
            for _, position in open_positions.iterrows():
                if self.strategy._check_exit_conditions(row):
                    # Close position if exit conditions are met
                    self.execute_trade(position['symbol'], position['qty'], side='sell')
                    trade_index = self.trades[(self.trades['symbol'] == position['symbol']) & (self.trades['exit_date'].isna())].index[0]
                    self.trades.loc[trade_index, 'exit_date'] = index
                    self.trades.loc[trade_index, 'exit_price'] = row['close']

                    entry_price = self.trades.loc[trade_index, 'entry_price']
                    position_size = self.trades.loc[trade_index, 'position_size']
                    pnl = self.strategy._calculate_pnl(entry_price, row['close'], position_size)
                    self.trades.loc[trade_index, 'pnl'] = pnl

        self.calculate_equity()
        self.calculate_metrics()

    def calculate_equity(self):
        """Calculate the equity curve based on trades."""
        self.equity = pd.Series(dtype=float)
        self.equity.loc[0] = self.account_balance

        for i, trade in self.trades.iterrows():
            pnl = trade['pnl'] if pd.notna(trade['pnl']) else 0
            self.equity.loc[i + 1] = self.equity.iloc[-1] + pnl

    def calculate_metrics(self):
        """Calculate performance metrics for the backtest."""
        initial_balance = self.account_balance
        self.strategy.metrics = self.strategy._calculate_metrics(self.trades, self.equity, initial_balance)
        print("Backtest Metrics:", self.strategy.metrics)

# Usage example
# strategy = Strategy()
# backtest = Backtest(strategy)
# backtest.run_backtest()

# Example usage:
# strategy = Strategy(config={}, keys={"api_key": "<YOUR_API_KEY>", "api_secret": "<YOUR_API_SECRET>"})
# backtest = Backtest(strategy, "config.yaml")
# backtest.run()
# backtest.summary()
