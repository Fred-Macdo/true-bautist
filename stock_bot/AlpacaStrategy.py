from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import time
from datetime import datetime, timedelta
from .Strategy import Strategy
from typing import Dict, Any
import pandas as pd


class AlpacaPaperStrategy(Strategy):
    """
    Extension of Strategy class for paper trading with Alpaca
    """
    def __init__(self, config: Dict[str, Any], keys: Dict[str, str]):
        super().__init__(config, keys)
        self.trading_client = TradingClient(
            api_key=keys['api_key'],
            secret_key=keys['api_secret'],
            paper=True
        )
        self.current_positions = {}
        self.account_balance = float(self.trading_client.get_account().cash)

    def get_current_market_data(self) -> pd.DataFrame:
        """
        Get the most recent market data for strategy calculations
        Including enough historical data to calculate indicators accurately
        """
        # Temporarily modify config dates for data fetch
        original_start_date = self.config['start_date']
        original_end_date = self.config['end_date']
        
        try:
            # Set dates for fetching recent data
            end_date = datetime.now()
            # Calculate start date based on longest lookback period needed
            max_lookback = self._get_max_lookback_period()
            start_date = end_date - timedelta(days=max_lookback + 5)  # Add buffer days
            
            # Update config temporarily
            self.config['start_date'] = start_date.strftime("%Y-%m-%d")
            self.config['end_date'] = end_date.strftime("%Y-%m-%d")
            
            # Fetch and process data
            df = self.get_data()
            
            # Return only the most recent data point for each symbol
            latest_data = df.groupby('symbol').last().reset_index()
            
            return latest_data
            
        finally:
            # Restore original config dates
            self.config['start_date'] = original_start_date
            self.config['end_date'] = original_end_date
    
    def _get_max_lookback_period(self) -> int:
        """
        Calculate the maximum lookback period needed based on indicators
        """
        max_period = 0
        
        # Check indicator periods
        for indicator, params in self.indicators.items():
            if 'period' in params:
                max_period = max(max_period, params['period'])
            elif indicator == 'macd':
                # MACD typically needs 26 + 9 periods
                max_period = max(max_period, 35)
            elif indicator == 'bbands':
                max_period = max(max_period, params.get('timeperiod', 20))
                
        # Convert to days based on timeframe
        timeframe = self.config['timeframe']
        if timeframe == '1D':
            return max_period
        elif timeframe == '1H':
            return max_period // 24 + 1
        elif timeframe == '15Min':
            return max_period // 96 + 1
        elif timeframe == '5Min':
            return max_period // 288 + 1
        elif timeframe == '1Min':
            return max_period // 1440 + 1
        
        # Default to 30 days if timeframe is not recognized
        return 30

    def submit_order(self, symbol: str, quantity: float, side: str) -> None:
        """
        Submit a market order to Alpaca
        """
        try:
            order_details = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_details)
            print(f"Order submitted - {side.upper()}: {quantity} {symbol}")
            return order
        except Exception as e:
            print(f"Error submitting order: {str(e)}")
            return None

    def check_and_update_positions(self):
        """
        Update current positions from Alpaca
        """
        try:
            positions = self.trading_client.get_all_positions()
            self.current_positions = {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            print(f"Error updating positions: {str(e)}")

    def run_paper_trading(self, interval_seconds: int = 60):
        """
        Run the paper trading strategy
        
        Args:
            interval_seconds: How often to check for new signals (in seconds)
        """
        print("Starting paper trading strategy...")
        
        while True:
            try:
                # Get current market data and calculate indicators
                current_data = self.get_current_market_data()
                if current_data.empty:
                    print("No market data available")
                    time.sleep(interval_seconds)
                    continue

                # Update account information
                self.account_balance = float(self.trading_client.get_account().cash)
                self.check_and_update_positions()

                # Check for signals on each symbol
                for symbol in self.config['symbols']:
                    symbol_data = current_data[current_data['symbol'] == symbol].iloc[-1]

                    # Check entry conditions if we don't have a position
                    if symbol not in self.current_positions and self._check_entry_conditions(symbol_data):
                        position_size = self._calculate_position_size(symbol_data, self.account_balance)
                        if position_size > 0:
                            self.submit_order(symbol, position_size, "buy")

                    # Check exit conditions if we have a position
                    elif symbol in self.current_positions and self._check_exit_conditions(symbol_data):
                        position_size = self.current_positions[symbol]
                        self.submit_order(symbol, position_size, "sell")

                print(f"Strategy check completed at {datetime.now()}")
                print(f"Current positions: {self.current_positions}")
                print(f"Account balance: ${self.account_balance:.2f}")
                
                time.sleep(interval_seconds)

            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
                time.sleep(interval_seconds)

    def check_market_hours(self) -> bool:
        """
        Check if the market is currently open
        """
        clock = self.trading_client.get_clock()
        return clock.is_open

    def wait_for_market_open(self):
        """
        Wait for market to open if it's currently closed
        """
        clock = self.trading_client.get_clock()
        if not clock.is_open:
            time_to_open = clock.next_open - clock.timestamp
            sleep_duration = time_to_open.total_seconds()
            print(f"Market is closed. Waiting for market to open in {sleep_duration/3600:.2f} hours")
            time.sleep(sleep_duration)
