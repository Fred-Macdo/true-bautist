from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Optional, Union, Dict

class AlpacaDataFetcher:
    def __init__(self, api_key: str, secret_key: str):
        self.hist_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.crypto_hist_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        
        self.timeframe_map = {
            '1m': TimeFrame(1, TimeFrame.Minute),
            '5m': TimeFrame(5, TimeFrame.Minute),
            '15m': TimeFrame(15, TimeFrame.Minute),
            '1h': TimeFrame(1, TimeFrame.Hour),
            '1d': TimeFrame(1, TimeFrame.Day)
        }

    def _format_crypto_symbol(self, symbol: str) -> str:
        """Convert crypto symbol to correct format"""
        if symbol.endswith('USDT'):
            return f"{symbol[:-4]}/USD"
        elif symbol.endswith('USD'):
            return f"{symbol[:-3]}/USD"
        return symbol

    def _convert_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Convert string timeframe to Alpaca TimeFrame object"""
        if timeframe_str not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}. "
                           f"Supported timeframes: {list(self.timeframe_map.keys())}")
        return self.timeframe_map[timeframe_str]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Alpaca
        
        Parameters:
        - symbol: Trading symbol (e.g., 'BTC/USD' for crypto, 'AAPL' for stocks)
        - timeframe: Time interval ('1m', '5m', '15m', '1h', '1d')
        - start_date: Start date for historical data (defaults to 100 bars before end_date)
        - end_date: End date for historical data (defaults to now)
        - limit: Maximum number of bars to fetch
        
        Returns:
        - pandas DataFrame with OHLCV data
        """
        # Set default end date to now if not provided
        if end_date is None:
            end_date = datetime.now(pytz.UTC)
        
        # Set default start date if not provided
        if start_date is None:
            timeframe_mins = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '1d': 1440
            }
            minutes = timeframe_mins[timeframe] * limit
            start_date = end_date - timedelta(minutes=minutes)

        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)

        # Determine if it's a crypto symbol and format accordingly
        is_crypto = symbol.endswith('USD') or symbol.endswith('USDT') or '/' in symbol
        if is_crypto:
            symbol = self._format_crypto_symbol(symbol)
        
        try:
            # Create the appropriate request
            if is_crypto:
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=self._convert_timeframe(timeframe),
                    start=start_date,
                    end=end_date,
                    limit=limit
                )
                bars = self.crypto_hist_client.get_crypto_bars(request_params)
            else:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=self._convert_timeframe(timeframe),
                    start=start_date,
                    end=end_date,
                    limit=limit
                )
                bars = self.hist_client.get_stock_bars(request_params)
            
            '''print(bars.data)
            # Convert to DataFrame
            # The bars object is now a dictionary with symbol as key
            df = pd.DataFrame(bars.data[symbol], index=bars.data[symbol]['timestamp'])
            
            
            if len(df) == 0:
                raise ValueError(f"No data received for {symbol}")

            # Set timestamp as index
            #df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)'''

            return bars.df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get the latest quote for a symbol"""
        try:
            is_crypto = symbol.endswith('USD') or symbol.endswith('USDT') or '/' in symbol
            if is_crypto:
                symbol = self._format_crypto_symbol(symbol)
                quote = self.trading_client.get_crypto_quote(symbol)
            else:
                quote = self.trading_client.get_latest_quote(symbol)
                
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'timestamp': quote.timestamp
            }
        except Exception as e:
            raise Exception(f"Error getting quote for {symbol}: {str(e)}")

    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            account = self.trading_client.get_account()
            return float(account.cash)
        except Exception as e:
            raise Exception(f"Error getting account balance: {str(e)}")