from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import json

from typing import Optional, Union, Dict

class AlpacaDataFetcher:
    def __init__(self, api_key: str, secret_key: str):
        self.hist_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.crypto_hist_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        self.url = "https://data.alpaca.markets/v2/stocks/bars?"

        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        self.payload = {

        }
        self.timeframe_map = [
            "1Min",
            "5Min",
            "15Min", 
            "30Min",
            "1H",
            "1D",
            "1W",
            "1month"]

       
    def _format_crypto_symbol(self, symbol: str) -> str:
        """Convert crypto symbol to correct format"""
        if symbol.endswith('USDT'):
            return f"{symbol[:-4]}/USD"
        elif symbol.endswith('USD'):
            return f"{symbol[:-3]}/USD"
        return symbol
    
    def _process_raw_bars_to_df(self, bars_data):
        df = pd.DataFrame(bars_data)
        df['t'] = pd.to_datetime(df['t'])
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high', 
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'n': 'trades',
            'vw': 'vwap'
        })
        return df

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
        if end_date in [None, '']:
            end_date = datetime.now(timezone(timedelta(hours=-5)))
        
        # Set default start date if not provided
        if start_date in [None, '']: 
            start_date = datetime.now(timezone(timedelta(hours=-5))) - timedelta(days=30)

        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=datetime.timezone.utc)
            
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=datetime.timezone.utc)

        # Determine if it's a crypto symbol and format accordingly
        is_crypto = False
        if is_crypto:
            symbol = self._format_crypto_symbol(symbol)

        if is_crypto:
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=limit
            )
            bars = self.crypto_hist_client.get_crypto_bars(request_params)
        else:
            if timeframe in self.timeframe_map:
                # construct api payload request.
                self.payload = {
                    "symbols": ','.join(symbol),
                    "timeframe": timeframe,
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                    "limit":limit,
                    "feed": 'iex'
                    }
            # format the bars alpaca api response into a dataframe
            response = requests.get(self.url, headers=self.headers, params=self.payload)
            data = response.json()
            dfs = []
            for symbol, bars in data['bars'].items():
                symbol_df = self._process_raw_bars_to_df(bars)
                symbol_df['symbol'] = symbol  # Add symbol here instead
                dfs.append(symbol_df)

            # Combine all symbols into one DataFrame
            final_df = pd.concat(dfs, axis=0)
            final_df = final_df.set_index(['timestamp', 'symbol'])

            return response, final_df


    def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get the latest quote for a symbol"""
        
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


    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            account = self.trading_client.get_account()
            return float(account.cash)
        except Exception as e:
            raise Exception(f"Error getting account balance: {str(e)}")