import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

class MarketData:
    """
    Handles market data fetching and preprocessing for both live and historical data.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_cache = {}
        
    def get_historical_data(self, 
                          symbols: Union[str, List[str]], 
                          start_date: str,
                          end_date: str,
                          interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for one or multiple symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        data_dict = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                df = df.rename(columns={'stock splits': 'splits'})
                
                # Add technical indicators if configured
                if self.config.get('add_indicators', False):
                    df = self._add_technical_indicators(df)
                    
                data_dict[symbol] = df
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return data_dict
    
    def get_live_data(self, 
                     symbols: Union[str, List[str]], 
                     lookback_days: int = 100,
                     interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch recent data for live trading, including enough history for indicators.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.get_historical_data(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval=interval
        )
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the dataset."""
        # Calculate basic indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # Add volatility indicators
        df['atr'] = self._calculate_atr(df, period=14)
        df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, 
                                 period: int = 20, 
                                 std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
