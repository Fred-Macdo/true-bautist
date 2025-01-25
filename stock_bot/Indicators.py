from typing import Dict, Any, Tuple, List, Set
import talib 
import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, df, params=None):
        """
        Initialize with dataframe and optional parameter dictionary
        
        Args:
            df: DataFrame with OHLCV data
            params: Dictionary of parameters for each indicator
        """
        self.df = df.copy()
        # Default parameters if none provided
        self.params = params or {
            'sma': {'period': 20},
            'ema': {'period': 5},
            'rsi': {'period': 14},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'keltner_channels': {'period': 20, 'atr_multiplier': 2},
            'adx': {'period': 14},
            'obv': {},  # No parameters needed
            'mfi': {'period': 14},
            'cci': {'period': 20}
        }
        

    # Example implementation of one method
    def calculate_sma(self, period: int=20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return talib.SMA(self.df.close, period)

    def calculate_ema(self, period: int=20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return talib.EMA(self.df.close, period)

    def calculate_rsi(self, period: int=14) -> pd.Series:
        """Calculate Relative Strength Index"""
        self.df[f'rsi'] = talib.RSI(self.df.close, period)

    def calculate_bollinger_bands(self, period: int=20, std_dev: float=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        upperband, middleband, lowerband = talib.BBANDS(self.df.close, period, std_dev)
        self.df[f'upperband'], self.df['middleband'], self.df['lowerband'] = upperband, middleband, lowerband

    def calculate_atr(self, period: int=14) -> pd.Series:
        """Calculate Average True Range"""
        self.df['atr'] = talib.ATR(self.df.high, self.df.low, self.df.close, period)


    def calculate_keltner_channels(self, period: int=20, atr_mult: float=2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        middle = typical_price.rolling(window=period).mean()
        atr = talib.ATR(self.df.high, self.df.low, self.df.close, period)
        
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        self.df['keltner_upper'], self.df['keltner_middle'], self.df['keltner_lower'] =  upper, middle, lower
        
    def calculate_adx(self, period: int=14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        self.df['adx'] = talib.ADX(self.df.high,
                                   self.df.low,
                                   self.df.close,
                                   period)

    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume"""
        self.df['obv'] = talib.OBV(self.df.close, 
                         self.df.volume)

    def calculate_mfi(self, period: int=14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        
        positive_flow = pd.Series(0, index=self.df.index)
        negative_flow = pd.Series(0, index=self.df.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(self.df)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            else:
                negative_flow[i] = money_flow[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        self.df['mfi'] = mfi

    def calculate_cci(self, period: int=20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        self.df['cci'] = talib.CCI(self.df.high, 
                         self.df.low, 
                         self.df.close, 
                         period)

    def calculate_indicators(self):
        """
        Calculate all technical indicators using parameters from self.params
        """
        # Dictionary mapping indicator names to their calculation methods
        indicator_methods = {
            'sma': self.calculate_sma,
            'ema': self.calculate_ema,
            'rsi': self.calculate_rsi,
            'bbands': self.calculate_bollinger_bands,
            'atr': self.calculate_atr,
            'keltner_channels': self.calculate_keltner_channels,
            'adx': self.calculate_adx,
            'obv': self.calculate_obv,
            'mfi': self.calculate_mfi,
            'cci': self.calculate_cci
        }
        
        for key, value in self.params.items():
            if key[0:3] in ['ema', 'sma']:
                ind = key[0:3]
                period = value.get('period')
                self.df[key] = indicator_methods[ind](period)
            else:
                indicator_methods[key](period)

        # Get all the previous day values in 
        # self.df for each indicator selected
        self.df = self._calculate_previous_values()

        return self.df

    def _calculate_previous_values(self) -> pd.DataFrame:
        '''
        Get all the previous values for close + indicator columns
        '''
        result_df = self.df.copy()
        exclude_cols = ['open', 'high', 'low', 'volume', 'trade_count', 'vwap'] # don't get prev values
        prev_col_list = list(filter(lambda x: x not in exclude_cols, result_df.columns))
        
        # shift the columns each by one to get previous days value
        for col in prev_col_list:
            if col in result_df.columns:
                result_df[f'{col}_prev'] = result_df[col].shift(1)

        return result_df

    def get_df(self):
        return self.df