import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

def create_ohlcv_dataframe(
    symbols: List[str],
    start_date: datetime,
    periods: int = 30,
    freq: str = 'D',
    with_indicators: bool = False
) -> pd.DataFrame:
    """
    Create a sample OHLCV DataFrame for testing
    
    Args:
        symbols: List of symbols to include
        start_date: Starting date for the data
        periods: Number of periods (days, hours, etc.)
        freq: Frequency of data ('D' for daily, 'H' for hourly, etc.)
        with_indicators: Whether to include basic technical indicators
        
    Returns:
        DataFrame with OHLCV data
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    data = []
    for symbol in symbols:
        # Set base price range based on symbol
        if symbol == 'AAPL':
            base_price = 150
            price_range = 10
            volume_base = 5000
        elif symbol == 'MSFT':
            base_price = 300
            price_range = 15
            volume_base = 3000
        elif symbol == 'SPY':
            base_price = 420
            price_range = 5
            volume_base = 10000
        else:
            base_price = 100
            price_range = 8
            volume_base = 2000
        
        # Create random price data with a trend
        trend = np.linspace(-price_range/2, price_range/2, periods)
        
        for i, date in enumerate(dates):
            # Generate OHLCV data
            close = base_price + trend[i] + np.random.uniform(-price_range/4, price_range/4)
            open_price = close + np.random.uniform(-price_range/10, price_range/10)
            high = max(open_price, close) + np.random.uniform(0, price_range/5)
            low = min(open_price, close) - np.random.uniform(0, price_range/5)
            volume = int(volume_base + np.random.uniform(-volume_base/4, volume_base/4))
            
            # Add row to data
            row_data = {
                'timestamp': date,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'trade_count': int(volume / 100),
                'vwap': (high + low + close) / 3
            }
            
            # Add indicators if requested
            if with_indicators:
                if i >= 14:  # Ensure enough data for indicators
                    # Simple indicators for testing
                    row_data['sma_20'] = np.nan if i < 20 else np.mean([d['close'] for d in data[-20:] if d['symbol'] == symbol])
                    row_data['rsi'] = 50 + np.random.uniform(-20, 20)  # Simplified RSI
                else:
                    row_data['sma_20'] = np.nan
                    row_data['rsi'] = np.nan
            
            data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def create_mock_alpaca_response(symbols: List[str], 
                               start_date: datetime,
                               periods: int = 10,
                               freq: str = 'D') -> Dict:
    """
    Create a mock Alpaca API response with bars data
    
    Args:
        symbols: List of symbols to include
        start_date: Starting date for the data
        periods: Number of periods
        freq: Frequency of data
        
    Returns:
        Dictionary formatted like an Alpaca API response
    """
    bars = {}
    
    for symbol in symbols:
        symbol_bars = []
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Set base price range based on symbol
        if symbol == 'AAPL':
            base_price = 150
            price_range = 10
            volume_base = 5000
        elif symbol == 'MSFT':
            base_price = 300
            price_range = 15
            volume_base = 3000
        else:
            base_price = 100
            price_range = 8
            volume_base = 2000
        
        # Create random price data with a trend
        trend = np.linspace(-price_range/2, price_range/2, periods)
        
        for i, date in enumerate(dates):
            # Generate OHLCV data
            close = base_price + trend[i] + np.random.uniform(-price_range/4, price_range/4)
            open_price = close + np.random.uniform(-price_range/10, price_range/10)
            high = max(open_price, close) + np.random.uniform(0, price_range/5)
            low = min(open_price, close) - np.random.uniform(0, price_range/5)
            volume = int(volume_base + np.random.uniform(-volume_base/4, volume_base/4))
            
            # Format as Alpaca bar
            bar = {
                't': date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'o': open_price,
                'h': high,
                'l': low,
                'c': close,
                'v': volume,
                'n': int(volume / 100),
                'vw': (high + low + close) / 3
            }
            
            symbol_bars.append(bar)
        
        bars[symbol] = symbol_bars
    
    return {'bars': bars}

def create_mock_timeframe():
    """
    Create a mock TimeFrame object for testing
    
    Returns:
        Mock TimeFrame object
    """
    class MockTimeFrame:
        Day = 'day'
        Hour = 'hour'
        Minute = 'minute'
        
        def __init__(self, value, unit):
            self.value = value
            self.unit = unit
    
    return MockTimeFrame

def create_sample_strategy_config(symbols: List[str] = None) -> Dict:
    """
    Create a sample strategy configuration for testing
    
    Args:
        symbols: List of symbols to include
        
    Returns:
        Strategy configuration dictionary
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT']
    
    return {
        'symbols': symbols,
        'timeframe': '1D',
        'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00'),
        'end_date': datetime.now().strftime('%Y-%m-%dT00:00:00'),
        'indicators': [
            {'name': 'EMA', 'params': {'period': 20}},
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'BBANDS', 'params': {'period': 20, 'std_dev': 2.0}}
        ],
        'entry_conditions': [
            {'indicator': 'rsi', 'condition': '<', 'value': 30},
            {'indicator': 'close', 'condition': '<', 'value': 'lowerband'}
        ],
        'exit_conditions': [
            {'indicator': 'rsi', 'condition': '>', 'value': 70},
            {'indicator': 'close', 'condition': '>', 'value': 'upperband'}
        ],
        'risk_management': {
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'max_open_positions': 5
        }
    }

def create_sample_market_condition() -> Dict:
    """
    Create a sample market condition result for testing
    
    Returns:
        Market condition dictionary
    """
    return {
        'trend': 'bullish',
        'volatility': 'normal',
        'breadth': 'expanding',
        'sentiment': 'positive',
        'risk_level': 'low',
        'details': {
            'trend_metrics': {
                'price_vs_sma50': 2.5,
                'price_vs_sma200': 5.7,
                'momentum': 3.2,
                'trend_strength': 3.2
            },
            'volatility_metrics': {
                'current_vix': 18.5,
                'vix_20d_avg': 20.2,
                'vix_percentile': 35.0,
                'vix_trend': -8.42
            },
            'breadth_metrics': {
                'advancing_sectors': 6,
                'total_sectors': 7,
                'breadth_ratio': 85.71,
                'sector_returns': {
                    'XLK': 4.5,
                    'XLF': 2.3,
                    'XLE': 1.8,
                    'XLV': 0.7,
                    'XLI': 3.2,
                    'XLP': -1.5,
                    'XLY': 2.8
                }
            },
            'sentiment_metrics': {
                'sentiment_score': 3.5,
                'vix_trend': -5.2,
                'gold_trend': -1.3,
                'treasury_trend': -0.5
            }
        }
    }

def create_sample_trades() -> pd.DataFrame:
    """
    Create a sample trades DataFrame for testing
    
    Returns:
        DataFrame with trade data
    """
    # Create trade data
    trades_data = {
        'entry_time': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'exit_time': pd.date_range(start='2023-01-02', periods=10, freq='D'),
        'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'entry_price': [150, 300, 152, 305, 148, 295, 155, 310, 160, 315],
        'exit_price': [155, 310, 148, 295, 155, 310, 160, 315, 155, 310],
        'quantity': [10, 5, 10, 5, 10, 5, 10, 5, 10, 5],
        'direction': ['long', 'long', 'long', 'long', 'long', 'long', 'long', 'long', 'short', 'short'],
        'pnl': [50, 50, -40, -50, 70, 75, 50, 25, 50, 25]
    }
    
    # Create DataFrame
    trades_df = pd.DataFrame(trades_data)
    
    return trades_df
