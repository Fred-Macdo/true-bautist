import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    """Trading signal details"""
    symbol: str
    timestamp: datetime
    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0 to 1
    trade_type: str  # 'entry', 'exit', 'adjustment'
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None

class Strategy(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.positions = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        pass

class MultiFactorStrategy(Strategy):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.indicators: Dict[str, Callable] = {}
        self.entry_conditions: List[Callable] = []
        self.exit_conditions: List[Callable] = []
        self.risk_filters: List[Callable] = []
        self.position_sizing: Optional[Callable] = None
        
    def add_indicator(self, name: str, calculation_func: Callable):
        """Add a technical indicator"""
        self.indicators[name] = calculation_func
        
    def add_entry_condition(self, condition_func: Callable):
        """Add entry condition"""
        self.entry_conditions.append(condition_func)
        
    def add_exit_condition(self, condition_func: Callable):
        """Add exit condition"""
        self.exit_conditions.append(condition_func)
        
    def add_risk_filter(self, filter_func: Callable):
        """Add risk management filter"""
        self.risk_filters.append(filter_func)
        
    def set_position_sizing(self, sizing_func: Callable):
        """Set position sizing function"""
        self.position_sizing = sizing_func

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using multi-factor analysis"""
        signals = pd.Series(0, index=data.index)
        
        try:
            # Convert single DataFrame to dict format expected by multi-factor analysis
            data_dict = {self.parameters.get('symbol', 'default'): data}
            market_condition = self.parameters.get('market_condition', {'risk_level': 'medium'})
            
            # Generate detailed signals
            detailed_signals = self._generate_detailed_signals(data_dict, market_condition)
            
            # Convert detailed signals to simple signal series (-1, 0, 1)
            for signal in detailed_signals:
                idx = signal.timestamp
                signals[idx] = 1 if signal.direction == 'buy' else -1 if signal.direction == 'sell' else 0
                
        except Exception as e:
            print(f"Error generating signals: {e}")
            
        return signals
    
    def _generate_detailed_signals(self, data: Dict[str, pd.DataFrame], market_condition: Dict) -> List[Signal]:
        """Generate detailed trading signals using multi-factor analysis"""
        signals = []
        
        for symbol, df in data.items():
            try:
                # Calculate indicators
                indicators = self._calculate_indicators(df)
                
                # Check risk filters
                if not self._pass_risk_filters(df, indicators, market_condition):
                    continue
                
                # Generate signal
                signal = self._evaluate_conditions(symbol, df, indicators, market_condition)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error generating signal for {symbol}: {e}")
                
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        return {
            name: func(df) for name, func in self.indicators.items()
        }
    
    def _pass_risk_filters(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        market_condition: Dict
    ) -> bool:
        """Check if all risk filters pass"""
        return all(
            filter_func(df, indicators, market_condition)
            for filter_func in self.risk_filters
        )
    
    def _evaluate_conditions(
        self,
        symbol: str,
        df: pd.DataFrame,
        indicators: Dict,
        market_condition: Dict
    ) -> Optional[Signal]:
        """Evaluate entry and exit conditions"""
        # Check entry conditions
        entry_strength = self._check_conditions(
            df, indicators, market_condition, self.entry_conditions
        )
        
        # Check exit conditions
        exit_strength = self._check_conditions(
            df, indicators, market_condition, self.exit_conditions
        )
        
        # Generate signal based on condition strengths
        if entry_strength > 0.5 and entry_strength > exit_strength:
            return self._create_signal(symbol, df, 'buy', entry_strength, indicators)
        elif exit_strength > 0.5:
            return self._create_signal(symbol, df, 'sell', exit_strength, indicators)
            
        return None
    
    def _check_conditions(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        market_condition: Dict,
        conditions: List[Callable]
    ) -> float:
        """Check conditions and return strength (0 to 1)"""
        if not conditions:
            return 0
            
        strengths = [
            condition(df, indicators, market_condition)
            for condition in conditions
        ]
        
        return sum(strengths) / len(strengths)
    
    def _create_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        direction: str,
        strength: float,
        indicators: Dict
    ) -> Signal:
        """Create a trading signal with stop loss and take profit"""
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        # Calculate stop loss and take profit based on ATR
        if direction == 'buy':
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
        else:
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
        
        return Signal(
            symbol=symbol,
            timestamp=df.index[-1],
            direction=direction,
            strength=strength,
            trade_type='entry',
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'indicators': indicators,
                'atr': atr
            }
        )
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

class MovingAverageCrossover(Strategy):
    """
    Example strategy implementing moving average crossover.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        short_window = self.parameters.get('short_window', 20)
        long_window = self.parameters.get('long_window', 50)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals

class RSIStrategy(Strategy):
    """
    Example strategy implementing RSI-based trading.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.parameters.get('period', 14)
        overbought = self.parameters.get('overbought', 70)
        oversold = self.parameters.get('oversold', 30)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals[rsi < oversold] = 1  # Buy signal
        signals[rsi > overbought] = -1  # Sell signal
        
        return signals

class BollingerBandsStrategy(Strategy):
    """
    Trading strategy based on Bollinger Bands.
    Generates signals when price crosses bands.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.parameters.get('period', 20)
        std_dev = self.parameters.get('std_dev', 2)
        mean_reversion = self.parameters.get('mean_reversion', True)
        
        # Calculate Bollinger Bands
        middle_band = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        signals = pd.Series(0, index=data.index)
        
        if mean_reversion:
            # Mean reversion strategy: Buy at lower band, sell at upper band
            signals[data['close'] <= lower_band] = 1
            signals[data['close'] >= upper_band] = -1
        else:
            # Trend following: Buy above upper band, sell below lower band
            signals[data['close'] >= upper_band] = 1
            signals[data['close'] <= lower_band] = -1
            
        return signals

class MACDStrategy(Strategy):
    """
    Trading strategy based on MACD (Moving Average Convergence Divergence).
    Generates signals on MACD crossovers and divergences.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_period = self.parameters.get('fast_period', 12)
        slow_period = self.parameters.get('slow_period', 26)
        signal_period = self.parameters.get('signal_period', 9)
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals on crossovers
        signals[macd > signal_line] = 1
        signals[macd < signal_line] = -1
        
        # Optional: Check for divergences
        if self.parameters.get('use_divergence', False):
            price_highs = self._get_local_maxima(data['close'])
            macd_highs = self._get_local_maxima(macd)
            
            # Bearish divergence
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                if price_highs[-1] > price_highs[-2] and macd_highs[-1] < macd_highs[-2]:
                    signals.iloc[-1] = -1
                    
        return signals
    
    def _get_local_maxima(self, series: pd.Series, window: int = 5) -> list:
        """Helper function to find local maxima in a series."""
        maxima = []
        for i in range(window, len(series) - window):
            if series[i-window:i].max() < series[i] > series[i:i+window].max():
                maxima.append(series[i])
        return maxima

class IchimokuStrategy(Strategy):
    """
    Trading strategy based on Ichimoku Cloud indicators.
    Generates signals based on multiple Ichimoku components.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        conversion_period = self.parameters.get('conversion_period', 9)
        base_period = self.parameters.get('base_period', 26)
        span_period = self.parameters.get('span_period', 52)
        
        # Calculate Ichimoku components
        high_values = data['high']
        low_values = data['low']
        
        # Tenkan-sen (Conversion Line)
        conversion_line = self._get_midpoint(high_values, low_values, conversion_period)
        
        # Kijun-sen (Base Line)
        base_line = self._get_midpoint(high_values, low_values, base_period)
        
        # Senkou Span A (Leading Span A)
        span_a = ((conversion_line + base_line) / 2).shift(base_period)
        
        # Senkou Span B (Leading Span B)
        span_b = self._get_midpoint(high_values, low_values, span_period).shift(base_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on multiple conditions
        for i in range(len(data)):
            if i < base_period:
                continue
                
            # Trend signals
            if (conversion_line[i] > base_line[i] and 
                data['close'][i] > span_a[i] and 
                data['close'][i] > span_b[i]):
                signals[i] = 1
            elif (conversion_line[i] < base_line[i] and 
                  data['close'][i] < span_a[i] and 
                  data['close'][i] < span_b[i]):
                signals[i] = -1
                
        return signals
    
    def _get_midpoint(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Calculate midpoint price for a given period."""
        return (high.rolling(window=period).max() + 
                low.rolling(window=period).min()) / 2

class VolumeWeightedMAStrategy(Strategy):
    """
    Trading strategy based on Volume Weighted Moving Average (VWMA).
    Incorporates volume into moving average calculations.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        short_period = self.parameters.get('short_period', 10)
        long_period = self.parameters.get('long_period', 30)
        volume_threshold = self.parameters.get('volume_threshold', 1.5)
        
        # Calculate VWMAs
        short_vwma = self._calculate_vwma(data, short_period)
        long_vwma = self._calculate_vwma(data, long_period)
        
        signals = pd.Series(0, index=data.index)
        
        # Volume surge detection
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_surge = data['volume'] > (volume_ma * volume_threshold)
        
        # Generate signals with volume confirmation
        signals[(short_vwma > long_vwma) & volume_surge] = 1
        signals[(short_vwma < long_vwma) & volume_surge] = -1
        
        return signals
    
    def _calculate_vwma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Moving Average."""
        return (data['close'] * data['volume']).rolling(window=period).sum() / \
               data['volume'].rolling(window=period).sum()

class AdaptiveMovingAverageStrategy(Strategy):
    """
    Trading strategy using Kaufman's Adaptive Moving Average (KAMA).
    Adjusts moving average based on market volatility.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.parameters.get('period', 10)
        fast_ef = self.parameters.get('fast_ef', 2)
        slow_ef = self.parameters.get('slow_ef', 30)
        
        # Calculate Kaufman's Adaptive Moving Average
        close = data['close']
        change = abs(close - close.shift(period))
        volatility = abs(close - close.shift(1)).rolling(window=period).sum()
        
        # Efficiency Ratio
        er = change / volatility
        
        # Smoothing Constant
        fast_sc = 2/(fast_ef + 1)
        slow_sc = 2/(slow_ef + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period - 1] = close.iloc[period - 1]
        
        for i in range(period, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on KAMA crossovers
        kama_slow = kama.shift(1)
        signals[kama > kama_slow] = 1
        signals[kama < kama_slow] = -1
        
        return signals

class DualThrustStrategy(Strategy):
    """
    Implementation of the Dual Thrust trading strategy.
    Uses price ranges to generate trading bands.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period = self.parameters.get('period', 4)
        k1 = self.parameters.get('k1', 0.7)  # Upper band multiplier
        k2 = self.parameters.get('k2', 0.7)  # Lower band multiplier
        
        signals = pd.Series(0, index=data.index)
        
        for i in range(period, len(data)):
            # Calculate ranges
            hh = data['high'][i-period:i].max()  # Highest high
            lc = data['close'][i-period:i].min() # Lowest close
            hc = data['close'][i-period:i].max() # Highest close
            ll = data['low'][i-period:i].min()   # Lowest low
            
            range1 = max(hh - lc, hc - ll)
            
            # Calculate bands
            upper_band = data['open'][i] + k1 * range1
            lower_band = data['open'][i] - k2 * range1
            
            # Generate signals
            if data['high'][i] > upper_band:
                signals[i] = 1
            elif data['low'][i] < lower_band:
                signals[i] = -1
                
        return signals