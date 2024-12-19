import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Tuple
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

class Strategy:  # Remove ABC inheritance since we only have one strategy class now
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.positions = []
        self.indicators: Dict[str, Callable] = {}
        self.entry_conditions: List[Callable] = []
        self.exit_conditions: List[Callable] = []
        self.risk_filters: List[Callable] = []
        self.position_sizing: Optional[Callable] = None
        
        # Initialize indicator map
        self.indicator_map = {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'wma': self._calculate_wma,
            'hull_ma': self._calculate_hull_ma,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'keltner_channels': self._calculate_keltner_channels,
            'atr': self._calculate_atr,
            'stochastic': self._calculate_stochastic,
            'roc': self._calculate_roc,
            'momentum': self._calculate_momentum,
            'williams_r': self._calculate_williams_r,
            'adx': self._calculate_adx,
            'cci': self._calculate_cci,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap,
            'mfi': self._calculate_mfi,
            'fibonacci': self._calculate_fibonacci_levels
        }
        
        # Initialize indicators from parameters
        if 'indicators' in parameters:
            for name, indicator_config in parameters['indicators'].items():
                indicator_type = indicator_config['type']
                if indicator_type in self.indicator_map:
                    indicator_params = {k: v for k, v in indicator_config.items() if k != 'type'}
                    
                    def make_indicator(indicator_func, fixed_params):
                        def indicator(df):
                            return indicator_func(df, period=fixed_params.get('period'))
                        return indicator
                    
                    self.indicators[name] = make_indicator(
                        self.indicator_map[indicator_type], 
                        indicator_params
                    )

            print(self.indicators)
                    
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
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        try:
            # Convert single DataFrame to dict format
            data_dict = {self.parameters.get('symbol', 'default'): data}
            market_condition = self.parameters.get('market_condition', {'risk_level': 'medium'})
            
            # Generate detailed signals
            detailed_signals = self._generate_detailed_signals(data_dict, market_condition)
            
            # Convert detailed signals to simple signal series (-1, 0, 1)
            if detailed_signals:
                for signal in detailed_signals:
                    if signal.timestamp in signals.index:
                        signals[signal.timestamp] = 1 if signal.direction == 'buy' else -1 if signal.direction == 'sell' else 0
                
        except Exception as e:
            print(f"Error in generate_signals: {e}")
            
        return signals.fillna(0)

    def _generate_detailed_signals(self, data: Dict[str, pd.DataFrame], market_condition: Dict) -> List[Signal]:
        """
        Generate detailed trading signals using multi-factor analysis.
        
        Args:
            data: Dictionary of symbol to DataFrame mappings
            market_condition: Dictionary containing market condition parameters
            
        Returns:
            List of Signal objects containing trade details
        """
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
        """
        Calculate all technical indicators defined in the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping indicator names to their calculated values
        """
        try:
            print(df.head())
            for n, f in self.indicators.items():
                print(n, f) 
            return {
                
                name: func(df) for name, func in self.indicators.items()
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}

    def _pass_risk_filters(self, df: pd.DataFrame, indicators: Dict, market_condition: Dict) -> bool:
        """
        Check if all risk filters pass.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dictionary of calculated indicators
            market_condition: Dictionary containing market condition parameters
            
        Returns:
            Boolean indicating if all risk filters pass
        """
        if not self.risk_filters:
            return True
            
        try:
            return all(
                filter_func(df, indicators, market_condition)
                for filter_func in self.risk_filters
            )
        except Exception as e:
            print(f"Error in risk filters: {e}")
            return False

    def _evaluate_conditions(self, symbol: str, df: pd.DataFrame, indicators: Dict, market_condition: Dict) -> Optional[Signal]:
        """
        Evaluate entry and exit conditions to generate trading signal.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
            indicators: Dictionary of calculated indicators
            market_condition: Dictionary containing market condition parameters
            
        Returns:
            Signal object if conditions are met, None otherwise
        """
        try:
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
        except Exception as e:
            print(f"Error evaluating conditions: {e}")
            return None

    def _check_conditions(self, df: pd.DataFrame, indicators: Dict, market_condition: Dict, conditions: List[Callable]) -> float:
        """
        Check conditions and return average strength.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dictionary of calculated indicators
            market_condition: Dictionary containing market condition parameters
            conditions: List of condition functions to check
            
        Returns:
            Float between 0 and 1 indicating average condition strength
        """
        if not conditions:
            return 0
            
        try:
            strengths = [
                condition(df, indicators, market_condition)
                for condition in conditions
            ]
            
            return sum(strengths) / len(strengths)
        except Exception as e:
            print(f"Error checking conditions: {e}")
            return 0

    def _create_signal(self, symbol: str, df: pd.DataFrame, direction: str, strength: float, indicators: Dict) -> Signal:
        """
        Create a trading signal with stop loss and take profit levels.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
            direction: Trade direction ('buy' or 'sell')
            strength: Signal strength between 0 and 1
            indicators: Dictionary of calculated indicators
            
        Returns:
            Signal object with trade details
        """
        try:
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
        except Exception as e:
            print(f"Error creating signal: {e}")
            return None
        
    # TECHNICAL INDICATORS

    def _calculate_sma(df: pd.DataFrame, sma_period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=sma_period).mean()

    def _calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_wma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return df['close'].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum())

    @staticmethod
    def _calculate_hull_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Hull Moving Average
        HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        """
        half_period = int(period/2)
        sqrt_period = int(np.sqrt(period))
        
        wma1 = _calculate_wma(df, half_period)
        wma2 = _calculate_wma(df, period)
        return _calculate_wma(pd.DataFrame({'close': 2*wma1 - wma2}), sqrt_period)

    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and MACD histogram"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper = middle + std * std_dev
        lower = middle - std * std_dev
        return upper, middle, lower

    def _calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.rolling(window=period).mean()
        atr = calculate_atr(df, period)
        
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        return upper, middle, lower

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()

    def _calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        return k, d

    def _calculate_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        return (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    def _calculate_momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Momentum"""
        return df['close'] - df['close'].shift(period)

    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        wr = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        return wr

    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Plus and Minus Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate Plus and Minus Directional Indicators
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di

    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
        cci = (tp - tp_sma) / (0.015 * mad)
        return cci

    def _calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(0, index=df.index)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    def _calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(df)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            else:
                negative_flow[i] = money_flow[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    def _calculate_fibonacci_levels(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Fibonacci Retracement Levels"""
        high = df['high'].rolling(window=period).max()
        low = df['low'].rolling(window=period).min()
        diff = high - low
        
        return {
            'level_0': high,
            'level_236': high - diff * 0.236,
            'level_382': high - diff * 0.382,
            'level_500': high - diff * 0.500,
            'level_618': high - diff * 0.618,
            'level_100': low
        }
