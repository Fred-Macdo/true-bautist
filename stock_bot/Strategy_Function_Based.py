from typing import Dict, List, Callable, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

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

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], market_condition: Dict) -> List[Signal]:
        """Generate trading signals based on data and market conditions"""
        pass

class MultiFactorStrategy(TradingStrategy):
    def __init__(self):
        """Initialize multi-factor strategy"""
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

    def generate_signals(self, data: Dict[str, pd.DataFrame], market_condition: Dict) -> List[Signal]:
        """Generate trading signals using multi-factor analysis"""
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

# Example technical indicators
def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return df['close'].rolling(window=period).mean()

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and MACD histogram"""
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower

# Example risk management filters
def volatility_filter(max_atr_percent: float) -> Callable:
    """Create a volatility filter"""
    def filter_func(df: pd.DataFrame, indicators: Dict, market_condition: Dict) -> bool:
        atr = MultiFactorStrategy._calculate_atr(df)
        atr_percent = (atr / df['close'].iloc[-1]) * 100
        return atr_percent <= max_atr_percent
    return filter_func

def market_condition_filter(min_risk_score: float) -> Callable:
    """Create a market condition filter"""
    def filter_func(df: pd.DataFrame, indicators: Dict, market_condition: Dict) -> bool:
        risk_score = market_condition.get('risk_level', 'high')
        risk_mapping = {'low': 1, 'medium': 2, 'high': 3}
        return risk_mapping[risk_score] >= min_risk_score
    return filter_func
