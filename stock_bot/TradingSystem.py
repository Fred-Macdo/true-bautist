from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import yaml
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class Signal:
    timestamp: datetime
    signal_type: SignalType
    symbol: str
    price: float
    quantity: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 1.0)
        self.max_drawdown = config.get('max_drawdown', 0.02)
        self.stop_loss = config.get('stop_loss', 0.01)
        self.take_profit = config.get('take_profit', 0.03)
        self.position_sizing_method = config.get('position_sizing_method', 'fixed')
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
    def calculate_position_size(self, 
                              account_balance: float,
                              current_price: float,
                              atr: Optional[float] = None) -> float:
        if self.position_sizing_method == 'fixed':
            return self.max_position_size
        elif self.position_sizing_method == 'risk_based':
            risk_amount = account_balance * self.risk_per_trade
            stop_distance = current_price * self.stop_loss
            return risk_amount / stop_distance
        elif self.position_sizing_method == 'atr_based' and atr is not None:
            risk_amount = account_balance * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier
            return risk_amount / stop_distance
        return self.max_position_size

    def check_stop_loss(self, entry_price: float, current_price: float, position_type: str) -> bool:
        if position_type == 'long':
            return current_price <= entry_price * (1 - self.stop_loss)
        return current_price >= entry_price * (1 + self.stop_loss)

    def check_take_profit(self, entry_price: float, current_price: float, position_type: str) -> bool:
        if position_type == 'long':
            return current_price >= entry_price * (1 + self.take_profit)
        return current_price <= entry_price * (1 - self.take_profit)

class Condition(ABC):
    def __init__(self, indicator_name: str, params: Dict[str, Any], comparison: str, threshold: float):
        self.indicator = IndicatorFactory.create(indicator_name, params)
        self.comparison = comparison
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, current_idx: int) -> bool:
        pass

class PriceCondition(Condition):
    def __init__(self, indicator_name: str, params: Dict[str, Any], comparison: str, threshold: Any):
        self.indicator = IndicatorFactory.create(indicator_name, params)
        self.comparison = comparison
        self.threshold = threshold
        self.indicator_name = indicator_name

    def evaluate(self, data: pd.DataFrame, current_idx: int) -> bool:
        try:
            indicator_values = self.indicator.calculate(data)
            
            # Special handling for Bollinger Bands
            if self.indicator_name == "Bollinger":
                if isinstance(indicator_values, pd.DataFrame):  # Bollinger returns DataFrame with 'mid', 'upper', 'lower'
                    if self.threshold == "lower":
                        threshold_value = indicator_values['lower'].iloc[current_idx]
                        current_value = data['close'].iloc[current_idx]
                    elif self.threshold == "upper":
                        threshold_value = indicator_values['upper'].iloc[current_idx]
                        current_value = data['close'].iloc[current_idx]
                    else:
                        threshold_value = float(self.threshold)
                        current_value = indicator_values['mid'].iloc[current_idx]
                    
                    if self.comparison == "above":
                        return current_value > threshold_value
                    elif self.comparison == "below":
                        return current_value < threshold_value
                    elif self.comparison == "crosses_above":
                        if current_idx == 0:
                            return False
                        prev_value = data['close'].iloc[current_idx - 1]
                        prev_threshold = indicator_values['lower'].iloc[current_idx - 1]
                        return prev_value <= prev_threshold and current_value > threshold_value
                    elif self.comparison == "crosses_below":
                        if current_idx == 0:
                            return False
                        prev_value = data['close'].iloc[current_idx - 1]
                        prev_threshold = indicator_values['upper'].iloc[current_idx - 1]
                        return prev_value >= prev_threshold and current_value < threshold_value
            else:
                # Regular indicator handling
                if current_idx >= len(indicator_values):
                    return False
                
                current_value = indicator_values.iloc[current_idx]
                threshold_value = float(self.threshold)
                
                if self.comparison == "above":
                    return current_value > threshold_value
                elif self.comparison == "below":
                    return current_value < threshold_value
                elif self.comparison == "crosses_above":
                    if current_idx == 0:
                        return False
                    prev_value = indicator_values.iloc[current_idx - 1]
                    return prev_value <= threshold_value and current_value > threshold_value
                elif self.comparison == "crosses_below":
                    if current_idx == 0:
                        return False
                    prev_value = indicator_values.iloc[current_idx - 1]
                    return prev_value >= threshold_value and current_value < threshold_value
                    
            return False
            
        except Exception as e:
            print(f"Error evaluating condition: {str(e)}")
            return False
        
class Indicator(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the indicator values for the given data"""
        pass

class IndicatorFactory:
    _indicators = {}

    @classmethod
    def register(cls, name: str):
        def decorator(indicator_class):
            cls._indicators[name] = indicator_class
            return indicator_class
        return decorator

    @classmethod
    def create(cls, name: str, params: Dict[str, Any]) -> Indicator:
        if name not in cls._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        return cls._indicators[name](params)

@IndicatorFactory.register("SMA")
class SimpleMovingAverage(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].rolling(window=self.params['window']).mean()

@IndicatorFactory.register("RSI")
class RSI(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['window']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['window']).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

@IndicatorFactory.register("MACD")
class MACD(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        fast = data['close'].ewm(span=self.params.get('fast_period', 12)).mean()
        slow = data['close'].ewm(span=self.params.get('slow_period', 26)).mean()
        macd = fast - slow
        signal = macd.ewm(span=self.params.get('signal_period', 9)).mean()
        return macd - signal

@IndicatorFactory.register("ATR")
class AverageTrueRange(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.params.get('window', 14)).mean()

@IndicatorFactory.register("Bollinger")
class BollingerBands(Indicator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        sma = data['close'].rolling(window=self.params.get('window', 20)).mean()
        std = data['close'].rolling(window=self.params.get('window', 20)).std()
        upper = sma + (std * self.params.get('std_dev', 2))
        lower = sma - (std * self.params.get('std_dev', 2))
        return pd.concat([sma, upper, lower], axis=1, keys=['mid', 'upper', 'lower'])

@dataclass
class BacktestResult:
    """Container for backtest results"""
    signals: List[Signal]
    metrics: Dict[str, float]
    history: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.Series

class Strategy:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.entry_conditions = self._setup_conditions(self.config['entry_conditions'])
        self.exit_conditions = self._setup_conditions(self.config['exit_conditions'])
        
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.mode = self.config['mode']  # 'live' or 'backtest'
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        
        # Initialize position tracking
        self.position = False
        self.entry_price = None
        self.position_size = 0
        self.current_trade = None

    def _setup_conditions(self, conditions_config: List[Dict]) -> List[Condition]:
        """Set up entry or exit conditions from config"""
        conditions = []
        for condition in conditions_config:
            conditions.append(PriceCondition(
                indicator_name=condition['indicator'],
                params=condition['params'],
                comparison=condition['comparison'],
                threshold=condition['threshold']
            ))
        return conditions

    def _check_conditions(self, conditions: List[Condition], data: pd.DataFrame, current_idx: int) -> bool:
        """Check if all conditions are met"""
        try:
            #print("Data:", data, '\n', "Current Index",)
            return all(condition.evaluate(data, current_idx) for condition in conditions)
        except Exception as e:
            print(f"Error checking conditions: {str(e)}")
            return False

    def _calculate_backtest_metrics(self, trades: List[Dict], equity: List[float], initial_balance: float) -> Dict:
        """Calculate backtest performance metrics"""
        if not trades:
            return {
                "total_trades": 0,
                "profit_factor": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_return": 0
            }
            
        # Calculate basic metrics
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        # Calculate returns and drawdown
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        total_return = (equity_series.iloc[-1] - initial_balance) / initial_balance
        
        # Calculate max drawdown
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_balance": equity[-1]
        }

    def backtest(self, historical_data: pd.DataFrame, initial_balance: float = 100000.0) -> BacktestResult:
        """Run backtest on historical data"""
        if self.mode != 'backtest':
            raise ValueError("Strategy not configured for backtesting")
        
        signals = []
        trades_list = []
        balance = initial_balance
        equity = [initial_balance]
        
        # Create a copy of historical data
        df = historical_data.copy()
        
        # Add columns for tracking
        df['position'] = 0
        df['equity'] = initial_balance
        df['returns'] = 0.0
        df['drawdown'] = 0.0
        df['signal'] = None
        
        for i in range(len(df)):
            timestamp = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check for risk management exits if in position
            if self.position:
                # Calculate unrealized P&L
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
                current_equity = balance + unrealized_pnl
                df.loc[timestamp, 'position'] = self.position_size
                
                # Check stop loss
                if self.risk_manager.check_stop_loss(self.entry_price, current_price, 'long'):
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=self.position_size,
                        metadata={'type': 'stop_loss'}
                    )
                    signals.append(signal)
                    df.loc[timestamp, 'signal'] = 'stop_loss_exit'
                    
                    # Record trade
                    trade_pnl = (current_price - self.entry_price) * self.position_size
                    balance += trade_pnl
                    trades_list.append({
                        'entry_time': self.current_trade,
                        'exit_time': timestamp,
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'quantity': self.position_size,
                        'pnl': trade_pnl,
                        'return': trade_pnl / (self.entry_price * self.position_size),
                        'exit_type': 'stop_loss'
                    })
                    
                    self.position = False
                    self.position_size = 0
                    continue

                # Check take profit
                if self.risk_manager.check_take_profit(self.entry_price, current_price, 'long'):
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=self.position_size,
                        metadata={'type': 'take_profit'}
                    )
                    signals.append(signal)
                    df.loc[timestamp, 'signal'] = 'take_profit_exit'
                    
                    # Record trade
                    trade_pnl = (current_price - self.entry_price) * self.position_size
                    balance += trade_pnl
                    trades_list.append({
                        'entry_time': self.current_trade,
                        'exit_time': timestamp,
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'quantity': self.position_size,
                        'pnl': trade_pnl,
                        'return': trade_pnl / (self.entry_price * self.position_size),
                        'exit_type': 'take_profit'
                    })
                    
                    self.position = False
                    self.position_size = 0
                    continue

            # Entry conditions
            if not self.position:
                if self._check_conditions(self.entry_conditions, df, i):
                    position_size = self.risk_manager.calculate_position_size(
                        account_balance=balance,
                        current_price=current_price
                    )
                    
                    self.position = True
                    self.entry_price = current_price
                    self.position_size = position_size
                    self.current_trade = timestamp
                    
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=position_size,
                        metadata={'type': 'entry'}
                    )
                    signals.append(signal)
                    df.loc[timestamp, 'signal'] = 'entry'
                    df.loc[timestamp, 'position'] = position_size
            
            # Exit conditions if in position
            elif self._check_conditions(self.exit_conditions, df, i):
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SELL,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=self.position_size,
                    metadata={'type': 'regular_exit'}
                )
                signals.append(signal)
                df.loc[timestamp, 'signal'] = 'regular_exit'
                
                # Record trade
                trade_pnl = (current_price - self.entry_price) * self.position_size
                balance += trade_pnl
                trades_list.append({
                    'entry_time': self.current_trade,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'quantity': self.position_size,
                    'pnl': trade_pnl,
                    'return': trade_pnl / (self.entry_price * self.position_size),
                    'exit_type': 'regular_exit'
                })
                
                self.position = False
                self.position_size = 0
            
            # Update equity and metrics for each bar
            current_equity = balance
            if self.position:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
                current_equity += unrealized_pnl
            
            equity.append(current_equity)
            df.loc[timestamp, 'equity'] = current_equity
        
        # Calculate returns and drawdown
        df['returns'] = df['equity'].pct_change()
        peak = df['equity'].expanding(min_periods=1).max()
        df['drawdown'] = (df['equity'] - peak) / peak
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades_list)
        if len(trades_df) > 0:
            trades_df.set_index('entry_time', inplace=True)
        
        # Calculate final metrics
        metrics = self._calculate_backtest_metrics(trades_list, df['equity'].values, initial_balance)
        
        equity = equity.pop(0)

        return BacktestResult(
            signals=signals,
            metrics=metrics,
            history=df,
            trades=trades_df,
            equity_curve=pd.Series(equity, index=df.index)
        )

    def update(self, new_data: pd.DataFrame) -> List[Signal]:
        """Update strategy with new data (for live trading)"""
        if self.mode != 'live':
            raise ValueError("Strategy not configured for live trading")

    def _setup_conditions(self, conditions_config: List[Dict]) -> List[Condition]:
        conditions = []
        for condition in conditions_config:
            conditions.append(PriceCondition(
                indicator_name=condition['indicator'],
                params=condition['params'],
                comparison=condition['comparison'],
                threshold=condition['threshold']
            ))
        return conditions

    def generate_signals(self, data: pd.DataFrame, account_balance: float) -> List[Signal]:
        signals = []
        
        # Calculate ATR for position sizing if needed
        atr = None
        if self.risk_manager.position_sizing_method == 'atr_based':
            atr_indicator = AverageTrueRange({'window': 14})
            atr = atr_indicator.calculate(data).iloc[-1]

        for i in range(len(data)):
            timestamp = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Check risk management exits if in position
            if self.position:
                if self.risk_manager.check_stop_loss(self.entry_price, current_price, self.position_type):
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=self.position_size,
                        metadata={'type': 'stop_loss'}
                    ))
                    self.position = False
                    continue

                if self.risk_manager.check_take_profit(self.entry_price, current_price, self.position_type):
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=self.position_size,
                        metadata={'type': 'take_profit'}
                    ))
                    self.position = False
                    continue

            if not self.position:
                # Check entry conditions
                if self._check_conditions(self.entry_conditions, data, i):
                    position_size = self.risk_manager.calculate_position_size(
                        account_balance=account_balance,
                        current_price=current_price,
                        atr=atr
                    )
                    
                    self.position = True
                    self.entry_price = current_price
                    self.position_type = 'long'
                    self.position_size = position_size
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=position_size,
                        metadata={'type': 'entry'}
                    ))
            else:
                # Check regular exit conditions
                if self._check_conditions(self.exit_conditions, data, i):
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=self.position_size,
                        metadata={'type': 'regular_exit'}
                    ))
                    self.position = False

        return signals
