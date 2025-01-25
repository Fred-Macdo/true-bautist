from abc import ABC, abstractmethod
from codecs import utf_16_be_decode
from dataclasses import dataclass
from math import e
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series
    evaluation_df: pd.DataFrame

class Condition(ABC):
    def __init__(self, indicator: str, comparison: str, value: Any):
        self.indicator = indicator
        self.comparison = comparison
        self.value = value
        

    @abstractmethod
    def evaluate(self, row: pd.Series) -> bool:
        pass

class IndicatorCondition(Condition):
    def __str__(self):
        return f'The indicator type is {self.indicator} and the comparison is {self.comparison} using {self.value}'

    def __repr__(self):
        return f'IndicatorCondition(\'{self.indicator}\', {self.comparison}, {self.value})'
    
    def evaluate(self, row: pd.Series) -> bool:
        # Print available columns for debugging
        print(f"\nEvaluating {self.indicator} condition:")
        print(f"Comparison: {self.comparison}")
        print(f"Value: {self.value}")
        
        # Handle special cases for indicators
        if self.indicator == "MACD" and self.value == "signal":
            macd_val = row.get('macd')
            signal_val = row.get('macdsignal')
            macd_prev = row.get('macd_prev')
            signal_prev = row.get('macdsignal_prev')
            
            print(f"MACD values:")
            print(f"Current MACD: {macd_val}")
            print(f"Current Signal: {signal_val}")
            print(f"Previous MACD: {macd_prev}")
            print(f"Previous Signal: {signal_prev}")
            
            if any(pd.isna([macd_val, signal_val, macd_prev, signal_prev])):
                print("Missing or NaN values detected")
                return False
                
            if self.comparison == "crosses_above":
                result = (macd_prev <= signal_prev) and (macd_val > signal_val)
                print(f"Crosses above evaluation: {result}")
                return result
            elif self.comparison == "crosses_below":
                result = (macd_prev >= signal_prev) and (macd_val < signal_val)
                print(f"Crosses below evaluation: {result}")
                return result
                
        elif self.indicator == "BBANDS" and self.value == "lower":
            price = row.get('close')
            lower = row.get('lowerband')  # lower band
            price_prev = row.get('close_prev')
            lower_prev = row.get('lowerband_prev')
            
            print(f"\nBBands check:")
            print(f"Current price: {price}")
            print(f"Current lower band: {lower}")
            print(f"Previous price: {price_prev}")
            print(f"Previous lower band: {lower_prev}")
            
            if any(pd.isna([price, lower, price_prev, lower_prev])):
                print("Missing or NaN values detected")
                return False
                
            if self.comparison == "crosses_below":
                result = price_prev >= lower_prev and price < lower
                print(f"Crosses below evaluation: {result}")
                return result
            elif self.comparison == "crosses_above":
                result = price_prev <= lower_prev and price > lower
                print(f"Crosses above evaluation: {result}")
                return result
                
        else:
            # Standard indicator comparison
            indicator_value = row.get(self.indicator.lower())
            if pd.isna(indicator_value):
                return False

            if isinstance(self.value, (int, float)):
                compare_value = self.value
            else:
                compare_value = row.get(self.value.lower())
                if pd.isna(compare_value):
                    return False

            if self.comparison == "above":
                print("indicator value: ", indicator_value, " / compare value: ", compare_value)
                return indicator_value > compare_value
            elif self.comparison == "below":
                print("indicator value: ", indicator_value, " / compare value: ", compare_value)
                return indicator_value < compare_value

        return False    

class DataFrameBacktester:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            print(self.config)
        
        # Store configuration
        self.symbol = self.config['symbol']
        self.risk_config = self.config['risk_management']
        
        # Initialize state
        self.position = False
        self.entry_price = 0
        self.position_size = 0

    def _setup_conditions(conditions_config: List[Dict]) -> List[Condition]:
        return [IndicatorCondition(
            indicator=cond['indicator'],
            comparison=cond['comparison'],
            value=cond['value']
        ) for cond in conditions_config]

    def _check_condition(self, row: pd.Series, condition_config: Dict) -> bool:
        """Check a single condition based on configuration"""
        try:
            indicator = condition_config['indicator']
            print("\n")
            print("1. indicator: ", indicator)
            comparison = condition_config['comparison']
            print("2. comparison: ", comparison)
            value = condition_config['value']
            print("3. Value: ", value)
            # Get current and previous values

            # COMPARISON FLOW
            if comparison == "above":
                current_value = row[indicator.lower()]        
                return current_value > value
            elif comparison == "below":
                current_value = row[indicator.lower()]
                return current_value < value
            

            elif comparison == "crosses_above":
                if indicator == "MACD":
                    macd_cross_above = (row['macd'] > row['macd_signal']) & (row['macd_prev'] <= row['macdsignal_prev'])
                    return macd_cross_above
                elif indicator == "BBANDS":
                    bb_cross_above = (row['close'] > row[value]) & (row['close_prev'] <= row[value])
                    return bb_cross_above
           
            elif comparison == "crosses_below":
                if indicator == "MACD":
                    macd_cross_below = (row['macd'] < row['macd_signal']) & (row['macd_prev'] >= row['macdsignal_prev'])
                    return macd_cross_below

                elif indicator == "BBANDS":
                    print(f"({row['close']} < {row[value]}) & ({row['close_prev']} >= {row[value]})")
                    bb_cross_below = (row['close'] < row[value]) & (row['close_prev'] >= row[value])
                    return bb_cross_below

            else:
                print(f"Unknown comparison type: {comparison}")
                return False
                
        except Exception as e:
            print(e)
            print(f"Error checking condition: {str(e)}")
            print(f"Condition config: {condition_config}")
            print(f"Available columns: {row.index.tolist()}")
            return False

    def _check_entry_conditions(self, row: pd.Series) -> bool:
        """Check if all entry conditions are met"""
        return all(
            self._check_condition(row, condition)
            for condition in self.config['entry_conditions']
        )

    def _check_exit_conditions(self, row: pd.Series) -> bool:
        """Check if any exit condition is met"""
        return any(
            self._check_condition(row, condition)
            for condition in self.config['exit_conditions']
        )
    
    def _calculate_position_size(self, row: pd.Series, account_balance: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            if self.risk_config['position_sizing_method'] == 'atr_based':
                # Make sure ATR exists and is not NaN
                if 'atr' not in row or pd.isna(row['atr']):
                    print(f"Warning: ATR is missing or NaN. Available columns: {row.index.tolist()}")
                    return 0.0
                
                risk_amount = account_balance * self.risk_config['risk_per_trade']
                stop_distance = float(row['atr']) * self.risk_config['atr_multiplier']
                
                # Avoid division by zero
                if stop_distance <= 0:
                    print(f"Warning: Invalid stop distance: {stop_distance}")
                    return 0.0
                    
                position_size = risk_amount / stop_distance
                max_position_size = self.risk_config['max_position_size']
                
                print(f"""
    Position Size Calculation:
    - Account Balance: {account_balance}
    - Risk Amount: {risk_amount}
    - ATR: {row['atr']}
    - Stop Distance: {stop_distance}
    - Calculated Position Size: {position_size}
    - Max Position Size: {max_position_size}
                """)
                
                return min(position_size, max_position_size)
                
            elif self.risk_config['position_sizing_method'] == 'fixed':
                return self.risk_config['max_position_size']
            
            elif self.risk_config['position_sizing_method'] == 'risk_based':
                risk_amount = account_balance * self.risk_config['risk_per_trade']
                stop_distance = row['close'] * self.risk_config['stop_loss']
                
                if stop_distance <= 0:
                    print(f"Warning: Invalid stop distance: {stop_distance}")
                    return 0.0
                    
                position_size = risk_amount / stop_distance
                return min(position_size, self.risk_config['max_position_size'])
                
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            print(f"Row data: {row}")
            return 0.0

    def _calculate_pnl(self, entry_price: float, exit_price: float, position_size: float) -> float:
        """Calculate PnL for a trade with validation"""
        try:
            if any(pd.isna([entry_price, exit_price, position_size])):
                print(f"""
    Invalid PnL calculation values:
    - Entry Price: {entry_price}
    - Exit Price: {exit_price}
    - Position Size: {position_size}
                """)
                return 0.0
                
            pnl = (exit_price - entry_price) * position_size
            return pnl
            
        except Exception as e:
            print(f"Error calculating PnL: {str(e)}")
            return 0.0

    def _check_stop_loss(self, row: pd.Series) -> bool:
        """Check if stop loss is hit"""
        if not self.position:
            return False
        return row['close'] <= self.entry_price * (1 - self.risk_config['stop_loss'])

    def _check_take_profit(self, row: pd.Series) -> bool:
        """Check if take profit is hit"""
        if not self.position:
            return False
        return row['close'] >= self.entry_price * (1 + self.risk_config['take_profit'])

    def backtest(self, df: pd.DataFrame, initial_balance: float = 100000.0) -> BacktestResult:
        """
        Run backtest on DataFrame with pre-calculated indicators
        """
        # Create copies of data for manipulation
        data = df.copy()
        
        # Add previous values for crossover calculations
        for col in ['close', 'macd', 'macd_signal', 'bb_lower', 'rsi']:
            if col in data.columns:
                data[f'{col}_prev'] = data[col].shift(1)
        
        # Initialize results tracking
        trades_list = []
        balance = initial_balance
        # Initialize equity as a numpy array with float dtype
        equity = np.full(len(data), initial_balance, dtype=np.float64)
        positions = pd.Series(0, index=data.index)
        signals = pd.Series(None, index=data.index)
        
        # Run backtest
        for i, (idx, row) in enumerate(data.iterrows()):
            try:
                current_equity = balance
                
                # Check for exits if in position
                if self.position:
                    # Calculate unrealized P&L
                    unrealized_pnl = (row['close'] - self.entry_price) * self.position_size
                    current_equity = balance + unrealized_pnl
                    positions[idx] = self.position_size
                    
                    # Check stop loss
                    if self._check_stop_loss(row):
                        trade_pnl = (row['close'] - self.entry_price) * self.position_size
                        balance += trade_pnl
                        trades_list.append({
                            'entry_time': self.entry_time,
                            'exit_time': idx,
                            'entry_price': self.entry_price,
                            'exit_price': row['close'],
                            'quantity': self.position_size,
                            'pnl': trade_pnl,
                            'exit_type': 'stop_loss'
                        })
                        signals[idx] = 'stop_loss_exit'
                        self.position = False
                        equity[i] = float(balance)
                        continue
                    
                    # Check take profit
                    if self._check_take_profit(row):
                        trade_pnl = (row['close'] - self.entry_price) * self.position_size
                        balance += trade_pnl
                        trades_list.append({
                            'entry_time': self.entry_time,
                            'exit_time': idx,
                            'entry_price': self.entry_price,
                            'exit_price': row['close'],
                            'quantity': self.position_size,
                            'pnl': trade_pnl,
                            'exit_type': 'take_profit'
                        })
                        
                        print(f"\nTrade recorded:")
                        print(f"Entry price: {self.entry_price}")
                        print(f"Exit price: {row['close']}")
                        print(f"Position size: {self.position_size}")
                        print(f"PnL: {trade_pnl}")

                        signals[idx] = 'take_profit_exit'
                        self.position = False
                        equity[i] = float(balance)
                        continue
                    
                    # Check regular exit conditions
                    if self._check_exit_conditions(row):
                        trade_pnl = (row['close'] - self.entry_price) * self.position_size
                        balance += trade_pnl
                        trades_list.append({
                            'entry_time': self.entry_time,
                            'exit_time': idx,
                            'entry_price': self.entry_price,
                            'exit_price': row['close'],
                            'quantity': self.position_size,
                            'pnl': trade_pnl,
                            'exit_type': 'signal'
                        })

                        print(f"\nTrade recorded:")
                        print(f"Entry price: {self.entry_price}")
                        print(f"Exit price: {row['close']}")
                        print(f"Position size: {self.position_size}")
                        print(f"PnL: {trade_pnl}")

                        signals[idx] = 'signal_exit'
                        self.position = False
                        equity[i] = float(balance)
                        continue
                
                # Check entry conditions if not in position
                elif self._check_entry_conditions(row):
                    self.position_size = self._calculate_position_size(row, balance)
                    self.position = True
                    self.entry_price = row['close']
                    self.entry_time = idx
                    positions[idx] = self.position_size
                    signals[idx] = 'entry'
                
                # Update equity for current bar
                equity[i] = float(current_equity)
                
            except Exception as e:
                print(f"Error at index {i}: {str(e)}")
                equity[i] = equity[i-1] if i > 0 else initial_balance
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades_list)
        if len(trades_df) > 0:
            trades_df.set_index('entry_time', inplace=True)
        
        # Convert equity array to Series with proper index
        equity_series = pd.Series(equity, index=data.index)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, equity_series, initial_balance)
        
        return BacktestResult(
            trades=trades_df,
            metrics=metrics,
            equity_curve=equity_series,
            positions=positions,
            signals=signals,
            evaluation_df=data
        )

    def _calculate_metrics(self, trades: pd.DataFrame, equity: pd.Series, initial_balance: float) -> Dict[str, float]:
        """Calculate backtest performance metrics"""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Calculate returns and drawdown
        returns = equity.pct_change().dropna()
        drawdown = (equity - equity.cummax()) / equity.cummax()
        
        # Calculate trade metrics
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades),
            "profit_factor": abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf'),
            "total_return": (equity.iloc[-1] - initial_balance) / initial_balance,
            "max_drawdown": abs(drawdown.min()),
            "sharpe_ratio": np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        }
        
        return metrics