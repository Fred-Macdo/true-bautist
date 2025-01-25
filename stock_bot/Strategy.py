from .AlpacaDataManager import AlpacaDataFetcher
from .Indicators import TechnicalIndicators

from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np


class Strategy:
    """
    Strategy class to define a trading strategy using a config file. 
    The config file should contain the following keys:

    - symbols: List of symbols to trade (e.g. ['AAPL', 'MSFT'])
    - timeframe: Timeframe for historical data (e.g. '1D', '1H')
    - start_date: Start date for historical data (e.g. '2021-01-01')
    - end_date: End date for historical data (e.g. '2021-12-31')
    - indicators: List of indicators to calculate
    - entry_conditions: List of conditions for entering a trade
    - exit_conditions: List of conditions for exiting a trade
    - risk_management: List of risk management rules
    
    The keys file should contain the following keys:
    - api_key: Alpaca API key for paper trading
    - api_secret: Alpaca API secret for paper trading
    """
    def __init__(self, config: Dict[str, Any], keys: Dict[str, str]):
        self.config = config
        self.keys = keys
        self.positions = []
        self.indicators: Dict[str, Callable] = {}
        self.entry_conditions: List[Callable] = []
        self.exit_conditions: List[Callable] = []
        self.risk_management: List[Callable] = []
        self.position_sizing: Optional[Callable] = None
        self.df = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.equity = pd.Series()
        self.metrics = {}   
        self.indicators = self.get_indicators()
        

    def get_indicators(self):
        '''
        Parse through list of indicators in the config
        '''
        for indicator in self.config['indicators']:
            # Use the indicator + period value to instantiate pd.Series 
            # for EMA SMA calculations
            # Else:  
            if indicator['name'].lower() in ['ema', 'sma']:
                key = f"{indicator['name'].lower()}_{str(indicator['params'].get('period'))}"
                self.indicators[key] = indicator['params']
            else:
                self.indicators[indicator['name'].lower()] = indicator['params']

        return self.indicators
    
    def get_risk_management(self):
        '''
        Parse through list of risk management strategies in the config
        '''
        for risk in self.config['risk_management']:
            self.risk_management.append(risk)
        return self.risk_management
    
    def get_data(self) -> pd.DataFrame:
        """
        Fetch data from the Alpaca API
        """
        cfg = self.config
        data_manager = AlpacaDataFetcher(self.keys['api_key'], 
                                            self.keys['api_secret'])

        print("Getting data for these symbols: ", cfg['symbols'])
        data_list = []
        for symbol in cfg['symbols']:
            data = data_manager.get_historical_data(
                symbol, 
                cfg['timeframe'],
                datetime.strptime(cfg['start_date'], "%Y-%m-%d"),
                datetime.strptime(cfg['end_date'], "%Y-%m-%d")
                )
            
            data_list.append(data)
        
        df = pd.concat(data_list)
        
        self.df = self.calculate_indicators(df)
        self.df.reset_index(inplace=True)
        print(self.df.head())
        return self.df
    
    def get_keys(self):
        return self.keys
    
    def get_config(self):
        return self.config

    def calculate_indicators(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators for the strategy
        """
        try:
            technicals = TechnicalIndicators(df, self.indicators)
            self.df = technicals.calculate_indicators()

        except: 
            Exception("Error calculating indicators")
        return self.df

    
    def _check_condition(self, row: pd.Series, condition_config: Dict) -> bool:
        """
        Checking the entry / exit condition for a given row
        """
        
        comparison_values = ['above', 'below', 'between', 'crosses_above', 'crosses_below']
        if condition_config['comparison'] not in comparison_values:
            raise Exception(f"Comparison value {condition_config['comparison']} not a valid comparison operator" )

        indicator = condition_config['indicator']
        comparison = condition_config['comparison']
        value = condition_config['value']
        
        # Value can be either a number or another indicator
        if (comparison == "above") & isinstance(value, str) :
            above = row[indicator.lower()] > row[value.lower()]
            return above
        elif (comparison == "above") & isinstance(value, (int, float)):
            above = row[indicator.lower()] > value
            return above

        # Value can be either a number or another indicator
        if (comparison == "below") & isinstance(value, str) :
            below = row[indicator.lower()] < row[value.lower()]
            return below
        elif (comparison == "below") & isinstance(value, int):
            below = row[indicator.lower()] < value
            return below

        if comparison == "crosses_above":
            if indicator == "MACD":
                macd_cross_above = (row['macd'] > row['macd_signal']) & (row['macd_prev'] <= row['macdsignal_prev'])
                return macd_cross_above
            elif indicator == "BBANDS":
                bb_cross_above = (row['close'] > row[value.lower()]) & (row['close_prev'] <= row[value.lower()])
                return bb_cross_above
            else:
                if isinstance(value, str):
                    indicator_cross_above = (row[indicator.lower()] > row[value.lower()]) & (row[f"{indicator.lower()}_prev"] <= row[f"{value.lower()}_prev"])
                    return indicator_cross_above
                elif isinstance(value, (int, float)):
                    indicator_cross_above = (row[indicator.lower()] > value) & (row[f"{indicator.lower()}_prev"] <= value)
                    return indicator_cross_above
                return indicator_cross_above
            
        elif comparison == "crosses_below":
            if indicator == "MACD":
                macd_cross_below = (row['macd'] < row['macd_signal']) & (row['macd_prev'] >= row['macdsignal_prev'])
                return macd_cross_below

            elif indicator == "BBANDS":
                bb_cross_below = (row['close'] < row[value]) & (row['close_prev'] >= row[value])
                return bb_cross_below
            else:
                indicator_cross_below = (row[indicator.lower()] < row[value.lower()]) & (row[f"{indicator.lower()}_prev"] >= row[f"{value.lower()}_prev"])
                return indicator_cross_below
            
        elif indicator == 'between':
            if any(isinstance(x, (int, float)) for x in value):
                indicator_between = row.between(value[0], value[1], inclusive="both")
                return indicator_between
            else:
                indicator_between = row.between(row[value[0]], row[value[1]], inclusive="both")
                return indicator_between

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
        """
        Calculate position size based on risk management rules, ensuring non-negative position sizes
        
        Args:
            row: DataFrame row containing price and indicator data
            account_balance: Current account balance
            
            
        Returns:
            float: Calculated position size, always >= 0
        """
        try:
            # Ensure account balance is positive
            account_balance = abs(account_balance)
            risk_config = self.config['risk_management']
            if ['position_sizing_method'] == 'atr_based':
                # Make sure ATR exists and is not NaN
                if 'atr' not in row or pd.isna(row['atr']):
                    print(f"Warning: ATR is missing or NaN. Available columns: {row.index.tolist()}")
                    return 0.0
                    
                risk_amount = account_balance * abs(risk_config['risk_per_trade'])
                stop_distance = abs(float(row['atr'])) * abs(risk_config['atr_multiplier'])
                
                # Avoid division by zero and ensure positive stop distance
                if stop_distance <= 0:
                    print(f"Warning: Invalid stop distance calculated: {stop_distance}")
                    return 0.0
                    
                position_size = risk_amount / stop_distance
                max_position_size = abs(risk_config['max_position_size'])
                
                print(f"""
    Position Size Calculation:
    - Account Balance: {account_balance}
    - Risk Amount: {risk_amount}
    - ATR: {abs(row['atr'])}
    - Stop Distance: {stop_distance}
    - Calculated Position Size: {position_size}
    - Max Position Size: {max_position_size}
                """)
                
                return min(position_size, max_position_size)
                
            elif risk_config['position_sizing_method'] == 'fixed':
                return abs(risk_config['max_position_size'])
            
            elif risk_config['position_sizing_method'] == 'risk_based':
                risk_amount = account_balance * abs(risk_config['risk_per_trade'])
                stop_distance = abs(row['close']) * abs(risk_config['stop_loss'])
                
                if stop_distance <= 0:
                    print(f"Warning: Invalid stop distance calculated: {stop_distance}")
                    return 0.0
                    
                position_size = risk_amount / stop_distance
                max_position_size = abs(risk_config['max_position_size'])
                
                return min(position_size, max_position_size)
                
            else:
                print(f"Warning: Unknown position sizing method: {risk_config['position_sizing_method']}")
                return 0.0
                
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

    def _check_stop_loss(self, row: pd.Series, entry_price, position) -> bool:
        """Check if stop loss is hit"""
        risk_config = self.config['risk_management']
        if not position:
            return False
        return row['close'] <= entry_price * (1 - risk_config['stop_loss'])

    def _check_take_profit(self, row: pd.Series, entry_price, position) -> bool:
        """Check if take profit is hit"""
        risk_config = self.config['risk_management']
        if not position:
            return False
        return row['close'] >= entry_price * (1 + risk_config['take_profit'])

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
                "win_rate": round(len(winning_trades) / len(trades), 2),
                "profit_factor": round(abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf'),2),
                "total_return":round((equity.iloc[-1] - initial_balance) / initial_balance, 2),
                "max_drawdown": round(abs(drawdown.min()), 4),
                "sharpe_ratio": round(np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0, 2)
            }
            
            return metrics
