from .AlpacaDataManager import AlpacaDataFetcher
from .Indicators import TechnicalIndicators

from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np


class TrueBautistStrategy:
    """
    True Bautist's Strategy class to define a trading strategy using a config file. 
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
        
        self.risk_management = {}
        self.indicators = {}
        self.entry_conditions = []
        self.exit_conditions = []
        self.get_risk_management()
        self.get_indicators()
        self.get_entry_conditions()
        self.get_exit_conditions()

        self.df = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.equity = pd.Series()
        self.metrics = {}   
        

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
        for risk_param, param_value in self.config['risk_management'].items():
            self.risk_management[risk_param] = param_value
        return self.risk_management
        
    def get_keys(self):
        return self.keys
    
    def get_config(self):
        return self.config
    
    def get_entry_conditions(self):
        for condition in self.config['entry_conditions']:
            self.entry_conditions.append(condition)
        return self.entry_conditions
    
    def get_exit_conditions(self):
        for condition in self.config['exit_conditions']:
            self.exit_conditions.append(condition)
        return self.exit_conditions
            
    
    def get_data(self) -> pd.DataFrame:
        """
        Fetch data from the Alpaca API
        """
        cfg = self.config
        # PAPER / LIVE

        if self.keys.get('api_key_paper'):

            data_manager = AlpacaDataFetcher(self.keys['API_KEY'], 
                                            self.keys['API_SECRET'])
        else:
            data_manager = AlpacaDataFetcher(self.keys['API_KEY'], 
                                            self.keys['API_SECRET'])

        print("Getting data for these symbols: ", cfg['symbols'])

        # FIX END DATE
        end_date = cfg['end_date']
        if end_date in [None,'']:
            end_date = datetime.now(timezone.utc)
        else:
            end_date = self._parse_eastern_date(end_date)

        # FIX START DATE
        start_date = cfg['start_date']
        if start_date in [None,'']:
            today = datetime.now(timezone.utc)
            start_date = today - timedelta(days=30)
        else:
            start_date = self._parse_eastern_date(start_date)
             
            response, data = data_manager.get_historical_data(
                cfg['symbols'], 
                cfg['timeframe'],
                start_date,
                end_date
                )
        
        self.df = self.calculate_indicators(data)
        self.df.reset_index(inplace=True)
        self.alpaca_response = response
        return self.df

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

    def _parse_eastern_date(self, date_str):
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
        return dt

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
