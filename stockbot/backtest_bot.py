from Strategy import *
import yaml
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from DataManager import *

class Backtest:

    """
    Main backtesting engine that processes strategies and generates results.
    """
    def __init__(self, config_path: str):
        """
        Initialize the backtester with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        # Initialize market data manager first
        self.market_data = MarketData(self.config.get('data', {}))
        self.strategies = self._initialize_strategies()
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and parse the YAML configuration file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Ensure required config sections exist
            required_sections = ['data', 'strategies']
            for section in required_sections:
                if section not in config:
                    config[section] = {}
                    
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def _initialize_strategies(self) -> Dict[str, Strategy]:
        """Initialize strategy objects based on configuration."""
        strategy_map = {
            'MovingAverageCrossover': MovingAverageCrossover,
            'RSIStrategy': RSIStrategy,
            'BollingerBandsStrategy': BollingerBandsStrategy,
            'MACDStrategy': MACDStrategy,
            'IchimokuStrategy': IchimokuStrategy,
            'VolumeWeightedMAStrategy': VolumeWeightedMAStrategy,
            'AdaptiveMovingAverageStrategy': AdaptiveMovingAverageStrategy,
            'DualThrustStrategy': DualThrustStrategy
        }
        
        strategies = {}
        for strategy_name, strategy_config in self.config['strategies'].items():
            strategy_class = strategy_map[strategy_config['type']]
            strategies[strategy_name] = strategy_class(strategy_config['parameters'])
        
        return strategies

    def _get_all_symbols(self) -> List[str]:
        """Get all unique symbols from the configuration."""
        symbols = set()
        
        # Add symbols from universe configuration
        if self.config['data']['universe']['type'] == 'static':
            symbols.update(self.config['data']['universe']['symbols'])
        
        # Add symbols from individual strategies
        for strategy_config in self.config['strategies'].values():
            if 'symbols' in strategy_config:
                symbols.update(strategy_config['symbols'])
        
        return list(symbols)

    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate detailed drawdown metrics."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        
        # Calculate drawdown duration
        is_drawdown = drawdowns < 0
        drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
        drawdown_duration = is_drawdown.groupby(drawdown_start).cumsum()
        
        return {
            'max_drawdown': drawdowns.min(),
            'avg_drawdown': drawdowns[drawdowns < 0].mean(),
            'max_drawdown_duration': drawdown_duration.max(),
            'avg_drawdown_duration': drawdown_duration[drawdown_duration > 0].mean(),
            'drawdown_frequency': (drawdowns < 0).mean()
        }

    def _calculate_trade_metrics(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate trade-specific metrics."""
        trades = signals.diff().fillna(0)
        trades = trades[trades != 0]  # Only consider actual trades
        
        trade_returns = []
        current_position = 0
        entry_price = None
        
        for date, signal in trades.items():
            if signal != 0:
                if current_position == 0:  # Opening position
                    entry_price = date
                else:  # Closing position
                    if entry_price is not None:
                        trade_return = returns[entry_price:date].sum()
                        trade_returns.append(trade_return)
                    entry_price = None if signal == 0 else date
                current_position = signal

        trade_returns = pd.Series(trade_returns)
        
        return {
            'total_trades': len(trades),
            'win_rate': (trade_returns > 0).mean() if len(trade_returns) > 0 else 0,
            'avg_trade_return': trade_returns.mean() if len(trade_returns) > 0 else 0,
            'best_trade': trade_returns.max() if len(trade_returns) > 0 else 0,
            'worst_trade': trade_returns.min() if len(trade_returns) > 0 else 0,
            'avg_winning_trade': trade_returns[trade_returns > 0].mean() if len(trade_returns) > 0 else 0,
            'avg_losing_trade': trade_returns[trade_returns < 0].mean() if len(trade_returns) > 0 else 0,
            'profit_factor': abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()) if len(trade_returns) > 0 and trade_returns[trade_returns < 0].sum() != 0 else 0
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        excess_returns = returns - daily_rf
        negative_returns = returns[returns < 0]
        
        return {
            'sharpe_ratio': np.sqrt(252) * (returns.mean() - daily_rf) / returns.std() if returns.std() != 0 else 0,
            'sortino_ratio': np.sqrt(252) * (returns.mean() - daily_rf) / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() != 0 else 0,
            'calmar_ratio': (returns.mean() * 252) / abs(self._calculate_drawdown_metrics(returns)['max_drawdown']) if self._calculate_drawdown_metrics(returns)['max_drawdown'] != 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'omega_ratio': returns[returns > 0].mean() / abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 and returns[returns < 0].mean() != 0 else 0
        }

    def _calculate_timing_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate market timing and consistency metrics."""
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=returns.index)  # Use zero returns as benchmark if none provided
            
        rolling_beta = returns.rolling(window=63).cov(benchmark_returns) / benchmark_returns.rolling(window=63).var()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_return_std': monthly_returns.std(),
            'positive_months': (monthly_returns > 0).mean(),
            'avg_up_month': monthly_returns[monthly_returns > 0].mean() if len(monthly_returns[monthly_returns > 0]) > 0 else 0,
            'avg_down_month': monthly_returns[monthly_returns < 0].mean() if len(monthly_returns[monthly_returns < 0]) > 0 else 0,
            'avg_rolling_beta': rolling_beta.mean(),
            'beta_std': rolling_beta.std(),
            'consecutive_wins': self._get_max_consecutive(returns > 0),
            'consecutive_losses': self._get_max_consecutive(returns < 0)
        }

    def _get_max_consecutive(self, series: pd.Series) -> int:
        """Helper function to calculate maximum consecutive True values."""
        return max((series * (series.groupby((series != series.shift()).cumsum()).cumcount() + 1)).max(), 0)
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run backtest for all strategies across their specified symbols.
        """
        # Fetch historical data for all symbols in the universe
        data = self.market_data.get_historical_data(
            symbols=self._get_all_symbols(),
            start_date=self.config['data']['parameters']['history_start'],
            end_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            interval=self.config['data']['parameters']['interval']
        )
        
        # Run strategies
        for strategy_name, strategy in self.strategies.items():
            strategy_config = self.config['strategies'][strategy_name]
            strategy_symbols = strategy_config.get('symbols', [])
            
            # Initialize strategy results
            self.results[strategy_name] = {
                'positions': pd.DataFrame(),
                'portfolio_value': pd.Series(dtype=float),
                'returns': pd.Series(dtype=float)
            }
            
            # Process each symbol for the strategy
            for symbol in strategy_symbols:
                if symbol not in data:
                    print(f"Warning: No data available for {symbol} in {strategy_name}")
                    continue
                    
                # Generate signals for this symbol
                signals = strategy.generate_signals(data[symbol])
                
                # Apply risk management if configured
                if 'risk_management' in self.config:
                    signals = self._apply_risk_management(signals, data[symbol], strategy_name)
                
                # Calculate symbol-specific returns
                position_changes = signals.diff()
                returns = pd.Series(0.0, index=data[symbol].index)
                
                for i in range(1, len(data[symbol])):
                    if signals[i-1] != 0:
                        returns[i] = signals[i-1] * (
                            data[symbol]['close'][i] / data[symbol]['close'][i-1] - 1
                        )
                
                # Store symbol-specific results
                self.results[strategy_name][symbol] = {
                    'signals': signals,
                    'returns': returns,
                    'cumulative_returns': (1 + returns).cumprod(),
                    'drawdown': self._calculate_drawdown_metrics(returns)
                }
            
            # Calculate portfolio-level metrics
            self.get_performance_metrics(strategy_name)
        
        return self.results
    
    def get_performance_metrics(self, benchmark_returns: pd.Series = None) -> Dict[str, Dict]:
        """
        Calculate and return comprehensive performance metrics for each strategy.
        
        Args:
            benchmark_returns: Optional benchmark returns series for comparison
            
        Returns:
            Dictionary containing detailed metrics for each strategy
        """
        metrics = {}
        
        for strategy_name, result in self.results.items():
            returns = result['returns']
            signals = result['signals']
            cum_returns = result['cumulative_returns']
            
            # Basic performance metrics
            basic_metrics = {
                'total_return': (cum_returns.iloc[-1] - 1) * 100,
                'annual_return': (cum_returns.iloc[-1] ** (252/len(returns)) - 1) * 100,
                'avg_daily_return': returns.mean() * 100,
                'return_std': returns.std() * 100
            }
            
            # Combine all metrics
            metrics[strategy_name] = {
                **basic_metrics,
                'drawdown_metrics': self._calculate_drawdown_metrics(returns),
                'trade_metrics': self._calculate_trade_metrics(signals, returns),
                'risk_metrics': self._calculate_risk_metrics(returns),
                'timing_metrics': self._calculate_timing_metrics(returns, benchmark_returns)
            }
        
        return metrics

    def _initialize_strategies(self) -> Dict[str, Strategy]:
        """Initialize strategy objects based on configuration."""
        strategy_map = {
            'MovingAverageCrossover': MovingAverageCrossover,
            'RSIStrategy': RSIStrategy,
            'BollingerBandsStrategy': BollingerBandsStrategy,
            'MACDStrategy': MACDStrategy,
            'IchimokuStrategy': IchimokuStrategy,
            'VolumeWeightedMAStrategy': VolumeWeightedMAStrategy,
            'AdaptiveMovingAverageStrategy': AdaptiveMovingAverageStrategy,
            'DualThrustStrategy': DualThrustStrategy
        }
        
        strategies = {}
        for strategy_name, strategy_config in self.config['strategies'].items():
            strategy_class = strategy_map[strategy_config['type']]
            strategies[strategy_name] = strategy_class(strategy_config['parameters'])
        
        return strategies