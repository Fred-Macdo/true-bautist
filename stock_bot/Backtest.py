from .Strategy import Strategy  # Note we only import the Strategy class now
import yaml
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from .DataManager import *
import warnings
warnings.filterwarnings("ignore")

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
        self.market_data = MarketData(self.config.get('data', {}))
        self.strategies = self._initialize_strategies()
        self.results = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load and parse the YAML configuration file."""
        try:
            print('Loading Config \n')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
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
        print('Initializing Strategies \n')
        strategies = {}
        for strategy_name, strategy_config in self.config['strategies'].items():
            print(strategy_name, "config: ", strategy_config)
            if strategy_config.get('enabled', True):
                strategies[strategy_name] = Strategy(strategy_config['parameters'])
        return strategies

    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate detailed drawdown metrics.
        
        Returns a dictionary containing:
        - max_drawdown: Maximum peak-to-trough decline (percentage)
        - avg_drawdown: Average drawdown when in drawdown (percentage)
        - max_drawdown_duration: Longest drawdown period (days)
        - avg_drawdown_duration: Average length of drawdown periods (days)
        - drawdown_frequency: Percentage of time spent in drawdown (percentage)
        """
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        
        # Calculate drawdown duration
        is_drawdown = drawdowns < 0
        drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
        drawdown_duration = is_drawdown.groupby(drawdown_start).cumsum()
        
        return {
            'max_drawdown': round(drawdowns.min() * 100, 2),  # Convert to percentage
            'avg_drawdown': round(drawdowns[drawdowns < 0].mean() * 100, 2),  # Convert to percentage
            'max_drawdown_duration': round(drawdown_duration.max(), 2),
            'avg_drawdown_duration': round(drawdown_duration[drawdown_duration > 0].mean(), 2),
            'drawdown_frequency': round((drawdowns < 0).mean() * 100, 2)  # Convert to percentage
        }
    
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

    def _calculate_trade_metrics(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate trade-specific metrics.
        
        Returns a dictionary containing:
        - total_trades: Total number of trades (count)
        - win_rate: Percentage of profitable trades (percentage)
        - avg_trade_return: Average return per trade (percentage)
        - best_trade: Highest return from a single trade (percentage)
        - worst_trade: Lowest return from a single trade (percentage)
        - avg_winning_trade: Average return of profitable trades (percentage)
        - avg_losing_trade: Average return of unprofitable trades (percentage)
        - profit_factor: Ratio of gross profits over gross losses (ratio)
        """
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
            'total_trades': round(len(trades), 2),
            'win_rate': round((trade_returns > 0).mean() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'avg_trade_return': round(trade_returns.mean() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'best_trade': round(trade_returns.max() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'worst_trade': round(trade_returns.min() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'avg_winning_trade': round(trade_returns[trade_returns > 0].mean() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'avg_losing_trade': round(trade_returns[trade_returns < 0].mean() * 100, 2) if len(trade_returns) > 0 else 0,  # Convert to percentage
            'profit_factor': round(abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()), 2) if len(trade_returns) > 0 and trade_returns[trade_returns < 0].sum() != 0 else 0
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        
        Returns a dictionary containing:
        - sharpe_ratio: Risk-adjusted return using volatility (ratio)
        - sortino_ratio: Risk-adjusted return using downside volatility (ratio)
        - calmar_ratio: Risk-adjusted return using maximum drawdown (ratio)
        - volatility: Annualized standard deviation of returns (percentage)
        - skewness: Asymmetry of returns distribution (ratio)
        - kurtosis: Tail thickness of returns distribution (ratio)
        - var_95: Value at Risk at 95% confidence level (percentage)
        - cvar_95: Conditional Value at Risk at 95% confidence level (percentage)
        - omega_ratio: Probability weighted ratio of gains over losses (ratio)
        """
        risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        excess_returns = returns - daily_rf
        negative_returns = returns[returns < 0]
        
        return {
            'sharpe_ratio': round(np.sqrt(252) * (returns.mean() - daily_rf) / returns.std() if returns.std() != 0 else 0, 2),
            'sortino_ratio': round(np.sqrt(252) * (returns.mean() - daily_rf) / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() != 0 else 0, 2),
            'calmar_ratio': round((returns.mean() * 252) / abs(self._calculate_drawdown_metrics(returns)['max_drawdown']/100) if self._calculate_drawdown_metrics(returns)['max_drawdown'] != 0 else 0, 2),
            'volatility': round(returns.std() * np.sqrt(252) * 100, 2),  # Convert to percentage
            'skewness': round(stats.skew(returns), 2),
            'kurtosis': round(stats.kurtosis(returns), 2),
            'var_95': round(returns.quantile(0.05) * 100, 2),  # Convert to percentage
            'cvar_95': round(returns[returns <= returns.quantile(0.05)].mean() * 100, 2),  # Convert to percentage
            'omega_ratio': round(returns[returns > 0].mean() / abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 and returns[returns < 0].mean() != 0 else 0, 2)
        }

    def _calculate_timing_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate market timing and consistency metrics.
        
        Returns a dictionary containing:
        - avg_monthly_return: Average return per month (percentage)
        - monthly_return_std: Standard deviation of monthly returns (percentage)
        - positive_months: Percentage of months with positive returns (percentage)
        - avg_up_month: Average return during positive months (percentage)
        - avg_down_month: Average return during negative months (percentage)
        - avg_rolling_beta: Average sensitivity to market movements (ratio)
        - beta_std: Stability of market sensitivity (ratio)
        - consecutive_wins: Longest streak of profitable days (count)
        - consecutive_losses: Longest streak of unprofitable days (count)
        """
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=returns.index)  # Use zero returns as benchmark if none provided
            
        rolling_beta = returns.rolling(window=63).cov(benchmark_returns) / benchmark_returns.rolling(window=63).var()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'avg_monthly_return': round(monthly_returns.mean() * 100, 2),  # Convert to percentage
            'monthly_return_std': round(monthly_returns.std() * 100, 2),  # Convert to percentage
            'positive_months': round((monthly_returns > 0).mean() * 100, 2),  # Convert to percentage
            'avg_up_month': round(monthly_returns[monthly_returns > 0].mean() * 100, 2) if len(monthly_returns[monthly_returns > 0]) > 0 else 0,  # Convert to percentage
            'avg_down_month': round(monthly_returns[monthly_returns < 0].mean() * 100, 2) if len(monthly_returns[monthly_returns < 0]) > 0 else 0,  # Convert to percentage
            'avg_rolling_beta': round(rolling_beta.mean(), 2),
            'beta_std': round(rolling_beta.std(), 2),
            'consecutive_wins': round(self._get_max_consecutive(returns > 0), 2),
            'consecutive_losses': round(self._get_max_consecutive(returns < 0), 2)
        }

    def _get_max_consecutive(self, series: pd.Series) -> int:
        """Helper function to calculate maximum consecutive True values."""
        return max((series * (series.groupby((series != series.shift()).cumsum()).cumcount() + 1)).max(), 0)
    
    def run(self, benchmark_returns: pd.Series = None) -> Dict[str, pd.DataFrame]:
        """
        Run the backtest for all configured strategies.
        
        Args:
            data: DataFrame with OHLCV data
            benchmark_returns: Optional benchmark returns series for comparison
            
        Returns:
            Dictionary containing results for each strategy
        """

        data = self.market_data.get_historical_data(
            symbols=self._get_all_symbols(),
            start_date=self.config['data']['parameters']['history_start'],
            end_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            interval=self.config['data']['parameters']['interval']
        )

        for strategy_name, strategy in self.strategies.items():
            for symbol, market_data in data.items():
                
                signals = strategy.generate_signals(market_data)
                
                # Calculate returns
                position_changes = signals.diff()
                returns = pd.Series(0.0, index=market_data.index)
                
                for i in range(1, len(market_data)):
                    if signals[i-1] != 0:  # If we have a position
                        returns[i] = signals[i-1] * (
                            market_data['close'][i] / market_data['close'][i-1] - 1
                        )
                
                # Calculate metrics
                cumulative_returns = (1 + returns).cumprod()
                drawdown = cumulative_returns / cumulative_returns.cummax() - 1
                
                self.results[strategy_name] = pd.DataFrame({
                    'signals': signals,
                    'returns': returns,
                    'cumulative_returns': cumulative_returns,
                    'drawdown': drawdown
                })
            
        return self.results
    
    def get_performance_metrics(self, benchmark_returns: pd.Series = None) -> Dict[str, Dict]:
        """
        Calculate and return comprehensive performance metrics for each strategy.
        
        Args:
            benchmark_returns: Optional benchmark returns series for comparison
            
        Returns:
            Dictionary containing metrics for each strategy:
            - Basic metrics:
                - total_return: Total return over the period (percentage)
                - annual_return: Annualized return (percentage)
                - avg_daily_return: Average daily return (percentage)
                - return_std: Standard deviation of daily returns (percentage)
            - Drawdown metrics (see _calculate_drawdown_metrics)
            - Trade metrics (see _calculate_trade_metrics)
            - Risk metrics (see _calculate_risk_metrics)
            - Timing metrics (see _calculate_timing_metrics)
        """
        metrics = {}
        
        for strategy_name, result in self.results.items():
            returns = result['returns']
            signals = result['signals']
            cum_returns = result['cumulative_returns']
            
            # Basic performance metrics
            basic_metrics = {
                'total_return': round((cum_returns.iloc[-1] - 1) * 100, 2),  # Convert to percentage
                'annual_return': round((cum_returns.iloc[-1] ** (252/len(returns)) - 1) * 100, 2),  # Convert to percentage
                'avg_daily_return': round(returns.mean() * 100, 2),  # Convert to percentage
                'return_std': round(returns.std() * 100, 2)  # Convert to percentage
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