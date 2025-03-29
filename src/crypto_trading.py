from components.TrueBautist import TrueBautistStrategy
from components.Indicators import TechnicalIndicators

from lumibot.strategies import Strategy
from lumibot.entities import Asset
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca

from datetime import datetime
import yaml
import argparse
import pandas as pd
from typing import Dict, Union, List, Any
import numpy as np

# Set consistent formatting options at the beginning of the script
pd.set_option('display.precision', 2)
np.set_printoptions(precision=2, suppress=True)  # Added suppress=True to avoid scientific notation


class YAMLStrategy(Strategy):
    """
    This class takes in a yaml configuration file as a strategy and runs it using Lumibot lifecycle methods. 
    """
    def initialize(self,
                   true_bautist_config: TrueBautistStrategy):
        
        self.strategy = true_bautist_config
        self.symbols = true_bautist_config.get_config()['symbols']
        self.timeframe = true_bautist_config.get_config()['timeframe']
        self.sleeptime = '10S'
        self.params = true_bautist_config.indicators
        self.entry_conditions = true_bautist_config.get_config()['entry_conditions']
        self.exit_conditions = true_bautist_config.get_config()['exit_conditions']
        self.risk_management = true_bautist_config.get_config()['risk_management']
        self.set_market("24/7")

        self.indicators = self.params
 

    def on_trading_iteration(self):

        cash = self.get_cash()
        positions = self.get_positions()
        print(f"Date: {self.get_datetime()}", f"Cash Balance: {self.get_cash():.2f}", f"Account Value: {self.get_portfolio_value():.2f}")
        print(f"Current Positions: {positions}")
        if cash <= 0 :
            self.sell_all()
            self.sleep
            
        else:
            asset = Asset("DOGE", asset_type=Asset.AssetType.CRYPTO)
            self.get_last_price(asset)
            
            # Get data for each symbol
            prices = self.get_historical_prices(asset, 30)
            position = self.get_position(asset)

            technicals = TechnicalIndicators(prices.df, self.params)
            df = technicals.calculate_indicators()
            
            if position:
                # IF WE HAVE A POSITION, CHECK THE EXIT CONDITIONS
                if self._check_exit_conditions(df.iloc[-1]):
                    # EVALUATES TO TRUE OR FALSE, IF TRUE SELL ALL
                    self.sell_all(asset)
                    print(f"Selling {position.quantity} shares of {asset}")
    
            else:
                # CHECK IF ENTRY CONDITIONS EVALUATE TO TRUE; TAKE POSITION
                if self._check_entry_conditions(df.iloc[-1]):
                    # Calculate the risk management
                        
                    price = df.close.iloc[-1]
                    risk_amount = cash * self.risk_management['risk_per_trade']

                    position_size = risk_amount // price 
                    stop_loss_price = price * (1 - self.risk_management['stop_loss'])
                    take_profit_price = price * (1 + self.risk_management['take_profit'])
                    self.stop_loss_price = {asset: stop_loss_price}
                    self.take_profit_price = {asset: take_profit_price}
                    # Execute the purchase
                    order = self.create_order(
                        asset=asset,
                        quantity = position_size,
                        side="buy",
                        type="market"
                    )
                    
                    print("Entry Conditions Met:", self.entry_conditions)
                    print("Latest Bar: ", df.iloc[-1].round(4))
                    print(f"Submitting Order: {asset}, Position Size: {position_size:.0f}")
                    print(f"Total Cost (Approximate): ${(position_size * df.close.iloc[-1]):.2f}")
                    
                    self.submit_order(order)

        print("*****************************************")
    def on_abrupt_closing(self):
        # Sell all positions
        self.sell_all()

    ######################################
    ########## HELPER FUNCTIONS ##########
    ######################################

    def _check_entry_conditions(self, row: pd.Series) -> bool:
        """Check if all entry conditions are met"""
        return all(
            self._check_condition(row, condition)
            for condition in self.entry_conditions
        )

    def _check_exit_conditions(self, row: pd.Series) -> bool:
        """Check if any exit condition is met"""
        return any(
            self._check_condition(row, condition)
            for condition in self.exit_conditions
        )


    def _check_condition(self, row: pd.Series, condition_config: Dict) -> bool:
        """
        Check entry/exit condition for a given row based on indicator values and comparison logic.
        
        Args:
            row: pandas Series containing indicator values and their previous values
            condition_config: Dictionary with keys 'indicator', 'comparison', and 'value'
        
        Returns:
            bool: Whether the condition is met
        
        Raises:
            ValueError: If comparison operator is invalid
        """
        valid_comparisons = ['above', 'below', 'between', 'crosses_above', 'crosses_below']
        comparison = condition_config['comparison']
        
        if comparison not in valid_comparisons:
            raise ValueError(f"Comparison '{comparison}' is not valid. Must be one of {valid_comparisons}")
        
        indicator = condition_config['indicator']
        value = condition_config['value']
        
        # Ensure indicator is lowercase for consistent access
        indicator_key = indicator.lower()
        
        # Handle special indicators with dedicated comparisons
        if indicator == "MACD" and comparison in ["crosses_above", "crosses_below"]:
            return self._check_macd_cross(row, comparison)
        
        elif indicator == "BBANDS" and comparison in ["crosses_above", "crosses_below"]:
            return self._check_bbands_cross(row, comparison, value)
        
        # Handle general comparison cases
        if comparison == "above":
            return self._check_above(row, indicator_key, value)
        
        elif comparison == "below":
            return self._check_below(row, indicator_key, value)
        
        elif comparison == "crosses_above":
            return self._check_crosses_above(row, indicator_key, value)
        
        elif comparison == "crosses_below":
            return self._check_crosses_below(row, indicator_key, value)
        
        elif comparison == "between":
            # The original code had a bug here - it used 'indicator == between'
            # The correct check should be looking at the indicator value being between bounds
            return self._check_between(row, indicator_key, value)
        
        # Default fallback (should never reach here due to validation)
        return False


    def _check_above(self, row: pd.Series, indicator_key: str, value: Union[str, int, float]) -> bool:
        """Check if indicator is above a value or another indicator"""
        if isinstance(value, str):
            return row[indicator_key] > row[value.lower()]
        else:  # int, float
            return row[indicator_key] > value


    def _check_below(self, row: pd.Series, indicator_key: str, value: Union[str, int, float]) -> bool:
        """Check if indicator is below a value or another indicator"""
        if isinstance(value, str):
            return row[indicator_key] < row[value.lower()]
        else:  # int, float
            return row[indicator_key] < value


    def _check_crosses_above(self, row: pd.Series, indicator_key: str, value: Union[str, int, float]) -> bool:
        """Check if indicator crosses above a value or another indicator"""
        if isinstance(value, str):
            value_key = value.lower()
            evaluation = (row[indicator_key] > row[value_key]) and (row[f"{indicator_key}_prev"] <= row[f"{value_key}_prev"])
            if evaluation:
                print("Crosses above evaluation")
                print(f"Current Values: {indicator_key}: {row[indicator_key]:.2f}, {value_key}: {row[value_key]:.2f}")
                print(f"Previous values: {indicator_key}_prev: {row[f'{indicator_key}_prev']:.2f}, {value_key}_prev: {row[f'{value_key}_prev']:.2f}")
            return evaluation
        else:  # int, float
            evaluation = (row[indicator_key] > value) and (row[f"{indicator_key}_prev"] <= value)
            if evaluation:
                print("Crosses above evaluation, numerical value")
                print(f"Current Values: {indicator_key}: {row[indicator_key]:.2f}")
                print(f"Previous Values: {indicator_key}_prev: {row[f'{indicator_key}_prev']:.2f}, value: {value:.2f}")
            return evaluation


    def _check_crosses_below(self, row: pd.Series, indicator_key: str, value: Union[str, int, float]) -> bool:
        """Check if indicator crosses below a value or another indicator"""
        if isinstance(value, str):
            value_key = value.lower()
            evaluation = (row[indicator_key] < row[value_key]) and (row[f"{indicator_key}_prev"] >= row[f"{value_key}_prev"])
            if evaluation:
                print("Crosses below evaluation")
                print(f"Current Values: {indicator_key}: {row[indicator_key]:.2f}, {value_key}: {row[value_key]:.2f}")
                print(f"Previous values: {indicator_key}_prev: {row[f'{indicator_key}_prev']:.2f}, {value_key}_prev: {row[f'{value_key}_prev']:.2f}")
            return evaluation
        else:  # int, float
            evaluation = (row[indicator_key] < value) and (row[f"{indicator_key}_prev"] >= value)
            if evaluation:
                print("Crosses below evaluation, numerical value")
                print(f"Current Values: {indicator_key}: {row[indicator_key]:.2f}")
                print(f"Previous Values: {indicator_key}_prev: {row[f'{indicator_key}_prev']:.2f}, value: {value:.2f}")
            return evaluation


    def _check_between(self, row: pd.Series, indicator_key: str, value: List[Union[str, int, float]]) -> bool:
        """
        Check if indicator value is between two bounds
        
        In pandas, .between() is a method on a Series, not on a single value.
        For a single value from a Series (like row[indicator_key]), we need to 
        do a direct comparison or convert it to a Series first.
        """
        try:
            # Try using scalar between method if it's available on this pandas version
            if all(isinstance(x, (int, float)) for x in value):
                return row[indicator_key].between(value[0], value[1])
            else:
                lower_value = row[value[0].lower()] if isinstance(value[0], str) else value[0]
                upper_value = row[value[1].lower()] if isinstance(value[1], str) else value[1]
                return row[indicator_key].between(lower_value, upper_value)
        except AttributeError:
            # Fallback for older pandas versions or if the value doesn't support between
            if all(isinstance(x, (int, float)) for x in value):
                return (value[0] <= row[indicator_key]) and (row[indicator_key] <= value[1])
            else:
                lower_value = row[value[0].lower()] if isinstance(value[0], str) else value[0]
                upper_value = row[value[1].lower()] if isinstance(value[1], str) else value[1]
                return (lower_value <= row[indicator_key]) and (row[indicator_key] <= upper_value)


    def _check_macd_cross(self, row: pd.Series, comparison: str) -> bool:
        """Handle MACD specific crossing logic"""
        if comparison == "crosses_above":
            evaluation = (row['macd'] > row['macd_signal']) and (row['macd_prev'] <= row['macdsignal_prev'])
            if evaluation:
                print("MACD crosses above signal")
                print(f"MACD: {row['macd']:.2f}, Signal: {row['macd_signal']:.2f}")
                print(f"MACD_prev: {row['macd_prev']:.2f}, Signal_prev: {row['macdsignal_prev']:.2f}")
            return evaluation
        else:  # crosses_below
            evaluation = (row['macd'] < row['macd_signal']) and (row['macd_prev'] >= row['macdsignal_prev'])
            if evaluation:
                print("MACD crosses below signal")
                print(f"MACD: {row['macd']:.2f}, Signal: {row['macd_signal']:.2f}")
                print(f"MACD_prev: {row['macd_prev']:.2f}, Signal_prev: {row['macdsignal_prev']:.2f}")
            return evaluation


    def _check_bbands_cross(self, row: pd.Series, comparison: str, value: str) -> bool:
        """Handle Bollinger Bands specific crossing logic"""
        value_key = value.lower() if isinstance(value, str) else value
        
        if comparison == "crosses_above":
            evaluation = (row['close'] > row[value_key]) and (row['close_prev'] <= row[value_key])
            if evaluation:
                print("Price crosses above Bollinger Band")
                print(f"Close: {row['close']:.2f}, {value_key}: {row[value_key]:.2f}")
                print(f"Close_prev: {row['close_prev']:.2f}, {value_key}_prev: {row[value_key]:.2f}")
            return evaluation
        else:  # crosses_below
            evaluation = (row['close'] < row[value_key]) and (row['close_prev'] >= row[value_key])
            if evaluation:
                print("Price crosses below Bollinger Band")
                print(f"Close: {row['close']:.2f}, {value_key}: {row[value_key]:.2f}")
                print(f"Close_prev: {row['close_prev']:.2f}, {value_key}_prev: {row[value_key]:.2f}")
            return evaluation
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Trading Strategy Backtester')
    parser.add_argument('-c','--config', 
                        required=True,
                        help='Trading strategy YAML Config to backtest')
    parser.add_argument('-k','--api_keys',
                        required=True,
                        help='API Keys for Live Trading')
    parser.add_argument("mode", choices=["live", "paper", "backtest"],
                        help="Run mode: 'live' for live trading on real money account, 'paper' for live trading on paper account, 'backtest' for historical testing")
    
    
    args = parser.parse_args()


    if args.api_keys:
        api_keys = args.api_keys 

    with open(api_keys, 'r') as file:
        keys = yaml.safe_load(file)

    with open(args.config, 'r') as file:
        yaml_trade_config = yaml.safe_load(file)

    strategy = TrueBautistStrategy(yaml_trade_config, keys)
    print(strategy.get_config())
    
    ALPACA_CONFIG = {
    # Put your own Alpaca key here:
    "API_KEY": keys["API_KEY"],
    # Put your own Alpaca secret here:
    "API_SECRET": keys["API_SECRET"],
    # If you want to go live, you must change this
    "PAPER": True,
    }  
    
    if args.mode == 'live':
        from lumibot.brokers import Alpaca
        # LIVE TRADE; ENSURE LIVE API KEYS IN COMMAND LINE ARGS
        ALPACA_CONFIG = ALPACA_CONFIG['PAPER'] = False
        broker = Alpaca(ALPACA_CONFIG)

        lumistrategy = YAMLStrategy(broker=broker, true_bautist_config=strategy)
        lumistrategy.run_live()

    elif args.mode == 'paper':
        # PAPER TRADE; ENSURE PAPER API KEYS IN COMMAND LINE ARGS
        broker = Alpaca(ALPACA_CONFIG)

        lumistrategy = YAMLStrategy(broker=broker, true_bautist_config=strategy)
        lumistrategy.run_live()
    else:
        # BACKTEST ONLY
        backtesting_start = datetime.strptime(strategy.config['start_date'], "%Y-%m-%d")
        backtesting_end = datetime.strptime(strategy.config['end_date'], "%Y-%m-%d")
        YAMLStrategy.backtest(YahooDataBacktesting,
                            backtesting_start=backtesting_start,
                            backtesting_end=backtesting_end, 
                            parameters={'true_bautist_config': strategy})