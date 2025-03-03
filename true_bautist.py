from components.TrueBautist import TrueBautistStrategy
from components.Indicators import TechnicalIndicators

from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting

from datetime import datetime
import yaml
import argparse
import pandas as pd
from typing import Dict

class YAMLStrategy(Strategy):
    """
    This class takes in a yaml configuration file as a strategy and runs it using Lumibot lifecycle methods. 
    """
    def initialize(self,
                   true_bautist_config: TrueBautistStrategy):
        
        self.strategy = true_bautist_config
        self.symbols = true_bautist_config.get_config()['symbols']
        self.timeframe = true_bautist_config.get_config()['timeframe']
        self.sleeptime = self.timeframe
        self.params = true_bautist_config.indicators
        self.entry_conditions = true_bautist_config.get_config()['entry_conditions']
        self.exit_conditions = true_bautist_config.get_config()['exit_conditions']
        self.risk_management = true_bautist_config.get_config()['risk_management']

    def before_market_opens(self):
        self.indicators = self.params
        print("\n")
        print(f"Date: {self.get_datetime()}",f"Cash Balance: {self.get_cash():.2f}", f"Account Value: {self.get_portfolio_value():.2f}") 

    def on_trading_iteration(self):
        cash = self.get_cash()
        positions = self.get_positions()
        print(f"Date: {self.get_datetime()}, Positions: {positions}" )
        if cash <= 0 :
            self.sell_all()
            self.sleep
            
        else:
            for symbol in self.symbols:
                
                # Get data for each symbol
                prices = self.get_historical_prices(symbol, 30)
                position = self.get_position(symbol)
                
                

                technicals = TechnicalIndicators(prices.df, self.params)
                df = technicals.calculate_indicators()

                if position:
                    # IF WE HAVE A POSITION, CHECK THE EXIT CONDITIONS
                    if self._check_exit_conditions(df.iloc[-1]):
                        # EVALUATES TO TRUE OR FALSE, IF TRUE SELL ALL
                        self.sell_all(symbol)
                        print(f"Selling {position.quantity} shares of {symbol}")
                
                else:
                    # CHECK IF ENTRY CONDITIONS EVALUATE TO TRUE; TAKE POSITION
                    if self._check_entry_conditions(df.iloc[-1]):
                        # Calculate the risk management
                            
                        price = df.close.iloc[-1]
                        risk_amount = cash * self.risk_management['risk_per_trade']

                        position_size = risk_amount // price 
                        stop_loss_price = price * (1 - self.risk_management['stop_loss'])
                        take_profit_price = price * (1 + self.risk_management['take_profit'])
                        # Execute the purchase
                        order = self.create_order(
                            asset=symbol,
                            quantity = position_size,
                            side="buy",
                            take_profit_price=take_profit_price,
                            stop_loss_price=stop_loss_price,
                            type="bracket"
                        )
                        print(f"Submitting Order: {symbol}, Position Size: {position_size} \n Total Cost (Approximate): {position_size * df.close.iloc[-1]}")
                        
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
    
    def _check_condition(self, row: pd.Series, condition_config: Dict):

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
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Trading Strategy Backtester')
    parser.add_argument('-c','--config', 
                        required=True,
                        help='Trading strategy YAML Config to backtest')
    parser.add_argument('-k','--api_keys',
                        required=True,
                        help='API Keys for Live Trading')

    args = parser.parse_args()


    if args.api_keys:
        api_keys = args.api_keys 

    with open(api_keys, 'r') as file:
        keys = yaml.safe_load(file)

    with open(args.config, 'r') as file:
        yaml_trade_config = yaml.safe_load(file)

    strategy = TrueBautistStrategy(yaml_trade_config, keys)
    print(strategy.get_config())
    is_live = False

    if is_live:
        from lumibot.credentials import ALPACA_CONFIG
        from lumibot.brokers import Alpaca

        broker = Alpaca(ALPACA_CONFIG)

        lumistrategy = YAMLStrategy(broker=broker, true_bautist_config=strategy)
        lumistrategy.run_live()

    else:

        backtesting_start = datetime.strptime(strategy.config['start_date'], "%Y-%m-%d")
        backtesting_end = datetime.strptime(strategy.config['end_date'], "%Y-%m-%d")
        YAMLStrategy.backtest(YahooDataBacktesting,
                            backtesting_start=backtesting_start,
                            backtesting_end=backtesting_end, 
                            parameters={'true_bautist_config': strategy})