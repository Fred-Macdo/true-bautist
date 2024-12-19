import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import schedule
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import OptionOrderRequest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AutomatedAlpacaOptionsTrading:
    def __init__(self, max_position_size=5000, max_loss_percentage=0.02):
        # Initialize API clients
        self.trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_API_SECRET'),
            paper=True  # Set to False for live trading
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_API_SECRET')
        )
        
        self.max_position_size = max_position_size
        self.max_loss_percentage = max_loss_percentage
        self.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    def get_account(self):
        """Get account information"""
        return self.trading_client.get_account()

    def get_positions(self):
        """Get current positions"""
        return self.trading_client.get_all_positions()

    def get_option_chain(self, symbol):
        """Get option chain for a symbol"""
        # Using the new options endpoint
        endpoint = f"https://data.alpaca.markets/v2/stocks/{symbol}/options"
        headers = {
            "APCA-API-KEY-ID": os.getenv('ALPACA_API_KEY'),
            "APCA-API-SECRET-KEY": os.getenv('ALPACA_API_SECRET')
        }
        response = requests.get(endpoint, headers=headers)
        return response.json()

    def submit_option_order(self, symbol, option_type, strike_price, expiration_date, quantity, side, order_type='market'):
        """Submit an option order with the new API"""
        if not self.check_risk_management(symbol, quantity, side):
            print("Order rejected due to risk management rules")
            return None

        order_details = OptionOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            style='american',  # or 'european' depending on the option
            strike=strike_price,
            right='call' if option_type.lower() == 'call' else 'put',
            expiry=expiration_date
        )

        try:
            order = self.trading_client.submit_order(order_details)
            print(f"Order submitted: {order}")
            return order
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None

    def check_risk_management(self, symbol, quantity, side):
        """Enhanced risk management checks"""
        account = self.get_account()
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)

        # Calculate position value
        option_price = self.get_option_price(symbol)  # Implement this method based on your data source
        position_value = option_price * quantity * 100  # 100 shares per contract

        # Check position size limits
        if position_value > self.max_position_size:
            print(f"Position size ${position_value} exceeds maximum allowed ${self.max_position_size}")
            return False

        # Check buying power
        if side.lower() == 'buy' and position_value > buying_power:
            print(f"Insufficient buying power: ${buying_power} available, ${position_value} required")
            return False

        # Check maximum loss potential
        max_loss = min(position_value, portfolio_value * self.max_loss_percentage)
        if position_value > max_loss:
            print(f"Position value ${position_value} exceeds maximum allowed loss ${max_loss}")
            return False

        return True

    def implement_strategy(self, symbol, strategy_type, **kwargs):
        """Implement various option strategies"""
        if strategy_type == 'bull_call_spread':
            return self.bull_call_spread(symbol, **kwargs)
        elif strategy_type == 'iron_condor':
            return self.iron_condor(symbol, **kwargs)
        # Add other strategies as needed

    def bull_call_spread(self, symbol, width=5):
        """Implement a bull call spread strategy"""
        chain = self.get_option_chain(symbol)
        current_price = self.get_current_price(symbol)
        
        # Find appropriate strike prices
        lower_strike = self._find_nearest_strike(chain, current_price)
        higher_strike = lower_strike + width

        expiration = self._get_next_monthly_expiration()

        # Submit the orders
        buy_order = self.submit_option_order(
            symbol=symbol,
            option_type='call',
            strike_price=lower_strike,
            expiration_date=expiration,
            quantity=1,
            side='buy'
        )

        if buy_order:
            sell_order = self.submit_option_order(
                symbol=symbol,
                option_type='call',
                strike_price=higher_strike,
                expiration_date=expiration,
                quantity=1,
                side='sell'
            )
            return buy_order, sell_order
        return None, None

    def get_current_price(self, symbol):
        """Get current price of a symbol"""
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(minutes=5)
        )
        bars = self.data_client.get_stock_bars(bars_request)
        return bars[symbol][-1].close

    def _find_nearest_strike(self, chain, price):
        """Find the nearest strike price to the current price"""
        strikes = [float(option['strike_price']) for option in chain['options']]
        return min(strikes, key=lambda x: abs(x - price))

    def _get_next_monthly_expiration(self):
        """Get the next monthly expiration date"""
        today = datetime.now()
        next_month = today.replace(day=1) + timedelta(days=32)
        third_friday = next_month.replace(day=1)
        while third_friday.weekday() != 4:
            third_friday += timedelta(days=1)
        third_friday += timedelta(days=14)
        return third_friday.strftime('%Y-%m-%d')

    def run_automation(self):
        """Run automated trading"""
        schedule.every().day.at("09:30").do(self.run_daily_analysis)
        schedule.every(15).minutes.do(self.monitor_positions)

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                print(f"Error in automation: {e}")
                # Implement error notification system here

    def run_daily_analysis(self):
        """Run daily analysis for all watchlist symbols"""
        for symbol in self.watchlist:
            try:
                strategy = self.analyze_market_conditions(symbol)
                if strategy:
                    self.implement_strategy(symbol, strategy)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

    def analyze_market_conditions(self, symbol):
        """Analyze market conditions to determine strategy"""
        # Get historical data using the new API
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=50)
        )
        bars = self.data_client.get_stock_bars(bars_request)
        
        # Convert to DataFrame
        df = pd.DataFrame([bar for bar in bars[symbol]])
        
        # Calculate indicators
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        
        current_price = df['close'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        
        # Strategy selection logic
        if current_price > sma20 > sma50:
            return 'bull_call_spread'
        elif sma50 > sma20 > current_price:
            return 'iron_condor'
        return None

if __name__ == "__main__":
    trader = AutomatedAlpacaOptionsTrading(max_position_size=5000, max_loss_percentage=0.02)
    
    # Example usage
    #trader.run_automation()
    
    # Or test a single strategy
    trader.implement_strategy('AAPL', 'bull_call_spread')