from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

class Portfolio:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """Initialize portfolio with Alpaca V2 API"""
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.positions: Dict = {}
        self.trades_history: List[Dict] = []
        self.portfolio_value_history: List[Dict] = []
        self._update_positions()

    def _update_positions(self):
        """Update positions from Alpaca"""
        try:
            positions = self.trading_client.get_all_positions()
            self.positions = {
                p.symbol: {
                    'quantity': float(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'market_value': float(p.market_value),
                    'profit_loss': float(p.unrealized_pl),
                    'profit_loss_pct': float(p.unrealized_plpc)
                }
                for p in positions
            }
        except Exception as e:
            print(f"Error updating positions: {e}")

    def get_account_details(self) -> Dict:
        """Get current account details"""
        account = self.trading_client.get_account()
        return {
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'position_market_value': float(account.position_market_value),
            'multiplier': float(account.multiplier)
        }

    def place_market_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Dict:
        """Place a market order"""
        try:
            order_details = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
                time_in_force=time_in_force
            )
            
            order = self.trading_client.submit_order(order_details)
            
            self._record_trade({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': 'market',
                'order_id': order.id,
                'status': order.status
            })
            
            self._update_positions()
            return {
                'id': order.id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'side': order.side.value,
                'status': order.status
            }
            
        except Exception as e:
            print(f"Error placing market order: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Dict:
        """Place a limit order"""
        try:
            order_details = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            
            order = self.trading_client.submit_order(order_details)
            
            self._record_trade({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': 'limit',
                'limit_price': limit_price,
                'order_id': order.id,
                'status': order.status
            })
            
            self._update_positions()
            return {
                'id': order.id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'side': order.side.value,
                'status': order.status,
                'limit_price': float(order.limit_price)
            }
            
        except Exception as e:
            print(f"Error placing limit order: {e}")
            return None

    def _record_trade(self, trade_details: Dict):
        """Record trade in history"""
        self.trades_history.append(trade_details)
        
        # Record portfolio value
        account = self.get_account_details()
        self.portfolio_value_history.append({
            'timestamp': trade_details['timestamp'],
            'portfolio_value': account['portfolio_value'],
            'cash': account['cash']
        })

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position details for a symbol"""
        self._update_positions()
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict:
        """Get all current positions"""
        self._update_positions()
        return self.positions

    def get_trading_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get trading history, optionally filtered by symbol"""
        if symbol:
            return [trade for trade in self.trades_history if trade['symbol'] == symbol]
        return self.trades_history

    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio value history as a DataFrame"""
        return pd.DataFrame(self.portfolio_value_history)
