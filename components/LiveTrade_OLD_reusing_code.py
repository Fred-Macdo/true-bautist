from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from .Portfolio_OLD_reusing_code import Portfolio
from .Scanner import MarketScanner
from .Strategy import TradingStrategy, Signal

class LiveTrader:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        strategy: TradingStrategy,
        scanner: MarketScanner,
        paper: bool = True
    ):
        """Initialize live trader"""
        self.portfolio = Portfolio(api_key, api_secret, paper)
        self.strategy = strategy
        self.scanner = scanner
        self.stream = StockDataStream(api_key, api_secret)
        self.active = False
        self.positions = {}
        self.pending_orders = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)

    async def start_trading(self, symbols: List[str]):
        """Start live trading"""
        self.active = True
        self.logger.info("Starting live trading...")

        # Initial market analysis
        market_condition = await self.scanner.analyze_macro_environment()
        self.logger.info(f"Market Condition: {market_condition}")

        # Set up data handlers
        async def handle_bar(bar):
            if not self.active:
                return

            try:
                # Update market data
                symbol = bar.symbol
                data = self._update_market_data(symbol, bar)
                
                # Get account info
                account_info = self.portfolio.get_account_details()
                
                # Generate signals
                signals = self.strategy.generate_signals(
                    {symbol: data},
                    account_info,
                    market_condition.__dict__ if market_condition else None
                )
                
                # Execute signals
                for signal in signals:
                    await self._execute_signal(signal)
                    
            except Exception as e:
                self.logger.error(f"Error processing bar data: {e}")

        # Subscribe to market data
        try:
            self.stream.subscribe_bars(handle_bar, *symbols)
            await self.stream.run()
        except Exception as e:
            self.logger.error(f"Error in market data stream: {e}")
            await self.stop_trading()

    async def stop_trading(self):
        """Stop live trading"""
        self.logger.info("Stopping live trading...")
        self.active = False
        await self.stream.stop()

    async def _execute_signal(self, signal: Signal):
        """Execute trading signal"""
        try:
            if not signal:
                return

            self.logger.info(f"Executing signal: {signal}")
            
            if signal.action == 'BUY':
                # Check if we already have a position
                position = self.portfolio.get_position(signal.symbol)
                if position is None:
                    # Calculate position size
                    account = self.portfolio.get_account_details()
                    position_value = (
                        signal.metadata['position_size'] * 
                        account['portfolio_value']
                    )
                    
                    # Place order
                    order = self.portfolio.place_market_order(
                        symbol=signal.symbol,
                        quantity=position_value,
                        side='buy'
                    )
                    
                    if order:
                        self.logger.info(f"Buy order placed: {order}")
                        self.pending_orders[order['id']] = order
                        
            elif signal.action == 'SELL':
                position = self.portfolio.get_position(signal.symbol)
                if position:
                    # Close position
                    order = self.portfolio.place_market_order(
                        symbol=signal.symbol,
                        quantity=position['quantity'],
                        side='sell'
                    )
                    
                    if order:
                        self.logger.info(f"Sell order placed: {order}")
                        self.pending_orders[order['id']] = order

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")

    def _update_market_data(self, symbol: str, bar) -> pd.DataFrame:
        """Update market data with new bar"""
        try:
            return pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }])
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return pd.DataFrame()

    async def monitor_performance(self):
        """Monitor trading performance"""
        while self.active:
            try:
                # Get current positions and account info
                positions = self.portfolio.get_all_positions()
                account = self.portfolio.get_account_details()
                
                # Log performance metrics
                self.logger.info("\nPerformance Update:")
                self.logger.info(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
                self.logger.info(f"Cash: ${account['cash']:,.2f}")
                
                for symbol, pos in positions.items():
                    self.logger.info(
                        f"{symbol}: {pos['quantity']} shares, "
                        f"P&L: ${pos['profit_loss']:,.2f} "
                        f"({pos['profit_loss_pct']:.2f}%)"
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(60)
