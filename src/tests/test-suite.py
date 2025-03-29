import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json

# Import your modules
# You may need to adjust these imports based on your project structure
from AlpacaDataManager import AlpacaDataFetcher
from Indicators import TechnicalIndicators
from TrueBautist import TrueBautistStrategy
from Scanner import MarketScanner, MarketCondition

class TestAlpacaDataFetcher(unittest.TestCase):
    """Test suite for the AlpacaDataFetcher class"""

    def setUp(self):
        """Set up test environment before each test method"""
        self.api_key = "test_api_key"
        self.secret_key = "test_secret_key"
        
        # Create a mock for StockHistoricalDataClient, CryptoHistoricalDataClient, and TradingClient
        self.mock_hist_client = MagicMock()
        self.mock_crypto_hist_client = MagicMock()
        self.mock_trading_client = MagicMock()
        
        # Apply patches for external dependencies
        self.hist_client_patcher = patch('AlpacaDataManager.StockHistoricalDataClient')
        self.crypto_hist_client_patcher = patch('AlpacaDataManager.CryptoHistoricalDataClient')
        self.trading_client_patcher = patch('AlpacaDataManager.TradingClient')
        self.requests_patcher = patch('AlpacaDataManager.requests')
        
        # Start the patchers and set the return values
        self.mock_hist_client_class = self.hist_client_patcher.start()
        self.mock_crypto_hist_client_class = self.crypto_hist_client_patcher.start()
        self.mock_trading_client_class = self.trading_client_patcher.start()
        self.mock_requests = self.requests_patcher.start()
        
        self.mock_hist_client_class.return_value = self.mock_hist_client
        self.mock_crypto_hist_client_class.return_value = self.mock_crypto_hist_client
        self.mock_trading_client_class.return_value = self.mock_trading_client
        
        # Create the AlpacaDataFetcher instance
        self.data_fetcher = AlpacaDataFetcher(self.api_key, self.secret_key)
    
    def tearDown(self):
        """Clean up after each test method"""
        # Stop all patchers
        self.hist_client_patcher.stop()
        self.crypto_hist_client_patcher.stop()
        self.trading_client_patcher.stop()
        self.requests_patcher.stop()
    
    def test_init_method(self):
        """Test if initialization sets up the clients correctly"""
        self.mock_hist_client_class.assert_called_once_with(api_key=self.api_key, secret_key=self.secret_key)
        self.mock_crypto_hist_client_class.assert_called_once_with(api_key=self.api_key, secret_key=self.secret_key)
        self.mock_trading_client_class.assert_called_once_with(api_key=self.api_key, secret_key=self.secret_key, paper=True)
        
        self.assertEqual(self.data_fetcher.headers["APCA-API-KEY-ID"], self.api_key)
        self.assertEqual(self.data_fetcher.headers["APCA-API-SECRET-KEY"], self.secret_key)
    
    def test_format_crypto_symbol(self):
        """Test crypto symbol formatting"""
        # Test USDT suffix
        self.assertEqual(self.data_fetcher._format_crypto_symbol("BTCUSDT"), "BTC/USD")
        
        # Test USD suffix
        self.assertEqual(self.data_fetcher._format_crypto_symbol("BTCUSD"), "BTC/USD")
        
        # Test already formatted
        self.assertEqual(self.data_fetcher._format_crypto_symbol("BTC/USD"), "BTC/USD")
        
        # Test other format
        self.assertEqual(self.data_fetcher._format_crypto_symbol("ETH"), "ETH")
    
    def test_process_raw_bars_to_df(self):
        """Test processing raw bars data to DataFrame"""
        # Create sample bars data
        raw_bars = [
            {'t': '2023-01-01T00:00:00Z', 'o': 100.0, 'h': 105.0, 'l': 99.0, 'c': 102.0, 'v': 1000, 'n': 50, 'vw': 101.5},
            {'t': '2023-01-02T00:00:00Z', 'o': 102.0, 'h': 107.0, 'l': 101.0, 'c': 106.0, 'v': 1200, 'n': 60, 'vw': 103.5}
        ]
        
        # Process the data
        result_df = self.data_fetcher._process_raw_bars_to_df(raw_bars)
        
        # Verify the result
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df.columns.tolist(), ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap'])
        self.assertEqual(result_df['close'].tolist(), [102.0, 106.0])
    
    @patch('AlpacaDataManager.requests.get')
    def test_get_historical_data(self, mock_get):
        """Test getting historical data for stocks"""
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': '2023-01-01T00:00:00Z', 'o': 100.0, 'h': 105.0, 'l': 99.0, 'c': 102.0, 'v': 1000, 'n': 50, 'vw': 101.5},
                    {'t': '2023-01-02T00:00:00Z', 'o': 102.0, 'h': 107.0, 'l': 101.0, 'c': 106.0, 'v': 1200, 'n': 60, 'vw': 103.5}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        # Call the method
        symbol = ['AAPL']
        timeframe = '1D'
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 31, tzinfo=timezone.utc)
        
        response, df = self.data_fetcher.get_historical_data(symbol, timeframe, start_date, end_date)
        
        # Verify the request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['headers'], self.data_fetcher.headers)
        self.assertEqual(kwargs['params']['symbols'], 'AAPL')
        self.assertEqual(kwargs['params']['timeframe'], '1D')
        
        # Verify the DataFrame is processed correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index.names, ['timestamp', 'symbol'])
    
    def test_get_latest_quote(self):
        """Test fetching the latest quote for a symbol"""
        # Mock quote for stock
        mock_stock_quote = MagicMock()
        mock_stock_quote.bid_price = "150.50"
        mock_stock_quote.ask_price = "150.75"
        mock_stock_quote.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        # Mock quote for crypto
        mock_crypto_quote = MagicMock()
        mock_crypto_quote.bid_price = "30000.50"
        mock_crypto_quote.ask_price = "30001.75"
        mock_crypto_quote.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        # Set return values for the mocks
        self.mock_trading_client.get_latest_quote.return_value = mock_stock_quote
        self.mock_trading_client.get_crypto_quote.return_value = mock_crypto_quote
        
        # Test stock quote
        stock_result = self.data_fetcher.get_latest_quote('AAPL')
        self.mock_trading_client.get_latest_quote.assert_called_once_with('AAPL')
        self.assertEqual(stock_result['bid'], 150.50)
        self.assertEqual(stock_result['ask'], 150.75)
        
        # Reset the mock
        self.mock_trading_client.get_latest_quote.reset_mock()
        
        # Test crypto quote
        crypto_result = self.data_fetcher.get_latest_quote('BTC/USD')
        self.mock_trading_client.get_crypto_quote.assert_called_once_with('BTC/USD')
        self.assertEqual(crypto_result['bid'], 30000.50)
        self.assertEqual(crypto_result['ask'], 30001.75)
    
    def test_get_account_balance(self):
        """Test getting account balance"""
        # Mock account response
        mock_account = MagicMock()
        mock_account.cash = "10000.50"
        self.mock_trading_client.get_account.return_value = mock_account
        
        # Call the method
        balance = self.data_fetcher.get_account_balance()
        
        # Verify the result
        self.mock_trading_client.get_account.assert_called_once()
        self.assertEqual(balance, 10000.50)
        
        # Test error handling
        self.mock_trading_client.get_account.side_effect = Exception("Test error")
        with self.assertRaises(Exception) as context:
            self.data_fetcher.get_account_balance()
        self.assertTrue("Error getting account balance: Test error" in str(context.exception))


class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for the TechnicalIndicators class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = {
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(95, 115, 50),
            'low': np.random.uniform(85, 105, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.randint(1000, 10000, 50)
        }
        self.sample_df = pd.DataFrame(data, index=dates)
        
        # Create indicator parameters
        self.params = {
            'sma_20': {'period': 20},
            'ema_14': {'period': 14},
            'rsi': {'period': 14},
            'atr': {'period': 14}
        }
        
        # Create the TechnicalIndicators instance
        self.indicators = TechnicalIndicators(self.sample_df, self.params)
        
        # Mock talib methods
        self.talib_patcher = patch('Indicators.talib')
        self.mock_talib = self.talib_patcher.start()
        
        # Configure mock returns
        self.mock_talib.SMA.return_value = pd.Series(np.random.uniform(90, 110, 50), index=dates)
        self.mock_talib.EMA.return_value = pd.Series(np.random.uniform(90, 110, 50), index=dates)
        self.mock_talib.RSI.return_value = pd.Series(np.random.uniform(30, 70, 50), index=dates)
        self.mock_talib.ATR.return_value = pd.Series(np.random.uniform(1, 5, 50), index=dates)
        self.mock_talib.BBANDS.return_value = (
            pd.Series(np.random.uniform(100, 120, 50), index=dates),
            pd.Series(np.random.uniform(90, 110, 50), index=dates),
            pd.Series(np.random.uniform(80, 100, 50), index=dates)
        )
        self.mock_talib.ADX.return_value = pd.Series(np.random.uniform(10, 50, 50), index=dates)
        self.mock_talib.OBV.return_value = pd.Series(np.random.uniform(100000, 200000, 50), index=dates)
        self.mock_talib.CCI.return_value = pd.Series(np.random.uniform(-100, 100, 50), index=dates)
    
    def tearDown(self):
        """Clean up after each test method"""
        self.talib_patcher.stop()
    
    def test_init_method(self):
        """Test if initialization sets up the parameters correctly"""
        # Test with provided parameters
        self.assertEqual(self.indicators.params, self.params)
        
        # Test with default parameters
        default_indicators = TechnicalIndicators(self.sample_df)
        self.assertIsNotNone(default_indicators.params)
        self.assertIn('sma', default_indicators.params)
        self.assertIn('ema', default_indicators.params)
        self.assertIn('rsi', default_indicators.params)
    
    def test_calculate_sma(self):
        """Test SMA calculation"""
        period = 20
        self.indicators.calculate_sma(period)
        self.mock_talib.SMA.assert_called_once_with(self.sample_df.close, period)
    
    def test_calculate_ema(self):
        """Test EMA calculation"""
        period = 14
        self.indicators.calculate_ema(period)
        self.mock_talib.EMA.assert_called_once_with(self.sample_df.close, period)
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        period = 14
        self.indicators.calculate_rsi(period)
        self.mock_talib.RSI.assert_called_once_with(self.sample_df.close, period)
        self.assertIn('rsi', self.indicators.df.columns)
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        period = 20
        std_dev = 2.0
        self.indicators.calculate_bollinger_bands(period, std_dev)
        self.mock_talib.BBANDS.assert_called_once_with(self.sample_df.close, period, std_dev)
        self.assertIn('upperband', self.indicators.df.columns)
        self.assertIn('middleband', self.indicators.df.columns)
        self.assertIn('lowerband', self.indicators.df.columns)
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        period = 14
        self.indicators.calculate_atr(period)
        self.mock_talib.ATR.assert_called_once_with(
            self.sample_df.high, self.sample_df.low, self.sample_df.close, period
        )
        self.assertIn('atr', self.indicators.df.columns)
    
    def test_calculate_indicators(self):
        """Test calculating all indicators"""
        # Mock the individual calculation methods
        with patch.object(self.indicators, '_calculate_previous_values') as mock_calc_prev:
            mock_calc_prev.return_value = self.sample_df
            
            # Call the method
            result = self.indicators.calculate_indicators()
            
            # Verify the result
            self.assertIsInstance(result, pd.DataFrame)
            mock_calc_prev.assert_called_once()
    
    def test_calculate_previous_values(self):
        """Test calculating previous values for indicators"""
        # Add some indicator columns to the DataFrame
        self.indicators.df['sma_20'] = np.random.uniform(90, 110, 50)
        self.indicators.df['rsi'] = np.random.uniform(30, 70, 50)
        
        # Call the method
        result = self.indicators._calculate_previous_values()
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('sma_20_prev', result.columns)
        self.assertIn('rsi_prev', result.columns)
        self.assertIn('close_prev', result.columns)


class TestTrueBautistStrategy(unittest.TestCase):
    """Test suite for the TrueBautistStrategy class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        # Create sample config and keys
        self.config = {
            'symbols': ['AAPL', 'MSFT'],
            'timeframe': '1D',
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-01-31T00:00:00',
            'indicators': [
                {'name': 'EMA', 'params': {'period': 20}},
                {'name': 'RSI', 'params': {'period': 14}}
            ],
            'entry_conditions': [
                {'indicator': 'rsi', 'condition': '<', 'value': 30}
            ],
            'exit_conditions': [
                {'indicator': 'rsi', 'condition': '>', 'value': 70}
            ],
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
        }
        
        self.keys = {
            'api_key_paper': 'test_api_key',
            'api_secret_paper': 'test_secret_key'
        }
        
        # Create the strategy instance
        self.strategy = TrueBautistStrategy(self.config, self.keys)
        
        # Mock external modules
        self.data_manager_patcher = patch('TrueBautist.AlpacaDataFetcher')
        self.indicators_patcher = patch('TrueBautist.TechnicalIndicators')
        
        self.mock_data_manager_class = self.data_manager_patcher.start()
        self.mock_indicators_class = self.indicators_patcher.start()
        
        # Configure mock data manager
        self.mock_data_manager = MagicMock()
        self.mock_data_manager_class.return_value = self.mock_data_manager
        
        # Sample data for get_historical_data
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        stock_data = {
            'timestamp': dates,
            'symbol': ['AAPL'] * 10 + ['MSFT'] * 10,
            'open': np.random.uniform(90, 110, 20),
            'high': np.random.uniform(95, 115, 20),
            'low': np.random.uniform(85, 105, 20),
            'close': np.random.uniform(90, 110, 20),
            'volume': np.random.randint(1000, 10000, 20)
        }
        self.sample_df = pd.DataFrame(stock_data)
        self.mock_data_manager.get_historical_data.return_value = (MagicMock(), self.sample_df)
    
    def tearDown(self):
        """Clean up after each test method"""
        self.data_manager_patcher.stop()
        self.indicators_patcher.stop()
    
    def test_init_method(self):
        """Test if initialization sets up the strategy correctly"""
        self.assertEqual(self.strategy.config, self.config)
        self.assertEqual(self.strategy.keys, self.keys)
        self.assertIsInstance(self.strategy.indicators, dict)
        self.assertIsInstance(self.strategy.entry_conditions, list)
        self.assertIsInstance(self.strategy.exit_conditions, list)
        self.assertIsInstance(self.strategy.risk_management, dict)
    
    def test_get_indicators(self):
        """Test parsing indicators from config"""
        # Reset indicators and call get_indicators
        self.strategy.indicators = {}
        result = self.strategy.get_indicators()
        
        # Verify the result
        self.assertIn('ema_20', result)
        self.assertIn('rsi', result)
        self.assertEqual(result['ema_20']['period'], 20)
        self.assertEqual(result['rsi']['period'], 14)
    
    def test_get_risk_management(self):
        """Test parsing risk management from config"""
        # Reset risk management and call get_risk_management
        self.strategy.risk_management = {}
        result = self.strategy.get_risk_management()
        
        # Verify the result
        self.assertIn('max_position_size', result)
        self.assertIn('stop_loss', result)
        self.assertIn('take_profit', result)
        self.assertEqual(result['max_position_size'], 0.1)
        self.assertEqual(result['stop_loss'], 0.05)
        self.assertEqual(result['take_profit'], 0.1)
    
    def test_get_entry_conditions(self):
        """Test parsing entry conditions from config"""
        # Reset entry conditions and call get_entry_conditions
        self.strategy.entry_conditions = []
        result = self.strategy.get_entry_conditions()
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['indicator'], 'rsi')
        self.assertEqual(result[0]['condition'], '<')
        self.assertEqual(result[0]['value'], 30)
    
    def test_get_exit_conditions(self):
        """Test parsing exit conditions from config"""
        # Reset exit conditions and call get_exit_conditions
        self.strategy.exit_conditions = []
        result = self.strategy.get_exit_conditions()
        
        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['indicator'], 'rsi')
        self.assertEqual(result[0]['condition'], '>')
        self.assertEqual(result[0]['value'], 70)
    
    def test_get_data(self):
        """Test fetching data and calculating indicators"""
        # Mock TechnicalIndicators
        mock_indicators = MagicMock()
        mock_indicators.calculate_indicators.return_value = self.sample_df
        self.mock_indicators_class.return_value = mock_indicators
        
        # Call get_data
        result = self.strategy.get_data()
        
        # Verify the result
        self.mock_data_manager_class.assert_called_once_with(
            self.keys['api_key_paper'], self.keys['api_secret_paper']
        )
        self.mock_data_manager.get_historical_data.assert_called_once()
        self.mock_indicators_class.assert_called_once()
        mock_indicators.calculate_indicators.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_parse_eastern_date(self):
        """Test parsing date strings to datetime objects"""
        # Test with naive datetime string
        result1 = self.strategy._parse_eastern_date('2023-01-01T12:00:00')
        self.assertIsInstance(result1, datetime)
        self.assertIsNotNone(result1.tzinfo)
        
        # Test with timezone-aware datetime string
        result2 = self.strategy._parse_eastern_date('2023-01-01T12:00:00-05:00')
        self.assertIsInstance(result2, datetime)
        self.assertIsNotNone(result2.tzinfo)
    
    def test_calculate_metrics(self):
        """Test calculating performance metrics"""
        # Create sample trades and equity data
        trades = pd.DataFrame({
            'entry_time': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'exit_time': pd.date_range(start='2023-01-02', periods=5, freq='D'),
            'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL'],
            'entry_price': [100, 200, 105, 205, 110],
            'exit_price': [105, 210, 100, 200, 115],
            'quantity': [10, 5, 10, 5, 10],
            'pnl': [50, 50, -50, -25, 50]
        })
        
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        equity = pd.Series([10000, 10050, 10100, 10050, 10025, 10075, 10100, 10150, 10200, 10250], index=dates)
        
        # Call the method
        metrics = self.strategy._calculate_metrics(trades, equity, 10000)
        
        # Verify the result
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['total_trades'], 5)
        self.assertEqual(metrics['winning_trades'], 3)
        self.assertEqual(metrics['losing_trades'], 2)
        self.assertEqual(metrics['win_rate'], 0.6)
        self.assertEqual(metrics['total_return'], 0.02)
        
        # Test with empty trades
        empty_metrics = self.strategy._calculate_metrics(pd.DataFrame(), pd.Series(), 10000)
        self.assertEqual(empty_metrics['total_trades'], 0)
        self.assertEqual(empty_metrics['win_rate'], 0)


class TestMarketScanner(unittest.TestCase):
    """Test suite for the MarketScanner class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.api_key = "test_api_key"
        self.secret_key = "test_secret_key"
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()
        from alpaca.data.timeframe import TimeFrame
        self.timeframe = TimeFrame.Day
        
        # Mock StockHistoricalDataClient
        self.data_client_patcher = patch('Scanner.StockHistoricalDataClient')
        self.mock_data_client_class = self.data_client_patcher.start()
        self.mock_data_client = MagicMock()
        self.mock_data_client_class.return_value = self.mock_data_client
        
        # Create the scanner instance
        self.scanner = MarketScanner(
            self.api_key, 
            self.secret_key, 
            [], 
            self.timeframe,
            self.start_date,
            self.end_date
        )
    
    def tearDown(self):
        """Clean up after each test method"""
        self.data_client_patcher.stop()
    
    def test_init_method(self):
        """Test if initialization sets up the scanner correctly"""
        self.mock_data_client_class.assert_called_once_with(self.api_key, self.secret_key)
        self.assertEqual(self.scanner.watchlist, [])
        self.assertEqual(self.scanner.timeframe, self.timeframe)
        self.assertEqual(self.scanner.starttime, self.start_date)
        self.assertEqual(self.scanner.endtime, self.end_date)
        
        # Check if macro symbols are set
        self.assertIn('indices', self.scanner.macro_symbols)
        self.assertIn('sectors', self.scanner.macro_symbols)
        self.assertIn('volatility', self.scanner.macro_symbols)
    
    def test_set_watchlist_to_iwm(self):
        """Test setting watchlist to IWM (Russell 2000) stocks"""
        result = self.scanner.set_watchlist_to_iwm()
        self.assertEqual(result, self.scanner.watchlist)
        self.assertEqual(len(result), len(self.scanner.iwm_stocks))
        self.assertIn('STBA', result)  # Sample stock from Financial Services
        self.assertIn('OMCL', result)  # Sample stock from Healthcare
    
    def test_set_watchlist_to_qqq(self):
        """Test setting watchlist to QQQ (Nasdaq-100) stocks"""
        result = self.scanner.set_watchlist_to_qqq()
        self.assertEqual(result, self.scanner.watchlist)
        self.assertEqual(len(result), len(self.scanner.qqq_stocks))
        self.assertIn('AAPL', result)  # Sample stock from Technology
        self.assertIn('AMZN', result)  # Sample stock from Consumer Discretionary
    
    @patch('Scanner.MarketScanner._get_stock_data')
    def test_scan_for_volume_breakouts(self, mock_get_stock_data):
        """Test scanning for volume breakouts"""
        # Set watchlist
        self.scanner.watchlist = ['AAPL', 'MSFT']
        
        # Create sample data
        aapl_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'open': np.random.uniform(90, 110, 30),
            'high': np.random.uniform(95, 115, 30),
            'low': np.random.uniform(85, 105, 30),
            'close': np.random.uniform(90, 110, 30),
            'volume': np.random.randint(1000, 10000, 30)
        })
        aapl_data.set_index('timestamp', inplace=True)
        
        # Set the last day's volume to trigger the breakout
        aapl_data.iloc[-1, aapl_data.columns.get_loc('volume')] = 15000
        aapl_data.iloc[-1, aapl_data.columns.get_loc('close')] = 105
        aapl_data.iloc[-2, aapl_data.columns.get_loc('close')] = 100
        
        msft_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'open': np.random.uniform(190, 210, 30),
            'high': np.random.uniform(195, 215, 30),
            'low': np.random.uniform(185, 205, 30),
            'close': np.random.uniform(190, 210, 30),
            'volume': np.random.randint(2000, 20000, 30)
        })
        msft_data.set_index('timestamp', inplace=True)
        
        # Create the mock return value
        mock_get_stock_data.return_value = {'AAPL': aapl_data, 'MSFT': msft_data}, None
        
        # Call the method
        breakout_candidates, data = self.scanner.scan_for_volume_breakouts(
            volume_threshold=1.2, price_change_threshold=1.0
        )
        
        # Verify the result
        self.assertIsInstance(breakout_candidates, dict)
        self.assertIn('AAPL', breakout_candidates)  # AAPL should be detected as a breakout
        self.assertIn('volume_ratio', breakout_candidates['AAPL'])
        self.assertIn('price_change', breakout_candidates['AAPL'])
        self.assertIn('breakout_direction', breakout_candidates['AAPL'])
    
    def test_calculate_atr(self):
        """Test ATR calculation in MarketScanner"""
        # Create sample data
        df = pd.DataFrame({
            'high': [110, 112, 108, 115, 113],
            'low': [105, 107, 103, 110, 108],
            'close': [108, 109, 105, 113, 110]
        })
        
        # Call the method
        atr = self.scanner._calculate_atr(df, period=3)
        
        # Verify the result
        self.assertIsInstance(atr, float)
        self.assertGreater(atr, 0)  # ATR should be positive
    
    @patch('Scanner.MarketScanner._get_stock_data')
    @patch('Scanner.MarketScanner._analyze_market_trend')
    @patch('Scanner.MarketScanner._analyze_volatility')
    @patch('Scanner.MarketScanner._analyze_market_breadth')
    @patch('Scanner.MarketScanner._analyze_sentiment')
    @patch('Scanner.MarketScanner._calculate_risk_level')
    @patch('Scanner.MarketScanner._analyze_sector_performance')
    @patch('Scanner.MarketScanner._calculate_correlation_matrix')
    async def test_analyze_macro_environment(
        self, mock_correlation, mock_sector, mock_risk, mock_sentiment, 
        mock_breadth, mock_volatility, mock_trend, mock_get_data
    ):
        """Test analyzing macro environment"""
        # Create mock return values
        mock_data = {'SPY': pd.DataFrame(), 'QQQ': pd.DataFrame()}
        mock_get_data.return_value = mock_data
        
        mock_trend.return_value = {'overall': 'bullish', 'details': {}}
        mock_volatility.return_value = {'overall': 'low', 'details': {}}
        mock_breadth.return_value = {'overall': 'expanding', 'details': {}}
        mock_sentiment.return_value = {'overall': 'positive', 'details': {}}
        mock_risk.return_value = 'low'
        mock_sector.return_value = {'XLK': {}, 'XLF': {}}
        mock_correlation.return_value = pd.DataFrame()
        
        # Call the method
        result = await self.scanner.analyze_macro_environment()
        
        # Verify the result
        self.assertIsInstance(result, MarketCondition)
        self.assertEqual(result.trend, 'bullish')
        self.assertEqual(result.volatility, 'low')
        self.assertEqual(result.breadth, 'expanding')
        self.assertEqual(result.sentiment, 'positive')
        self.assertEqual(result.risk_level, 'low')
        self.assertIsInstance(result.details, dict)
        
        # Verify method calls
        mock_get_data.assert_called_once()
        mock_trend.assert_called_once_with(mock_data)
        mock_volatility.assert_called_once_with(mock_data)
        mock_breadth.assert_called_once_with(mock_data)
        mock_sentiment.assert_called_once_with(mock_data)
        mock_risk.assert_called_once_with(
            mock_trend.return_value, 
            mock_volatility.return_value,
            mock_breadth.return_value,
            mock_sentiment.return_value
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for the trading system components"""
    
    def setUp(self):
        """Set up test environment for integration tests"""
        # Mock API client
        self.api_key = "test_api_key"
        self.secret_key = "test_secret_key"
        
        # Create sample config for TrueBautistStrategy
        self.config = {
            'symbols': ['AAPL', 'MSFT'],
            'timeframe': '1D',
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-01-31T00:00:00',
            'indicators': [
                {'name': 'EMA', 'params': {'period': 20}},
                {'name': 'RSI', 'params': {'period': 14}}
            ],
            'entry_conditions': [
                {'indicator': 'rsi', 'condition': '<', 'value': 30}
            ],
            'exit_conditions': [
                {'indicator': 'rsi', 'condition': '>', 'value': 70}
            ],
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
        }
        
        self.keys = {
            'api_key_paper': self.api_key,
            'api_secret_paper': self.secret_key
        }
        
        # Apply patches for external API calls
        self.alpaca_client_patcher = patch('AlpacaDataManager.StockHistoricalDataClient')
        self.alpaca_crypto_client_patcher = patch('AlpacaDataManager.CryptoHistoricalDataClient')
        self.alpaca_trading_client_patcher = patch('AlpacaDataManager.TradingClient')
        self.requests_patcher = patch('AlpacaDataManager.requests')
        
        # Start patchers
        self.mock_alpaca_client = self.alpaca_client_patcher.start()
        self.mock_alpaca_crypto_client = self.alpaca_crypto_client_patcher.start()
        self.mock_alpaca_trading_client = self.alpaca_trading_client_patcher.start()
        self.mock_requests = self.requests_patcher.start()
        
        # Configure mock response for requests
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': '2023-01-01T00:00:00Z', 'o': 100.0, 'h': 105.0, 'l': 99.0, 'c': 102.0, 'v': 1000, 'n': 50, 'vw': 101.5},
                    {'t': '2023-01-02T00:00:00Z', 'o': 102.0, 'h': 107.0, 'l': 101.0, 'c': 106.0, 'v': 1200, 'n': 60, 'vw': 103.5}
                ],
                'MSFT': [
                    {'t': '2023-01-01T00:00:00Z', 'o': 200.0, 'h': 205.0, 'l': 199.0, 'c': 202.0, 'v': 2000, 'n': 100, 'vw': 201.5},
                    {'t': '2023-01-02T00:00:00Z', 'o': 202.0, 'h': 207.0, 'l': 201.0, 'c': 206.0, 'v': 2200, 'n': 120, 'vw': 203.5}
                ]
            }
        }
        self.mock_requests.get.return_value = mock_response
    
    def tearDown(self):
        """Clean up after each test method"""
        self.alpaca_client_patcher.stop()
        self.alpaca_crypto_client_patcher.stop()
        self.alpaca_trading_client_patcher.stop()
        self.requests_patcher.stop()
    
    @patch('TrueBautist.TechnicalIndicators')
    def test_strategy_with_data_fetcher(self, mock_indicators_class):
        """Test integration between TrueBautistStrategy and AlpacaDataFetcher"""
        # Create sample DataFrame for indicators
        mock_indicators = MagicMock()
        mock_indicators.calculate_indicators.return_value = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=2, freq='D'),
            'symbol': ['AAPL', 'MSFT'],
            'open': [100.0, 200.0],
            'high': [105.0, 205.0],
            'low': [99.0, 199.0],
            'close': [102.0, 202.0],
            'volume': [1000, 2000],
            'ema_20': [101.0, 201.0],
            'rsi': [45.0, 55.0]
        })
        mock_indicators_class.return_value = mock_indicators
        
        # Create strategy
        strategy = TrueBautistStrategy(self.config, self.keys)
        
        # Get data
        result = strategy.get_data()
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        mock_indicators_class.assert_called_once()
        mock_indicators.calculate_indicators.assert_called_once()
    
    @patch('Scanner.StockHistoricalDataClient')
    def test_scanner_with_timeframe(self, mock_client_class):
        """Test MarketScanner with TimeFrame integration"""
        # Set up mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_bars = MagicMock()
        mock_bars.data = {
            'SPY': [
                MagicMock(timestamp='2023-01-01T00:00:00Z', open=400.0, high=405.0, low=399.0, close=402.0, volume=10000, trade_count=500, vwap=401.5),
                MagicMock(timestamp='2023-01-02T00:00:00Z', open=402.0, high=407.0, low=401.0, close=406.0, volume=12000, trade_count=600, vwap=403.5)
            ]
        }
        mock_client.get_stock_bars.return_value = mock_bars
        
        from alpaca.data.timeframe import TimeFrame
        
        # Create scanner with TimeFrame object
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        scanner = MarketScanner(
            self.api_key, 
            self.secret_key, 
            ['SPY'], 
            TimeFrame.Day,
            start_date,
            end_date
        )
        
        # Call get_stock_data
        result = scanner._get_stock_data(['SPY'], start_date, TimeFrame.Day, end_date)
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('SPY', result)
        self.assertIsInstance(result['SPY'], pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
