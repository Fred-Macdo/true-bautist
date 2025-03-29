# Alpaca Trading System Test Suite

This test suite provides comprehensive testing for the Alpaca trading system components:

- `AlpacaDataManager.py` - Data fetching
- `Indicators.py` - Technical indicators
- `TrueBautist.py` - Strategy implementation
- `Scanner.py` - Market scanning

## Files

- `test_suite.py` - Main test file containing all unit tests
- `test_utils.py` - Utility functions for creating test data
- `run_tests.py` - Script to run the tests

## Setup

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Make sure your project structure is:

```
project/
├── AlpacaDataManager.py
├── Indicators.py
├── TrueBautist.py
├── Scanner.py
├── tests/
│   ├── test_suite.py
│   ├── test_utils.py
│   └── run_tests.py
```

## Running Tests

To run all tests:

```bash
python tests/run_tests.py
```

To run specific test classes:

```bash
python -m unittest tests.test_suite.TestAlpacaDataFetcher
```

To run a specific test method:

```bash
python -m unittest tests.test_suite.TestAlpacaDataFetcher.test_get_historical_data
```

## Test Categories

### Unit Tests

- `TestAlpacaDataFetcher` - Tests for the AlpacaDataFetcher class
- `TestTechnicalIndicators` - Tests for the TechnicalIndicators class
- `TestTrueBautistStrategy` - Tests for the TrueBautistStrategy class
- `TestMarketScanner` - Tests for the MarketScanner class

### Integration Tests

- `TestIntegration` - Tests for component interactions

## Mocking

The tests use `unittest.mock` to mock external dependencies:

- Alpaca API clients are mocked to avoid actual API calls
- `requests` module is mocked for HTTP requests
- `talib` functions are mocked for indicator calculations

## Test Data Utilities

The `test_utils.py` file provides helper functions for creating test data:

- `create_ohlcv_dataframe()` - Create sample OHLCV data
- `create_mock_alpaca_response()` - Create mock Alpaca API responses
- `create_sample_strategy_config()` - Create sample strategy configurations
- `create_sample_market_condition()` - Create sample market condition results
- `create_sample_trades()` - Create sample trade data

## Adding New Tests

When adding new components or features:

1. Add unit tests to the appropriate test class
2. Add integration tests if the feature interacts with other components
3. Add utility functions in `test_utils.py` if needed for test data

## Coverage

To run tests with coverage reporting:

```bash
pip install coverage
coverage run --source=. tests/run_tests.py
coverage report -m
coverage html  # For HTML report
```

This will generate a coverage report showing which lines of code are tested.
