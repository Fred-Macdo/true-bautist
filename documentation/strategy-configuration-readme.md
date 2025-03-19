# Trading Strategy System

A powerful framework for developing, testing, and executing algorithmic trading strategies. This system combines technical indicators, customizable entry/exit conditions, and comprehensive risk management to create robust trading strategies.

## Table of Contents

- [Overview](#overview)
- [Technical Indicators](#technical-indicators)
- [Strategy Configuration](#strategy-configuration)
- [Entry and Exit Conditions](#entry-and-exit-conditions)
- [Risk Management](#risk-management)
- [Usage Guide](#usage-guide)
- [Strategy Examples](#strategy-examples)

## Overview

The Trading Strategy System allows you to:

1. **Define strategies** using a simple YAML configuration
2. **Test strategies** with historical market data
3. **Analyze performance** through detailed metrics
4. **Optimize parameters** to improve results
5. **Execute strategies** in live or paper trading environments

The system is built around the concept of technical analysis indicators combined with rule-based entry and exit conditions.

## Technical Indicators

The following technical indicators are available through the `Indicators.py` module:

### Moving Averages

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| [**SMA**](https://www.investopedia.com/terms/s/sma.asp) | Simple Moving Average | `period`: Number of periods to average (default: 20) |
| [**EMA**](https://www.investopedia.com/terms/e/ema.asp) | Exponential Moving Average | `period`: Number of periods with exponential weighting (default: 20) |

### Oscillators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| [**RSI**](https://www.investopedia.com/terms/r/rsi.asp) | Relative Strength Index | `period`: Lookback period (default: 14) |
| [**CCI**](https://www.investopedia.com/investing/timing-trades-with-commodity-channel-index/#toc-understanding-the-cci) | Commodity Channel Index | `period`: Lookback period (default: 20) |
| [**MFI**](https://www.investopedia.com/terms/m/mfi.asp) | Money Flow Index | `period`: Lookback period (default: 14) |

### Volatility Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| [**ATR**](https://www.investopedia.com/terms/a/atr.asp) | Average True Range | `period`: Lookback period (default: 14) |
| [**BBANDS**](https://www.investopedia.com/terms/b/bollingerbands.asp) | Bollinger Bands | `period`: MA period (default: 20)<br>`std_dev`: Standard deviation multiplier (default: 2) |
| [**Keltner Channels**](https://www.avatrade.com/education/technical-analysis-indicators-strategies/keltner-channel-trading) | Volatility-based channels | `period`: MA period (default: 20)<br>`atr_multiplier`: ATR multiplier (default: 2) |

### Trend Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| [**ADX**](https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp) | Average Directional Index | `period`: Lookback period (default: 14) |
| [**MACD**](https://www.investopedia.com/terms/m/macd.asp) | Moving Average Convergence Divergence | `fast_period`: Fast EMA period (default: 12)<br>`slow_period`: Slow EMA period (default: 26)<br>`signal_period`: Signal line period (default: 9) |
| [**STOCH**](https://www.investopedia.com/terms/s/stochasticoscillator.asp) | Stochastic Oscillator | `k_period`: %K period (default: 14)<br>`d_period`: %D period (default: 3)<br>`slowing`: Slowing period (default: 3) |

### Volume Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| [**OBV**](https://www.investopedia.com/terms/o/onbalancevolume.asp) | On-Balance Volume | No parameters required |
| [**VWAP**](https://www.investopedia.com/terms/v/vwap.asp) | Volume-Weighted Average Price | `period`: Lookback period (default: 5) |

### Output Values

Each indicator produces specific output values that can be referenced in entry/exit conditions:

- **SMA/EMA**: Single value (`sma_XX`, `ema_XX` where XX is the period)
- **RSI**: Single value (`rsi`)
- **BBANDS**: Three values (`upperband`, `middleband`, `lowerband`)
- **ATR**: Single value (`atr`)
- **Keltner Channels**: Three values (`keltner_upper`, `keltner_middle`, `keltner_lower`)
- **ADX**: Single value (`adx`)
- **OBV**: Single value (`obv`)
- **MFI**: Single value (`mfi`)
- **CCI**: Single value (`cci`)
- **VWAP**: Single value (`vwap`)
- **MACD**: Three values (`macd`, `macd_signal`, `macd_hist`)
- **STOCH**: Two values (`slowk`, `slowd`)

## Strategy Configuration

Strategies are defined using YAML configurations. Here's the structure:

```yaml
# Trading symbols
symbols: ["AAPL", "MSFT", "GOOG"]

# Timeframe for analysis
timeframe: "5Min"  # Options: 1Min, 5Min, 15Min, 30Min, 1h, 1d, 1w, 1month

# Date range for backtesting
start_date: "2023-01-01"
end_date: "2024-01-01"

# Entry conditions
entry_conditions:
  - indicator: "close"
    comparison: "above"
    value: "sma_20"
  - indicator: "rsi"
    comparison: "below"
    value: "30"

# Exit conditions
exit_conditions:
  - indicator: "close"
    comparison: "below"
    value: "sma_20"
  - indicator: "rsi"
    comparison: "above"
    value: "70"

# Risk management parameters
risk_management:
  position_sizing_method: "risk_based"  # Options: risk_based, atr_based, fixed, percentage
  risk_per_trade: 1.0  # Percentage of account to risk per trade
  stop_loss: 0.02  # 2% stop loss
  take_profit: 0.06  # 6% take profit
  max_position_size: 1000.0  # Maximum position size in currency units
  atr_multiplier: 2.0  # For ATR-based position sizing

# Required indicators
indicators:
  - name: "SMA"
    params:
      period: 20
  - name: "RSI"
    params:
      period: 14
  - name: "BBANDS"
    params:
      period: 20
      std_dev: 2
```

## Entry and Exit Conditions

Entry and exit conditions use a simple rule-based syntax:

```yaml
- indicator: "INDICATOR_NAME"
  comparison: "COMPARISON_OPERATOR"
  value: "VALUE_OR_INDICATOR"
```

### Available Indicators

You can use any of these values as the `indicator` or `value`:

- Price data: `open`, `high`, `low`, `close`, `volume`
- Technical indicators: Any indicator output from the configured indicators
  - Moving averages: `sma_XX`, `ema_XX` (where XX is the period)
  - Oscillators: `rsi`, `cci`, `mfi`
  - Volatility: `atr`, `upperband`, `middleband`, `lowerband`, `keltner_upper`, `keltner_middle`, `keltner_lower`
  - Trend: `adx`, `macd`, `macd_signal`, `macd_hist`
  - Volume: `obv`, `vwap`

### Comparison Operators

The following comparison operators are available:

- `above`: Indicator is greater than value
- `below`: Indicator is less than value
- `between`: Indicator is between two values (requires a range passed between brackets: [ ])
- `crosses_above`: Indicator crosses above value
- `crosses_below`: Indicator crosses below value

### Value Types

The `value` field can be:

1. Another indicator (e.g., `sma_50`)
2. A numeric constant (e.g., `70` for RSI)
3. A price level (e.g., `145.50`)

## Risk Management

The system provides several position sizing methods and risk control parameters:

### Position Sizing Methods

1. **Risk-Based**: Sizes positions based on the specified percentage risk per trade and the distance to stop loss
2. **ATR-Based**: Uses Average True Range to determine position size
3. **Fixed**: Uses a fixed position size for all trades
4. **Percentage**: Uses a percentage of available capital for each trade

### Risk Parameters

- **Risk Per Trade**: Percentage of account to risk on each trade
- **Stop Loss**: Percentage or ATR-based stop loss level
- **Take Profit**: Percentage or ATR-based take profit level
- **Max Position Size**: Maximum position size in currency units
- **ATR Multiplier**: Multiplier for ATR-based stops and targets

## Usage Guide

### Creating a Strategy

1. Start by defining the symbols, timeframe, and date range
2. Configure the technical indicators needed for your strategy
3. Define entry conditions that must be met to enter a trade
4. Define exit conditions that trigger closing a position
5. Set up risk management parameters

### Using the YAML Editor

The system includes a web-based YAML editor for easily creating and editing strategies:

1. Access the editor at `/strategy-editor`
2. Configure your strategy using the interactive forms
3. Download the generated YAML file
4. Use the file with the backtesting or live trading system

### Backtesting a Strategy

Run your strategy against historical data:

```bash
python backtest.py --config your_strategy.yaml
```

### Analyzing Results

The backtest will generate performance metrics including:

- Total return
- Win rate
- Profit factor
- Maximum drawdown
- Sharpe ratio

## Strategy Examples

### Simple Moving Average Crossover

```yaml
symbols: ["SPY"]
timeframe: "1d"
start_date: "2023-01-01"
end_date: "2024-01-01"

entry_conditions:
  - indicator: "ema_12"
    comparison: "crosses_above"
    value: "ema_26"

exit_conditions:
  - indicator: "ema_12"
    comparison: "crosses_below"
    value: "ema_26"

risk_management:
  position_sizing_method: "percentage"
  risk_per_trade: 2.0
  stop_loss: 0.05
  take_profit: 0.15
  max_position_size: 10000.0

indicators:
  - name: "EMA"
    params:
      period: 12
  - name: "EMA"
    params:
      period: 26
```

### RSI Mean Reversion

```yaml
symbols: ["AAPL", "MSFT", "GOOG", "AMZN"]
timeframe: "1d"
start_date: "2023-01-01"
end_date: "2024-01-01"

entry_conditions:
  - indicator: "rsi"
    comparison: "below"
    value: "30"
  - indicator: "close"
    comparison: "above"
    value: "sma_200"

exit_conditions:
  - indicator: "rsi"
    comparison: "above"
    value: "70"
  - indicator: "close"
    comparison: "below"
    value: "sma_50"

risk_management:
  position_sizing_method: "risk_based"
  risk_per_trade: 1.0
  stop_loss: 0.03
  take_profit: 0.09
  max_position_size: 5000.0

indicators:
  - name: "RSI"
    params:
      period: 14
  - name: "SMA"
    params:
      period: 50
  - name: "SMA"
    params:
      period: 200
```

### Bollinger Band Breakout

```yaml
symbols: ["QQQ"]
timeframe: "1h"
start_date: "2023-01-01"
end_date: "2024-01-01"

entry_conditions:
  - indicator: "close"
    comparison: "above"
    value: "upperband"
  - indicator: "volume"
    comparison: "above"
    value: "50000"

exit_conditions:
  - indicator: "close"
    comparison: "below"
    value: "middleband"
  - indicator: "rsi"
    comparison: "above"
    value: "80"

risk_management:
  position_sizing_method: "atr_based"
  risk_per_trade: 1.0
  stop_loss: 0.0
  take_profit: 0.0
  max_position_size: 5000.0
  atr_multiplier: 3.0

indicators:
  - name: "BBANDS"
    params:
      period: 20
      std_dev: 2
  - name: "RSI"
    params:
      period: 14
  - name: "ATR"
    params:
      period: 14
```
