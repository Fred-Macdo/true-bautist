from typing import Dict, Any, Callable
import talib
import pandas as pd

class TALibIndicators:
    """Mapping of TA-Lib functions with their parameters and descriptions"""
    
    # Momentum Indicators
    MOMENTUM_INDICATORS = {
        'RSI': {
            'function': talib.RSI,
            'params': {
                'timeperiod': 14
            },
            'inputs': ['close'],
            'outputs': ['rsi']
        },
        'MACD': {
            'function': talib.MACD,
            'params': {
                'fastperiod': 12,
                'slowperiod': 26,
                'signalperiod': 9
            },
            'inputs': ['close'],
            'outputs': ['macd', 'macdsignal', 'macdhist']
        },
        'STOCH': {
            'function': talib.STOCH,
            'params': {
                'fastk_period': 5,
                'slowk_period': 3,
                'slowk_matype': 0,
                'slowd_period': 3,
                'slowd_matype': 0
            },
            'inputs': ['high', 'low', 'close'],
            'outputs': ['slowk', 'slowd']
        },
        'ADX': {
            'function': talib.ADX,
            'params': {
                'timeperiod': 14
            },
            'inputs': ['high', 'low', 'close'],
            'outputs': ['adx']
        }
    }

    # Trend Indicators
    TREND_INDICATORS = {
        'SMA': {
            'function': talib.SMA,
            'params': {
                'timeperiod': 30
            },
            'inputs': ['close'],
            'outputs': ['sma']
        },
        'EMA': {
            'function': talib.EMA,
            'params': {
                'timeperiod': 30
            },
            'inputs': ['close'],
            'outputs': ['ema']
        },
        'BBANDS': {
            'function': talib.BBANDS,
            'params': {
                'timeperiod': 20,
                'nbdevup': 2,
                'nbdevdn': 2,
                'matype': 0
            },
            'inputs': ['close'],
            'outputs': ['upperband', 'middleband', 'lowerband']
        },
        'SAR': {
            'function': talib.SAR,
            'params': {
                'acceleration': 0.02,
                'maximum': 0.2
            },
            'inputs': ['high', 'low'],
            'outputs': ['sar']
        }
    }

    # Volume Indicators
    VOLUME_INDICATORS = {
        'OBV': {
            'function': talib.OBV,
            'params': {},
            'inputs': ['close', 'volume'],
            'outputs': ['obv']
        },
        'AD': {
            'function': talib.AD,
            'params': {},
            'inputs': ['high', 'low', 'close', 'volume'],
            'outputs': ['ad']
        },
        'ADOSC': {
            'function': talib.ADOSC,
            'params': {
                'fastperiod': 3,
                'slowperiod': 10
            },
            'inputs': ['high', 'low', 'close', 'volume'],
            'outputs': ['adosc']
        }
    }

    # Volatility Indicators
    VOLATILITY_INDICATORS = {
        'ATR': {
            'function': talib.ATR,
            'params': {
                'timeperiod': 14
            },
            'inputs': ['high', 'low', 'close'],
            'outputs': ['atr']
        },
        'NATR': {
            'function': talib.NATR,
            'params': {
                'timeperiod': 14
            },
            'inputs': ['high', 'low', 'close'],
            'outputs': ['natr']
        }
    }

    # Cycle Indicators
    CYCLE_INDICATORS = {
        'HT_DCPERIOD': {
            'function': talib.HT_DCPERIOD,
            'params': {},
            'inputs': ['close'],
            'outputs': ['dcperiod']
        },
        'HT_DCPHASE': {
            'function': talib.HT_DCPHASE,
            'params': {},
            'inputs': ['close'],
            'outputs': ['dcphase']
        }
    }

    # Price Pattern Recognition
    PATTERN_INDICATORS = {
        'CDLDOJI': {
            'function': talib.CDLDOJI,
            'params': {},
            'inputs': ['open', 'high', 'low', 'close'],
            'outputs': ['doji']
        },
        'CDLENGULFING': {
            'function': talib.CDLENGULFING,
            'params': {},
            'inputs': ['open', 'high', 'low', 'close'],
            'outputs': ['engulfing']
        },
        'CDLHAMMER': {
            'function': talib.CDLHAMMER,
            'params': {},
            'inputs': ['open', 'high', 'low', 'close'],
            'outputs': ['hammer']
        }
    }

    @classmethod
    def get_all_indicators(cls) -> Dict[str, Dict]:
        """Get all available indicators"""
        return {
            **cls.MOMENTUM_INDICATORS,
            **cls.TREND_INDICATORS,
            **cls.VOLUME_INDICATORS,
            **cls.VOLATILITY_INDICATORS,
            **cls.CYCLE_INDICATORS,
            **cls.PATTERN_INDICATORS
        }

    @classmethod
    def calculate_indicator(cls, indicator_name: str, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Calculate a specific indicator"""
        indicators = cls.get_all_indicators()
        if indicator_name not in indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        indicator = indicators[indicator_name]
        function = indicator['function']
        default_params = indicator['params']
        inputs = indicator['inputs']
        outputs = indicator['outputs']
        
        # Merge default params with provided params
        if params:
            default_params.update(params)
        
        # Prepare input data
        input_data = [data[input_name] for input_name in inputs]
        
        # Calculate indicator
        results = function(*input_data, **default_params)
        
        # Handle multiple outputs
        if isinstance(results, tuple):
            return pd.DataFrame({
                output_name: result 
                for output_name, result in zip(outputs, results)
            }, index=data.index)
        else:
            return pd.DataFrame({
                outputs[0]: results
            }, index=data.index)