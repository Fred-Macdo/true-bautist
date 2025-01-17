from abc import ABC, abstractmethod
from codecs import utf_16_be_decode
from dataclasses import dataclass
from math import e
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

class Condition(ABC):
    def __init__(self, indicator: str, comparison: str, value: Any):
        self.indicator = indicator
        self.comparison = comparison
        self.value = value
        

    @abstractmethod
    def evaluate(self, row: pd.Series) -> bool:
        pass

class IndicatorCondition(Condition):
    def __str__(self):
        return f'The indicator type is {self.indicator} and the comparison is {self.comparison} using {self.value}'

    def __repr__(self):
        return f'IndicatorCondition(\'{self.indicator}\', {self.comparison}, {self.value})'
    
    def evaluate(self, row: pd.Series) -> bool:
        # Print available columns for debugging
        print(f"\nEvaluating {self.indicator} condition:")
        print(f"Comparison: {self.comparison}")
        print(f"Value: {self.value}")
        
        # Handle special cases for indicators
        if self.indicator == "MACD" and self.value == "signal":
            macd_val = row.get('macd')
            signal_val = row.get('macdsignal')
            macd_prev = row.get('macd_prev')
            signal_prev = row.get('macdsignal_prev')
            
            print(f"MACD values:")
            print(f"Current MACD: {macd_val}")
            print(f"Current Signal: {signal_val}")
            print(f"Previous MACD: {macd_prev}")
            print(f"Previous Signal: {signal_prev}")
            
            if any(pd.isna([macd_val, signal_val, macd_prev, signal_prev])):
                print("Missing or NaN values detected")
                return False
                
            if self.comparison == "crosses_above":
                result = (macd_prev <= signal_prev) and (macd_val > signal_val)
                print(f"Crosses above evaluation: {result}")
                return result
            elif self.comparison == "crosses_below":
                result = (macd_prev >= signal_prev) and (macd_val < signal_val)
                print(f"Crosses below evaluation: {result}")
                return result
                
        elif self.indicator == "BBANDS" and self.value == "lower":
            price = row.get('close')
            lower = row.get('lowerband')  # lower band
            price_prev = row.get('close_prev')
            lower_prev = row.get('lowerband_prev')
            
            print(f"\nBBands check:")
            print(f"Current price: {price}")
            print(f"Current lower band: {lower}")
            print(f"Previous price: {price_prev}")
            print(f"Previous lower band: {lower_prev}")
            
            if any(pd.isna([price, lower, price_prev, lower_prev])):
                print("Missing or NaN values detected")
                return False
                
            if self.comparison == "crosses_below":
                result = price_prev >= lower_prev and price < lower
                print(f"Crosses below evaluation: {result}")
                return result
            elif self.comparison == "crosses_above":
                result = price_prev <= lower_prev and price > lower
                print(f"Crosses above evaluation: {result}")
                return result
                
        else:
            # Standard indicator comparison
            indicator_value = row.get(self.indicator.lower())
            if pd.isna(indicator_value):
                return False

            if isinstance(self.value, (int, float)):
                compare_value = self.value
            else:
                compare_value = row.get(self.value.lower())
                if pd.isna(compare_value):
                    return False

            if self.comparison == "above":
                print("indicator value: ", indicator_value, " / compare value: ", compare_value)
                return indicator_value > compare_value
            elif self.comparison == "below":
                print("indicator value: ", indicator_value, " / compare value: ", compare_value)
                return indicator_value < compare_value
            elif self.comparison == "between":
                print("indicator value: ", indicator_value, " between compare values: ", compare_value)
                # compare values found in columns
                if any(isinstance(x, str) for x in self.value):
                    return row[self.indicator.lower()].between(row[compare_value[0]], row[compare_value[1]])
                # compare numerical val
                if any(isinstance(x, (int, float)) for x in self.value):
                    return row[self.indicator.lower()].between(compare_value[0], compare_value[1])

        return False   