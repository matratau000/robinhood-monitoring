import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_indicator import BaseIndicator

class MovingAverages(BaseIndicator):
    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__("Moving Averages")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        df['fast_ema'] = df['price'].ewm(span=self.fast_period, adjust=False).mean()
        df['slow_ema'] = df['price'].ewm(span=self.slow_period, adjust=False).mean()
        df['sma'] = df['price'].rolling(window=self.slow_period).mean()
        
        return {
            'fast_ema': df['fast_ema'].iloc[-1],
            'slow_ema': df['slow_ema'].iloc[-1],
            'sma': df['sma'].iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        current_price = data['price'].iloc[-1]
        
        signals = []
        strength = "Medium"
        
        # EMA Crossover
        if calc['fast_ema'] > calc['slow_ema']:
            signals.append({"signal": "BUY", "reason": "Fast EMA above Slow EMA", "strength": strength})
        elif calc['fast_ema'] < calc['slow_ema']:
            signals.append({"signal": "SELL", "reason": "Fast EMA below Slow EMA", "strength": strength})
        
        # Price vs SMA
        if current_price > calc['sma']:
            signals.append({"signal": "BUY", "reason": "Price above SMA", "strength": "Weak"})
        elif current_price < calc['sma']:
            signals.append({"signal": "SELL", "reason": "Price below SMA", "strength": "Weak"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }
