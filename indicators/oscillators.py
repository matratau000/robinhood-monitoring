import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_indicator import BaseIndicator

class RSI(BaseIndicator):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {'rsi': rsi.iloc[-1]}
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        signals = []
        
        if calc['rsi'] > self.overbought:
            signals.append({"signal": "SELL", "reason": "RSI Overbought", "strength": "Strong"})
        elif calc['rsi'] < self.oversold:
            signals.append({"signal": "BUY", "reason": "RSI Oversold", "strength": "Strong"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }

class Stochastic(BaseIndicator):
    def __init__(self, k_period: int = 14, d_period: int = 3, overbought: float = 80, oversold: float = 20):
        super().__init__("Stochastic")
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        low_min = df['price'].rolling(window=self.k_period).min()
        high_max = df['price'].rolling(window=self.k_period).max()
        
        k_line = 100 * ((df['price'] - low_min) / (high_max - low_min))
        d_line = k_line.rolling(window=self.d_period).mean()
        
        return {
            'k_line': k_line.iloc[-1],
            'd_line': d_line.iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        signals = []
        
        if calc['k_line'] > self.overbought and calc['d_line'] > self.overbought:
            signals.append({"signal": "SELL", "reason": "Stochastic Overbought", "strength": "Medium"})
        elif calc['k_line'] < self.oversold and calc['d_line'] < self.oversold:
            signals.append({"signal": "BUY", "reason": "Stochastic Oversold", "strength": "Medium"})
        
        # Crossover signals
        if calc['k_line'] > calc['d_line']:
            signals.append({"signal": "BUY", "reason": "Stochastic K-line crossed above D-line", "strength": "Weak"})
        elif calc['k_line'] < calc['d_line']:
            signals.append({"signal": "SELL", "reason": "Stochastic K-line crossed below D-line", "strength": "Weak"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }
