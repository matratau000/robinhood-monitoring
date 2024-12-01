import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_indicator import BaseIndicator

class MACD(BaseIndicator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        exp1 = df['price'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = df['price'].ewm(span=self.slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        signals = []
        
        if calc['macd'] > calc['signal']:
            signals.append({"signal": "BUY", "reason": "MACD above Signal line", "strength": "Medium"})
        elif calc['macd'] < calc['signal']:
            signals.append({"signal": "SELL", "reason": "MACD below Signal line", "strength": "Medium"})
        
        # Histogram momentum
        if calc['histogram'] > 0 and calc['histogram'] > data['price'].std():
            signals.append({"signal": "BUY", "reason": "Strong positive MACD momentum", "strength": "Strong"})
        elif calc['histogram'] < 0 and abs(calc['histogram']) > data['price'].std():
            signals.append({"signal": "SELL", "reason": "Strong negative MACD momentum", "strength": "Strong"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }

class BollingerBands(BaseIndicator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        sma = df['price'].rolling(window=self.period).mean()
        std = df['price'].rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        return {
            'sma': sma.iloc[-1],
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        current_price = data['price'].iloc[-1]
        signals = []
        
        if current_price > calc['upper_band']:
            signals.append({"signal": "SELL", "reason": "Price above Upper Bollinger Band", "strength": "Strong"})
        elif current_price < calc['lower_band']:
            signals.append({"signal": "BUY", "reason": "Price below Lower Bollinger Band", "strength": "Strong"})
        
        # Mean reversion potential
        if abs(current_price - calc['sma']) > data['price'].std() * 1.5:
            if current_price > calc['sma']:
                signals.append({"signal": "SELL", "reason": "Strong mean reversion potential (high)", "strength": "Medium"})
            else:
                signals.append({"signal": "BUY", "reason": "Strong mean reversion potential (low)", "strength": "Medium"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }
