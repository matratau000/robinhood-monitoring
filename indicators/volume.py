import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_indicator import BaseIndicator

class VWAP(BaseIndicator):
    def __init__(self):
        super().__init__("VWAP")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        
        # If volume data is available
        if 'volume' in df.columns:
            df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            # If no volume data, use a simple moving average as approximation
            df['vwap'] = df['price'].rolling(window=20).mean()
        
        return {
            'vwap': df['vwap'].iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        current_price = data['price'].iloc[-1]
        signals = []
        
        if current_price > calc['vwap']:
            signals.append({"signal": "BUY", "reason": "Price above VWAP", "strength": "Medium"})
        elif current_price < calc['vwap']:
            signals.append({"signal": "SELL", "reason": "Price below VWAP", "strength": "Medium"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }

class ATR(BaseIndicator):
    def __init__(self, period: int = 14):
        super().__init__("ATR")
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        df = data.copy()
        high = df['price'].rolling(2).max()
        low = df['price'].rolling(2).min()
        
        tr1 = high - low
        tr2 = abs(high - df['price'].shift())
        tr3 = abs(low - df['price'].shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        return {
            'atr': atr.iloc[-1]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        calc = self.calculate(data)
        price_std = data['price'].std()
        signals = []
        
        # High volatility warning
        if calc['atr'] > price_std * 2:
            signals.append({"signal": "NEUTRAL", "reason": "High volatility - use wider stops", "strength": "Strong"})
        elif calc['atr'] < price_std * 0.5:
            signals.append({"signal": "NEUTRAL", "reason": "Low volatility - potential breakout incoming", "strength": "Medium"})
        
        return {
            "indicator": self.name,
            "values": calc,
            "signals": signals
        }
