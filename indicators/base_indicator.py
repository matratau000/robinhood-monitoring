from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseIndicator(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate the indicator values"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on the indicator"""
        pass
