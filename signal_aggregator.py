import pandas as pd
from typing import List, Dict, Any
from indicators import MovingAverages, RSI, Stochastic, MACD, BollingerBands, VWAP, ATR

class SignalAggregator:
    def __init__(self):
        self.indicators = {}
        self.min_profit_threshold = 0.02  # 2% minimum profit target to overcome spread
        
    def analyze(self, df, current_spread_pct=None):
        """
        Analyze market data considering spread costs
        """
        if current_spread_pct is None:
            current_spread_pct = df['spread'].iloc[-1] if 'spread' in df.columns else 0
            
        # Calculate required price movement to overcome spread
        required_movement = current_spread_pct + self.min_profit_threshold
        
        # Get base signals from indicators
        signals = self._get_base_signals(df)
        
        # Adjust signal strengths based on spread
        adjusted_signals = self._adjust_signals_for_spread(signals, current_spread_pct)
        
        # Calculate sentiment scores
        buy_sentiment, sell_sentiment = self._calculate_sentiment(adjusted_signals)
        
        # Determine recommendation considering spread
        recommendation = self._get_recommendation(
            buy_sentiment, 
            sell_sentiment, 
            required_movement
        )
        
        return {
            "recommendation": recommendation,
            "buy_sentiment": buy_sentiment,
            "sell_sentiment": sell_sentiment,
            "signals": adjusted_signals,
            "indicator_values": self._get_indicator_values(df),
            "spread_analysis": {
                "current_spread": current_spread_pct,
                "required_movement": required_movement,
                "is_spread_favorable": current_spread_pct < 0.8  # Consider spread favorable if < 0.8%
            }
        }
    
    def _adjust_signals_for_spread(self, signals, spread_pct):
        """
        Adjust signal strengths based on current spread with risk-aware bias
        """
        adjusted = {"buy": [], "sell": []}
        
        # Risk-aware spread adjustment
        if spread_pct < 0.5:  # Very tight spread
            strength_multiplier = 1.2
        elif spread_pct < 1.0:  # Normal spread
            strength_multiplier = 1.0
        else:  # Wide spread
            strength_multiplier = 0.6  # Reduced multiplier for high spreads
            
        # Adjust buy/sell sentiment with conservative bias
        signals['buy_sentiment'] *= strength_multiplier
        signals['sell_sentiment'] *= (strength_multiplier * 0.8)  # More conservative for sells
        
        for signal in signals["buy"]:
            adj_signal = signal.copy()
            
            # Progressive strength reduction based on spread
            if spread_pct >= 1.0:
                adj_signal["strength"] = "Weak"
                adj_signal["reason"] += f" (High spread: {spread_pct:.2f}%)"
            elif spread_pct >= 0.5:
                if signal["strength"] == "Strong":
                    adj_signal["strength"] = "Medium"
                elif signal["strength"] == "Medium":
                    adj_signal["strength"] = "Weak"
            
            adjusted["buy"].append(adj_signal)
        
        # Conservative sell signal adjustment
        for signal in signals["sell"]:
            adj_signal = signal.copy()
            if spread_pct >= 1.0:
                # Only maintain strong sell signals with high confidence
                if signal["strength"] == "Strong" and signal.get("confidence", 0) > 0.8:
                    adj_signal["strength"] = "Strong"
                else:
                    adj_signal["strength"] = "Medium"
            adjusted["sell"].append(adj_signal)
        
        return adjusted
    
    def _get_recommendation(self, buy_sentiment, sell_sentiment, required_movement):
        """
        Get trading recommendation with balanced risk management
        """
        # Conservative thresholds for better risk management
        BUY_THRESHOLD = 60  # Maintained for entries
        SELL_THRESHOLD = 95  # Higher threshold for exits
        
        if buy_sentiment > BUY_THRESHOLD:
            return "BUY"
        elif sell_sentiment > SELL_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"
    
    def get_risk_assessment(self, df):
        """
        Enhanced risk assessment including spread analysis
        """
        volatility = df['price'].pct_change().std() * 100
        atr = self._calculate_atr(df)
        current_spread = df['spread'].iloc[-1] if 'spread' in df.columns else 0
        
        # Determine risk level considering spread
        risk_level = "High" if current_spread >= 1.0 or volatility >= 2.0 else \
                    "Medium" if current_spread >= 0.5 or volatility >= 1.0 else \
                    "Low"
        
        return {
            "risk_level": risk_level,
            "volatility": volatility,
            "atr": atr,
            "spread_risk": {
                "current_spread": current_spread,
                "spread_impact": "High" if current_spread >= 1.0 else \
                               "Medium" if current_spread >= 0.5 else \
                               "Low"
            }
        }
    
    def get_trade_parameters(self, df, max_position_size):
        """
        Calculate trade parameters considering spread
        """
        current_price = df['price'].iloc[-1]
        current_spread = df['spread'].iloc[-1] if 'spread' in df.columns else 0
        volatility = df['price'].pct_change().std() * 100
        
        # Adjust position size based on spread
        spread_factor = max(0.2, 1 - (current_spread / 0.5))  # Reduce position size as spread increases
        suggested_position = max_position_size * spread_factor
        
        # Calculate stop loss and take profit considering spread
        stop_loss_pct = max(2.0, volatility * 2)  # Minimum 2% stop loss
        take_profit_pct = max(current_spread * 3, stop_loss_pct * 2)  # Ensure profit target overcomes spread
        
        return {
            "suggested_position_size": suggested_position,
            "stop_loss": current_price * (1 - stop_loss_pct/100),
            "take_profit": current_price * (1 + take_profit_pct/100),
            "min_profit_target_pct": take_profit_pct,
            "spread_adjustment": {
                "spread_factor": spread_factor,
                "position_size_reduction": f"{(1-spread_factor)*100:.1f}%"
            }
        }
    
    def _get_base_signals(self, df):
        """
        Get base signals from indicators
        """
        # Initialize all indicators
        indicators = [
            MovingAverages(),
            RSI(),
            Stochastic(),
            MACD(),
            BollingerBands(),
            VWAP(),
            ATR()
        ]
        
        all_signals = []
        indicator_values = {}
        
        # Get signals from each indicator
        for indicator in indicators:
            result = indicator.generate_signals(df)
            all_signals.extend([{**signal, "indicator": indicator.name} for signal in result["signals"]])
            indicator_values[indicator.name] = result["values"]
        
        # Count and weigh signals
        buy_signals = []
        sell_signals = []
        neutral_signals = []
        
        strength_weights = {
            "Strong": 3,
            "Medium": 2,
            "Weak": 1
        }
        
        weighted_buy_count = 0
        weighted_sell_count = 0
        
        for signal in all_signals:
            weight = strength_weights[signal["strength"]]
            if signal["signal"] == "BUY":
                weighted_buy_count += weight
                buy_signals.append(signal)
            elif signal["signal"] == "SELL":
                weighted_sell_count += weight
                sell_signals.append(signal)
            else:
                neutral_signals.append(signal)
        
        # Calculate overall sentiment
        total_weight = weighted_buy_count + weighted_sell_count
        if total_weight > 0:
            buy_sentiment = (weighted_buy_count / total_weight) * 100
            sell_sentiment = (weighted_sell_count / total_weight) * 100
        else:
            buy_sentiment = sell_sentiment = 0
        
        return {
            "buy": buy_signals,
            "sell": sell_signals,
            "neutral": neutral_signals,
            "buy_sentiment": buy_sentiment,
            "sell_sentiment": sell_sentiment
        }
    
    def _calculate_sentiment(self, signals):
        """
        Calculate sentiment scores
        """
        buy_sentiment = len(signals["buy"]) * 50  # More weight to buy signals
        sell_sentiment = len(signals["sell"]) * 40  # Less weight to sell signals
        
        return buy_sentiment, sell_sentiment
    
    def _get_indicator_values(self, df):
        """
        Get indicator values
        """
        # Initialize all indicators
        indicators = [
            MovingAverages(),
            RSI(),
            Stochastic(),
            MACD(),
            BollingerBands(),
            VWAP(),
            ATR()
        ]
        
        indicator_values = {}
        
        # Get values from each indicator
        for indicator in indicators:
            result = indicator.generate_signals(df)
            indicator_values[indicator.name] = result["values"]
        
        return indicator_values
    
    def _calculate_atr(self, df):
        """
        Calculate Average True Range (ATR)
        """
        atr = ATR()
        return atr.calculate(df)['atr']
