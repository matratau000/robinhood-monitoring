import os
import base64
import datetime
import json
from typing import Any, Dict, Optional, List
import uuid
import requests
from nacl.signing import SigningKey
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timezone, timedelta
from signal_aggregator import SignalAggregator

# Load environment variables
load_dotenv()

class RobinhoodBTCAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('ROBINHOOD_API_KEY')
        private_key_seed = base64.b64decode(os.getenv('ROBINHOOD_PRIVATE_KEY'))
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"
        self.symbol = "BTC-USD"
        self.signal_aggregator = SignalAggregator()

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp())

    def get_authorization_header(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = self._get_current_timestamp()
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        headers = self.get_authorization_header(method, path, body)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def get_btc_price(self) -> Dict:
        """Get current BTC best bid and ask prices"""
        path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={self.symbol}"
        return self.make_api_request("GET", path)

    def get_estimated_price(self, side: str, quantity: str) -> Dict:
        """Get estimated price for a specific quantity"""
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={self.symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def get_btc_holdings(self) -> Dict:
        """Get current BTC holdings"""
        path = "/api/v1/crypto/trading/holdings/?asset_code=BTC"
        return self.make_api_request("GET", path)

    def get_account_info(self) -> Dict:
        """Get account information"""
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def analyze_market(self, time_period_minutes: int = 15) -> Dict[str, Any]:
        """
        Comprehensive market analysis using multiple technical indicators
        """
        prices = []
        timestamps = []
        volumes = []  # For tracking trading volume if available
        spreads = []  # For tracking bid-ask spreads
        start_time = datetime.now()
        
        print(f"\nStarting market analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Collecting data for {time_period_minutes} minutes...")
        print("\nReal-time Data:")
        print(f"{'Time':^20} | {'Price':^12} | {'Spread':^10} | {'Change %':^8}")
        print("-" * 55)
        
        last_price = None
        
        while (datetime.now() - start_time).total_seconds() < time_period_minutes * 60:
            try:
                price_data = self.get_btc_price()
                
                if price_data and 'results' in price_data and price_data['results']:
                    result = price_data['results'][0]
                    
                    # Extract and validate price data
                    ask_price = float(result['ask_inclusive_of_buy_spread'])
                    bid_price = float(result['bid_inclusive_of_sell_spread'])
                    mid_price = (ask_price + bid_price) / 2
                    spread = ask_price - bid_price
                    spread_pct = (spread / mid_price) * 100
                    
                    # Calculate price change
                    if last_price:
                        price_change_pct = ((mid_price - last_price) / last_price) * 100
                    else:
                        price_change_pct = 0.0
                    
                    # Store data
                    prices.append(mid_price)
                    timestamps.append(datetime.now())
                    spreads.append(spread_pct)
                    last_price = mid_price
                    
                    # Print real-time data
                    print(f"{datetime.now().strftime('%H:%M:%S'):^20} | "
                          f"${mid_price:,.2f} | "
                          f"{spread_pct:,.3f}% | "
                          f"{price_change_pct:+.2f}%")
                    
                    # Alert on significant changes
                    if abs(price_change_pct) > 0.5:  # Alert on 0.5% or greater price moves
                        print(f"\n🚨 ALERT: Significant price movement detected: {price_change_pct:+.2f}%")
                    
                    if spread_pct > 1.0:  # Alert on wide spreads
                        print(f"\n⚠️  Warning: Wide spread detected: {spread_pct:.2f}%")
                
                else:
                    print(f"\n❌ Error: Invalid price data received at {datetime.now().strftime('%H:%M:%S')}")
                    if price_data:
                        print(f"Data: {json.dumps(price_data, indent=2)}")
            
            except Exception as e:
                print(f"\n❌ Error fetching price data: {str(e)}")
            
            # Sleep for 30 seconds between checks
            time.sleep(30)
        
        if not prices:
            return {"error": "No price data collected"}
        
        # Create DataFrame with all collected data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'spread': spreads
        })
        df.set_index('timestamp', inplace=True)
        
        # Print data collection summary
        print("\nData Collection Summary:")
        print(f"Total data points: {len(prices)}")
        print(f"Average spread: {np.mean(spreads):.3f}%")
        print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
        
        # Get comprehensive analysis from SignalAggregator
        analysis = self.signal_aggregator.analyze(df)
        
        # Get risk assessment and trade parameters
        risk_assessment = self.signal_aggregator.get_risk_assessment(df)
        
        # Get trade parameters (using 10% of available buying power as max position size)
        account_info = self.get_account_info()
        if account_info:
            max_position = float(account_info.get('buying_power', '0')) * 0.1
            trade_params = self.signal_aggregator.get_trade_parameters(df, max_position)
        else:
            trade_params = None
        
        return {
            "market_analysis": analysis,
            "risk_assessment": risk_assessment,
            "trade_parameters": trade_params,
            "data_points": len(prices),
            "time_period_minutes": time_period_minutes,
            "avg_spread": np.mean(spreads),
            "price_volatility": np.std(prices) / np.mean(prices) * 100
        }

def format_price(price: float) -> str:
    return f"${price:,.2f}"

def main():
    analyzer = RobinhoodBTCAnalyzer()
    
    try:
        while True:
            print("\n" + "="*80)
            print(f"Starting new analysis cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            
            # Get account information
            account_info = analyzer.get_account_info()
            print("\nAccount Information:")
            print(json.dumps(account_info, indent=2))
            
            # Get current BTC holdings
            holdings = analyzer.get_btc_holdings()
            print("\nBTC Holdings:")
            print(json.dumps(holdings, indent=2))
            
            # Get current BTC price
            price_data = analyzer.get_btc_price()
            print("\nCurrent BTC Price:")
            print(json.dumps(price_data, indent=2))
            
            # Get estimated price for buying 0.1 BTC
            estimated_price = analyzer.get_estimated_price("ask", "0.1")
            print("\nEstimated Price for buying 0.1 BTC:")
            print(json.dumps(estimated_price, indent=2))
            
            # Perform comprehensive market analysis
            print("\nPerforming comprehensive market analysis...")
            analysis = analyzer.analyze_market(15)  # 15-minute analysis
            
            print("\n=== MARKET ANALYSIS RESULTS ===")
            print(f"\nOverall Recommendation: {analysis['market_analysis']['recommendation']}")
            print(f"Buy Sentiment: {analysis['market_analysis']['buy_sentiment']:.1f}%")
            print(f"Sell Sentiment: {analysis['market_analysis']['sell_sentiment']:.1f}%")
            
            print("\n=== INDICATOR VALUES ===")
            for indicator, values in analysis['market_analysis']['indicator_values'].items():
                print(f"\n{indicator}:")
                for key, value in values.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            
            print("\n=== TRADING SIGNALS ===")
            print("\nBuy Signals:")
            for signal in analysis['market_analysis']['signals']['buy']:
                print(f"- {signal['indicator']}: {signal['reason']} ({signal['strength']})")
            
            print("\nSell Signals:")
            for signal in analysis['market_analysis']['signals']['sell']:
                print(f"- {signal['indicator']}: {signal['reason']} ({signal['strength']})")
            
            print("\n=== RISK ASSESSMENT ===")
            print(f"Risk Level: {analysis['risk_assessment']['risk_level']}")
            print(f"Volatility: {analysis['risk_assessment']['volatility']:.2f}%")
            print(f"ATR: {analysis['risk_assessment']['atr']:.2f}")
            print(f"Average Spread: {analysis['avg_spread']:.3f}%")
            print(f"Price Volatility: {analysis['price_volatility']:.2f}%")
            
            if analysis['trade_parameters']:
                print("\n=== TRADE PARAMETERS ===")
                print(f"Suggested Position Size: {format_price(analysis['trade_parameters']['suggested_position_size'])}")
                print(f"Stop Loss: {format_price(analysis['trade_parameters']['stop_loss'])}")
                print(f"Take Profit: {format_price(analysis['trade_parameters']['take_profit'])}")
            
            print("\nWaiting 1 minute before starting next analysis cycle...")
            time.sleep(60)  # Wait 1 minute before starting next cycle
            
    except KeyboardInterrupt:
        print("\n\nAnalysis stopped by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
