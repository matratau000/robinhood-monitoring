import os
import time
import json
import uuid
import base64
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from nacl.signing import SigningKey
from dotenv import load_dotenv
from signal_aggregator import SignalAggregator
from tabulate import tabulate
from rh_helpers.rh_entry_price import RHEntryPriceTracker

class CryptoAnalyzer:
    def __init__(self):
        load_dotenv()
        self.api_url = "https://trading.robinhood.com"
        self.api_key = os.getenv('ROBINHOOD_API_KEY')
        self.private_key = base64.b64decode(os.getenv('ROBINHOOD_PRIVATE_KEY'))
        self.signing_key = SigningKey(self.private_key)
        self.signal_aggregator = SignalAggregator()
        
        # Initialize price tracker
        self.price_tracker = RHEntryPriceTracker(
            api_url=self.api_url,
            get_auth_header_func=self.get_authorization_header
        )
        
        # Trading parameters - adjusted for crypto volatility
        self.min_profit_threshold = 0.02  # 2% minimum profit target
        self.position_size = 100  # Position size in USD
        self.max_spread = 0.015  # Maximum acceptable spread 1.5%
        self.max_positions = 5  # Maximum number of open positions
        self.stop_loss_pct = 0.10  # 10% stop loss for crypto volatility
        self.take_profit_pct = 0.15  # 15% take profit target
        self.min_volume_threshold = 1000000  # Minimum volume for valid signals
        self.volatility_window = 24  # Hours for volatility calculation
        self.min_profit_pct = 0.0135  # 1.35% minimum profit before considering exit
        
        # Initialize database
        self.init_database()
        
        # Reset any incorrect stop losses
        self.reset_stop_losses()
        
        # Load active trades
        self.active_trades = self.get_open_positions()
        
        # Store initial reference price
        self.reference_price = None  
        self.last_price = None
        
    def init_database(self):
        """Initialize SQLite database for trade tracking"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        # Create trades table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                    (id TEXT PRIMARY KEY,
                     symbol TEXT,
                     side TEXT,
                     entry_price REAL,
                     exit_price REAL,
                     quantity REAL,
                     entry_time TIMESTAMP,
                     exit_time TIMESTAMP,
                     status TEXT,
                     pnl REAL,
                     pnl_percentage REAL,
                     stop_loss REAL,
                     take_profit REAL)''')
        
        # Create signals table for tracking prediction accuracy
        c.execute('''CREATE TABLE IF NOT EXISTS signals
                    (id TEXT PRIMARY KEY,
                     timestamp TIMESTAMP,
                     symbol TEXT,
                     signal TEXT,
                     price REAL,
                     spread REAL,
                     result TEXT,
                     correct INTEGER)''')
        
        conn.commit()
        conn.close()

    def check_stop_loss_take_profit(self, current_price: float):
        """Check and execute stop loss/take profit orders"""
        if current_price is None:
            print("❌ Error: Cannot check stop loss/take profit - current price is None")
            return
            
        positions = self.get_open_positions()
        for pos in positions:
            trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
            
            try:
                entry_price = float(entry_price)
                stop_loss = float(stop_loss)
                take_profit = float(take_profit)
                current_price = float(current_price)
            except (TypeError, ValueError) as e:
                print(f"❌ Error converting prices to float: {e}")
                continue
                
            if side == 'buy':
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Check stop loss
                if current_price <= stop_loss:
                    print(f"\n🔴 Stop Loss triggered for {symbol} @ {current_price:.8f}")
                    if self.close_position(trade_id, current_price, "stop_loss"):
                        print(f"✅ Position closed successfully")
                    else:
                        print(f"❌ Failed to close position")
                
                # Check take profit
                elif current_price >= take_profit:
                    print(f"\n🟢 Take Profit triggered for {symbol} @ {current_price:.8f}")
                    if self.close_position(trade_id, current_price, "take_profit"):
                        print(f"✅ Position closed successfully")
                    else:
                        print(f"❌ Failed to close position")
                    
                # Update position status
                print(f"Position P&L: {pnl_pct:.2f}%")

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Place a real order on Robinhood"""
        path = "/api/v1/crypto/trading/orders/"
        client_order_id = str(uuid.uuid4())
        trade_id = str(uuid.uuid4())  # Generate unique trade ID
        
        # Ensure symbol has -USD suffix
        if not symbol.endswith('-USD'):
            symbol = f"{symbol}-USD"
            
        # Format quantity as string with proper precision
        if symbol == "SHIB-USD":
            quantity = int(quantity)  # Convert to integer for SHIB
            quantity_str = str(quantity)  # No decimal places for SHIB
        else:
            quantity_str = f"{quantity:.8f}"  # Use 8 decimal places for other cryptos
        
        # Construct order payload according to API docs
        order_config = {
            "asset_quantity": quantity_str
        }
        
        body = {
            "client_order_id": client_order_id,
            "side": side.lower(),  # API expects lowercase
            "type": "market",
            "symbol": symbol,  # Use full symbol with -USD
            "market_order_config": order_config
        }
        
        print(f"\n📤 Sending order to Robinhood:")
        print(f"Symbol: {symbol}")
        print(f"Side: {side.upper()}")
        print(f"Quantity: {quantity_str}")
        
        headers = self.get_authorization_header("POST", path, json.dumps(body))
        headers['Content-Type'] = 'application/json'  # Required by API
        
        try:
            response = requests.post(
                f"{self.api_url}{path}",
                headers=headers,
                json=body,
                timeout=10
            )
            
            if response.status_code == 201:
                response_data = response.json()
                print("✅ Order placed successfully!")
                print(f"Order ID: {response_data.get('id')}")
                
                if side == 'buy':
                    # Get current price for initial estimate
                    price_data = self.get_shib_price()
                    if price_data:
                        estimated_price = price_data['ask']  # Use ask price for buys
                        
                        # Store initial trade details with price tracker
                        self.price_tracker.store_initial_trade(
                            trade_id=trade_id,
                            symbol=symbol,
                            side=side,
                            estimated_price=estimated_price,
                            quantity=float(quantity_str),
                            order_id=response_data.get('id'),
                            stop_loss_pct=self.stop_loss_pct,
                            take_profit_pct=self.take_profit_pct
                        )
                        
                        # Update with actual execution price
                        self.price_tracker.update_trade_with_actual_price(trade_id)
                    
                    # Also record in existing database for compatibility
                    self.record_trade(response_data, side)
                    
                return response_data
            else:
                try:
                    response_data = response.json()
                    error_type = response_data.get('type')
                    errors = response_data.get('errors', [])
                    error_details = '; '.join(error['detail'] for error in errors) if errors else response.text
                    print(f"❌ Order failed: {error_type} - {error_details}")
                except:
                    print(f"❌ Order failed: Status {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error placing order: {str(e)}")
            return None

    def record_trade(self, order_data: dict, side: str):
        """Record trade in database with stop loss and take profit"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        trade_id = order_data['id']
        symbol = order_data['symbol']  # Keep full symbol with -USD
        quantity = float(order_data['market_order_config']['asset_quantity'])
        
        # Get price from executions if available, otherwise use current market price
        try:
            if order_data.get('executions') and len(order_data['executions']) > 0:
                execution = order_data['executions'][0]
                if execution.get('price'):
                    price = float(execution['price'])
                else:
                    raise ValueError("No price in execution data")
            else:
                # Use current market price as fallback
                current_price = self.get_shib_price()
                if current_price and current_price.get('price'):
                    price = float(current_price['price'])
                else:
                    raise ValueError("Could not determine current market price")
                
            if price <= 0:
                raise ValueError("Invalid price value")
                
        except (TypeError, ValueError) as e:
            print(f"⚠️ Warning: Could not determine execution price - {str(e)}")
            conn.close()
            return
        
        # Calculate stop loss and take profit with 10% stop loss and 15% take profit
        stop_loss = round(price * (1 - self.stop_loss_pct), 8)  # 10% below entry
        take_profit = round(price * (1 + self.take_profit_pct), 8)  # 15% above entry
        
        try:
            c.execute('''INSERT INTO trades 
                        (id, symbol, side, entry_price, quantity, entry_time, status,
                         stop_loss, take_profit)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (trade_id, symbol, side, price, quantity, 
                         datetime.now(), 'OPEN', stop_loss, take_profit))
            
            conn.commit()
            
            print(f"\n✅ New {side.upper()} position opened:")
            print(f"Symbol: {symbol}")
            print(f"Quantity: {quantity:.8f}")
            print(f"Entry Price: ${price:.8f}")
            print(f"Stop Loss: ${stop_loss:.8f}")
            print(f"Take Profit: ${take_profit:.8f}")
            
        except sqlite3.Error as e:
            print(f"❌ Database error: {str(e)}")
        finally:
            conn.close()

    def close_position(self, trade_id: str, exit_price: float, reason: str = "manual"):
        """Close a position and update the database"""
        try:
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            print(f"❌ Error: Invalid exit price {exit_price}")
            return False
            
        if exit_price <= 0:
            print("❌ Error: Exit price must be greater than 0")
            return False
            
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        try:
            # Get trade details
            c.execute('''SELECT id, symbol, side, entry_price, quantity, entry_time, status, 
                        stop_loss, take_profit 
                        FROM trades 
                        WHERE id = ?''', (trade_id,))
            trade = c.fetchone()
            
            if not trade:
                print(f"❌ Error: Trade {trade_id} not found")
                conn.close()
                return False
            
            # Debug logging
            print(f"\nTrade details:")
            print(f"ID: {trade[0]}")
            print(f"Symbol: {trade[1]}")
            print(f"Side: {trade[2]}")
            print(f"Entry Price: {trade[3]}")
            print(f"Quantity: {trade[4]}")
                
            if trade[3] is None or trade[4] is None:
                print("❌ Error: Missing entry price or quantity")
                conn.close()
                return False
                
            try:
                entry_price = float(trade[3])
                quantity = float(trade[4])
                
                # Execute sell order through Robinhood API
                if trade[2] == 'buy':  # Only close long positions with sell orders
                    sell_order = self.place_order(
                        symbol=trade[1],
                        side='sell',
                        quantity=quantity
                    )
                    
                    if not sell_order:
                        print("❌ Failed to execute sell order through Robinhood")
                        conn.close()
                        return False
                    
                    # Wait for order confirmation (you may want to implement proper order status checking)
                    time.sleep(2)
                
                # Calculate P&L
                if trade[2] == 'buy':
                    pnl = (exit_price - entry_price) * quantity
                    pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl = (entry_price - exit_price) * quantity
                    pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
                
                # Update trade record
                c.execute('''UPDATE trades 
                           SET status = 'CLOSED',
                               exit_price = ?,
                               exit_time = ?,
                               pnl = ?,
                               pnl_percentage = ?
                           WHERE id = ?''', 
                           (exit_price, datetime.now(), pnl, pnl_percentage, trade_id))
                
                conn.commit()
                print(f"\n✅ Position closed:")
                print(f"Symbol: {trade[1]}")
                print(f"Entry: ${entry_price:.8f}")
                print(f"Exit: ${exit_price:.8f}")
                print(f"P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
                print(f"Reason: {reason}")
                return True
                
            except (TypeError, ValueError) as e:
                print(f"❌ Error processing trade values: {e}")
                print(f"Entry price type: {type(trade[3])}, value: {trade[3]}")
                print(f"Quantity type: {type(trade[4])}, value: {trade[4]}")
                conn.close()
                return False
                
        except sqlite3.Error as e:
            print(f"❌ Database error: {str(e)}")
            return False
        finally:
            conn.close()

    def display_trading_dashboard(self):
        """Display current trading status with enhanced metrics"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        # Get open positions with stop loss and take profit
        open_positions = self.get_open_positions()
        
        # Format open positions for display
        formatted_positions = []
        for pos in open_positions:
            trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
            formatted_positions.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"{quantity:.0f}",
                f"${format(float(stop_loss), '.8f')}",
                f"${format(float(take_profit), '.8f')}",
                entry_time
            ])
        
        # Get recent closed trades
        c.execute('''SELECT symbol, side, entry_price, exit_price, pnl_percentage,
                           entry_time, exit_time 
                    FROM trades WHERE status = "CLOSED" 
                    ORDER BY exit_time DESC LIMIT 5''')
        recent_trades = c.fetchall()
        
        # Format recent trades for display
        formatted_trades = []
        for trade in recent_trades:
            symbol, side, entry_price, exit_price, pnl_pct, entry_time, exit_time = trade
            formatted_trades.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"${format(float(exit_price), '.8f')}",
                f"{pnl_pct:+.2f}%",
                entry_time,
                exit_time
            ])
        
        # Calculate trading metrics
        c.execute('SELECT COUNT(*), SUM(pnl), AVG(pnl_percentage) FROM trades WHERE status = "CLOSED"')
        total_trades, total_pnl, avg_pnl = c.fetchone()
        
        # Handle null values
        total_trades = total_trades or 0
        total_pnl = total_pnl or 0.0
        avg_pnl = avg_pnl or 0.0
        
        c.execute('SELECT COUNT(*) FROM trades WHERE status = "CLOSED" AND pnl > 0')
        winning_trades = c.fetchone()[0] or 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n📊 === Trading Dashboard ===")
        
        print("\n💼 Open Positions:")
        if formatted_positions:
            headers = ['Symbol', 'Side', 'Entry $', 'Qty', 'Stop Loss', 'Take Profit', 'Time']
            print(tabulate(formatted_positions, headers=headers, tablefmt='grid'))
        else:
            print("No open positions")
            
        print("\n📈 Recent Trades:")
        if formatted_trades:
            headers = ['Symbol', 'Side', 'Entry $', 'Exit $', 'P&L %', 'Entry Time', 'Exit Time']
            print(tabulate(formatted_trades, headers=headers, tablefmt='grid'))
        else:
            print("No recent trades")
            
        print("\n📉 Trading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average P&L: {avg_pnl:.2f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        
        conn.close()

    def analyze_market(self, symbol: str = "SHIB-USD", time_period_minutes: int = 1):
        """Analyze market and execute trades with enhanced risk management"""
        
        # Get current open positions count
        open_positions = self.get_open_positions()
        num_positions = len(open_positions)
        
        # Check if we've hit position limit
        if num_positions >= self.max_positions:
            print(f"\n⚠️ Maximum positions ({self.max_positions}) reached. Monitoring only.")
            
            # Create DataFrame for analysis
            prices = []
            timestamps = []
            spreads = []
            
            try:
                # Collect some price data for analysis
                for _ in range(6):  # 30 seconds of data
                    price_data = self.get_shib_price()
                    if price_data:
                        mid_price = price_data['price']
                        spread = price_data['spread']
                        spread_pct = (spread / mid_price) * 100
                        
                        prices.append(mid_price)
                        timestamps.append(datetime.now())
                        spreads.append(spread_pct)
                    time.sleep(5)
                
                # Create analysis DataFrame
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'price': prices,
                    'spread': spreads
                })
                df.set_index('timestamp', inplace=True)
                
                # Get trading signals and risk assessment
                signals = self.signal_aggregator.analyze(df, np.mean(spreads))
                risk = self.signal_aggregator.get_risk_assessment(df)
                
                # Monitor and potentially exit positions based on signals
                current_price = prices[-1]
                avg_spread = np.mean(spreads)
                
                print(f"\n🔍 Market Analysis:")
                print(f"Current Price: ${current_price:.8f}")
                print(f"Average Spread: {avg_spread:.2f}%")
                print(f"Signal: {signals['recommendation']}")
                print(f"Risk Level: {risk['risk_level']}")
                
                for pos in open_positions:
                    trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
                    entry_price = float(entry_price)
                    current_pnl = ((current_price - entry_price) / entry_price) * 100
                    
                    print(f"\n📈 Position {trade_id}:")
                    print(f"Entry: ${entry_price:.8f}")
                    print(f"Current: ${current_price:.8f}")
                    print(f"P&L: {current_pnl:+.2f}%")
                    
                    # Check if we should exit based on signals
                    if signals['recommendation'] in ['SELL', 'TAKE_PROFIT']:
                        if current_pnl > avg_spread:  # Only exit if profit covers spread
                            print(f"🎯 Taking profit based on market signals")
                            if self.close_position(trade_id, current_price, "signal_based_exit"):
                                print(f"✅ Position closed successfully with {current_pnl:+.2f}% profit")
                            else:
                                print(f"❌ Failed to close position")
                    
                    # Check stop loss and take profit levels
                    self.check_stop_loss_take_profit(current_price)
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
            
            self.display_trading_dashboard()
            return
            
        # Continue with market analysis for new positions
        prices = []
        timestamps = []
        spreads = []
        last_price = None
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🔄 Starting Market Analysis: {symbol}")
        print(f"{'='*80}")
        
        try:
            while time.time() - start_time < time_period_minutes * 60:
                try:
                    price_data = self.get_shib_price()
                    
                    if price_data:
                        mid_price = price_data['price']
                        spread = price_data['spread']
                        spread_pct = (spread / mid_price) * 100
                        
                        if self.reference_price is None:
                            self.reference_price = mid_price
                            self.last_price = mid_price
                            
                        if self.last_price:
                            price_change_pct = ((mid_price - self.reference_price) / self.reference_price) * 100
                        else:
                            price_change_pct = 0.0
                        
                        prices.append(mid_price)
                        timestamps.append(datetime.now())
                        spreads.append(spread_pct)
                        self.last_price = mid_price
                        
                        # Display current market info
                        print(f"\n{datetime.now().strftime('%H:%M:%S')} Market Update:")
                        print(f"Price: ${mid_price:.8f}")
                        print(f"Spread: {spread_pct:.2f}%")
                        print(f"Change from start: {price_change_pct:+.2f}%")
                        
                        # Display position updates
                        if open_positions:
                            print("\n📈 Open Positions:")
                            for pos in open_positions:
                                trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
                                pos_pnl = ((mid_price - float(entry_price)) / float(entry_price)) * 100
                                print(f"ID {trade_id}: Entry ${float(entry_price):.8f} | P&L: {pos_pnl:+.2f}%")
                        
                        # Check stop loss and take profit for open positions
                        self.check_stop_loss_take_profit(mid_price)
                    
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print("\n⚠️ Analysis interrupted")
            return
            
        # Create analysis DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'spread': spreads
        })
        df.set_index('timestamp', inplace=True)
        
        # Get trading signals and risk assessment
        signals = self.signal_aggregator.analyze(df, np.mean(spreads))
        risk = self.signal_aggregator.get_risk_assessment(df)
        
        # Execute trades if conditions are met
        avg_spread = np.mean(spreads)
        current_positions = len(self.get_open_positions())
        
        print("\n🤔 Trade Decision Analysis:")
        print(f"Current Spread: {avg_spread:.2f}% (Max allowed: {self.max_spread*100:.2f}%)")
        print(f"Open Positions: {current_positions}/{self.max_positions}")
        print(f"Signal Recommendation: {signals['recommendation']}")
        
        # Enhanced indicator analysis output
        print("\n📊 Indicator Analysis:")
        indicator_values = signals['indicator_values']
        for indicator, value in indicator_values.items():
            print(f"{indicator}: {value}")
            
        print(f"\n📈 Market Conditions:")
        print(f"Market Volatility: {risk['volatility']*100:.1f}%")
        print(f"Spread Favorability: {'Favorable' if signals['spread_analysis']['is_spread_favorable'] else 'Unfavorable'}")
        
        # Calculate position size based on risk
        volatility_factor = 1 - risk['volatility']
        spread_factor = 1 - (avg_spread / (self.max_spread * 100))
        risk_adjusted_size = self.position_size * min(volatility_factor, spread_factor)
        
        print(f"\n🎯 Risk Analysis:")
        print(f"Volatility Factor: {volatility_factor:.2f}")
        print(f"Spread Factor: {spread_factor:.2f}")
        print(f"Risk-Adjusted Position Size: ${risk_adjusted_size:.2f}")
        
        if current_positions >= self.max_positions:
            print("\n❌ No trade: Maximum positions ({self.max_positions}) reached")
            print(f"Currently holding {current_positions} positions. Wait for positions to close before new entries.")
        elif signals['recommendation'] == 'WAIT':
            print("\n❌ No trade: Market conditions not optimal")
            if signals['buy_sentiment'] < 60:
                print(f"Buy sentiment ({signals['buy_sentiment']:.1f}%) below minimum threshold (60%)")
                print("Technical indicators suggest weak upward momentum")
            if avg_spread > (self.max_spread * 100 * 0.8):
                print(f"Spread ({avg_spread:.2f}%) approaching maximum threshold ({self.max_spread*100:.2f}%)")
                print("High spread would eat into potential profits")
            if risk['volatility']*100 > 15:
                print(f"High market volatility ({risk['volatility']*100:.1f}%) suggesting caution")
                print("Waiting for more stable market conditions")
        elif signals['recommendation'] == 'HOLD':
            print("\n⏸️ No trade: Market in consolidation")
            print("Technical Analysis Summary:")
            if 'buy_sentiment' in signals:
                print(f"- Buy Sentiment: {signals['buy_sentiment']:.1f}% (Need > 80% for entry)")
            if 'sell_sentiment' in signals:
                print(f"- Sell Sentiment: {signals['sell_sentiment']:.1f}% (Need > 80% for exit)")
            print("- Market showing mixed signals")
            print("- Waiting for clearer directional movement")
        elif avg_spread > (self.max_spread * 100):
            print(f"\n❌ No trade: Spread too high ({avg_spread:.2f}% > {self.max_spread*100:.2f}%)")
            print("Transaction costs would significantly impact potential returns")
            print("Waiting for better market liquidity conditions")
        else:
            # Execute trades based on strong signals
            if signals['recommendation'] == 'BUY' and signals['buy_sentiment'] > 80:
                try:
                    current_price = prices[-1]
                    if current_price is None or current_price == 0:
                        print("\n❌ Error: Invalid price for BUY order")
                        return
                    quantity = risk_adjusted_size / float(current_price)
                    print(f"\n✅ Executing BUY:")
                    print(f"Quantity: {quantity:.2f} {symbol}")
                    print(f"Price: ${current_price:.8f}")
                    self.place_order(symbol, 'buy', quantity)
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    print(f"\n❌ Error calculating BUY order details: {str(e)}")
                    return
            
            elif signals['recommendation'] == 'SELL' and signals['sell_sentiment'] > 80:
                try:
                    current_price = prices[-1]
                    if current_price is None:
                        print("\n❌ Error: Invalid price for SELL order")
                        return
                    positions = self.get_open_positions()
                    for pos in positions:
                        if pos[1] == symbol and pos[2] == 'buy':
                            should_close, reason = self.should_close_position(pos, current_price, signals['recommendation'])
                            if should_close:
                                print(f"\n✅ Executing SELL:")
                                print(f"Price: ${current_price:.8f}")
                                self.close_position(pos[0], float(current_price), reason)
                            else:
                                print(f"\n❌ Not closing position: {reason}")
                except (TypeError, ValueError) as e:
                    print(f"\n❌ Error calculating SELL order details: {str(e)}")
                    return
            else:
                print("\n❌ No trade: Signal sentiment not strong enough")
        
        # Display comprehensive dashboard
        self.display_trading_dashboard()
        
        return {
            "signals": signals,
            "risk_assessment": risk,
            "average_spread": avg_spread,
            "current_price": prices[-1]
        }

    def should_close_position(self, position, current_price, signal_recommendation):
        """Determine if a position should be closed based on various factors"""
        
        # Calculate current P&L
        entry_price = float(position[3])  # Entry price from position data
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Get historical volatility and volume trend
        volatility = self._calculate_historical_volatility()
        volume_analysis = self._analyze_volume_trend()
        
        # Don't sell at a loss unless strong indicators present
        if pnl_pct < 0:
            if pnl_pct <= -self.stop_loss_pct:
                return True, "hard stop loss triggered"
                
            # Only exit on loss if strong downtrend confirmed
            if volume_analysis['strong_downtrend'] and pnl_pct < -5:
                return True, "strong downtrend detected"
                
            # Hold through normal market fluctuations
            return False, "holding through market fluctuation"
            
        # Enhanced profit taking logic based on market conditions
        if pnl_pct > 0:
            # Quick scalp in high volatility conditions
            if volatility > 2.0 and pnl_pct >= self.min_profit_threshold:
                return True, "scalping profit in high volatility"
                
            # Take larger profits in strong trends
            if pnl_pct >= self.take_profit_pct:
                if volume_analysis['strong_uptrend']:
                    # Hold for potential further gains
                    return False, "holding in strong uptrend"
                else:
                    return True, "taking full profit target"
                    
            # Take profits if momentum is weakening
            if pnl_pct >= self.min_profit_threshold * 2:  # At least 2x min profit
                if volume_analysis['momentum_weakening']:
                    return True, "taking profit on weakening momentum"
                    
            # Exit on strong sell signals if we have decent profit
            if signal_recommendation == "SELL" and pnl_pct >= self.min_profit_threshold:
                if volume_analysis['strong_downtrend'] or volume_analysis['momentum_weakening']:
                    return True, "taking profit on sell signal"
        
        return False, "holding position"

    def get_shib_price(self) -> dict:
        """Get real-time SHIB price from Robinhood"""
        path = "/api/v1/crypto/marketdata/best_bid_ask/?symbol=SHIB-USD"
        headers = self.get_authorization_header("GET", path)
        
        try:
            response = requests.get(f"{self.api_url}{path}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and 'results' in data and data['results']:
                    result = data['results'][0]
                    return {
                        'bid': float(result['bid_inclusive_of_sell_spread']),
                        'ask': float(result['ask_inclusive_of_buy_spread']),
                        'price': float(result['price']),
                        'spread': float(result['ask_inclusive_of_buy_spread']) - float(result['bid_inclusive_of_sell_spread'])
                    }
            return None
        except Exception as e:
            print(f"Error getting SHIB price: {str(e)}")
            return None

    def get_authorization_header(self, method: str, path: str, body: str = "") -> dict:
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        message = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.signing_key.sign(message.encode("utf-8"))
        
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp)
        }

    def get_open_positions(self) -> list:
        """Get all open positions"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        try:
            c.execute('''SELECT id, symbol, side, entry_price, quantity, entry_time, status, 
                        stop_loss, take_profit 
                        FROM trades 
                        WHERE status = 'OPEN' 
                        ORDER BY entry_time DESC''')
            positions = c.fetchall()
            return positions
        except sqlite3.Error as e:
            print(f"Database error in get_open_positions: {str(e)}")
            return []
        finally:
            conn.close()

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for the given symbol"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM trades WHERE symbol = ? AND status = "OPEN"', (symbol,))
        count = c.fetchone()[0]
        
        conn.close()
        return count > 0

    def reset_stop_losses(self):
        """Reset stop losses for all open positions to be below entry price"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        try:
            c.execute('SELECT id, entry_price FROM trades WHERE status = "OPEN"')
            positions = c.fetchall()
            
            for trade_id, entry_price in positions:
                entry_price = float(entry_price)
                correct_stop_loss = round(entry_price * (1 - self.stop_loss_pct), 8)  # 10% below entry
                take_profit = round(entry_price * (1 + self.take_profit_pct), 8)  # 15% above entry
                c.execute('UPDATE trades SET stop_loss = ?, take_profit = ? WHERE id = ?', 
                         (correct_stop_loss, take_profit, trade_id))
            
            conn.commit()
            print("\n🔄 Reset stop losses and take profits to correct levels")
        except Exception as e:
            print(f"\n❌ Error resetting stop losses: {str(e)}")
        finally:
            conn.close()

    def display_trading_dashboard(self):
        """Display current trading status with enhanced metrics"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        # Get open positions with stop loss and take profit
        open_positions = self.get_open_positions()
        
        # Format open positions for display
        formatted_positions = []
        for pos in open_positions:
            trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
            formatted_positions.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"{quantity:.0f}",
                f"${format(float(stop_loss), '.8f')}",
                f"${format(float(take_profit), '.8f')}",
                entry_time
            ])
        
        # Get recent closed trades
        c.execute('''SELECT symbol, side, entry_price, exit_price, pnl_percentage,
                           entry_time, exit_time 
                    FROM trades WHERE status = "CLOSED" 
                    ORDER BY exit_time DESC LIMIT 5''')
        recent_trades = c.fetchall()
        
        # Format recent trades for display
        formatted_trades = []
        for trade in recent_trades:
            symbol, side, entry_price, exit_price, pnl_pct, entry_time, exit_time = trade
            formatted_trades.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"${format(float(exit_price), '.8f')}",
                f"{pnl_pct:+.2f}%",
                entry_time,
                exit_time
            ])
        
        # Calculate trading metrics
        c.execute('SELECT COUNT(*), SUM(pnl), AVG(pnl_percentage) FROM trades WHERE status = "CLOSED"')
        total_trades, total_pnl, avg_pnl = c.fetchone()
        
        # Handle null values
        total_trades = total_trades or 0
        total_pnl = total_pnl or 0.0
        avg_pnl = avg_pnl or 0.0
        
        c.execute('SELECT COUNT(*) FROM trades WHERE status = "CLOSED" AND pnl > 0')
        winning_trades = c.fetchone()[0] or 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n📊 === Trading Dashboard ===")
        
        print("\n💼 Open Positions:")
        if formatted_positions:
            headers = ['Symbol', 'Side', 'Entry $', 'Qty', 'Stop Loss', 'Take Profit', 'Time']
            print(tabulate(formatted_positions, headers=headers, tablefmt='grid'))
        else:
            print("No open positions")
            
        print("\n📈 Recent Trades:")
        if formatted_trades:
            headers = ['Symbol', 'Side', 'Entry $', 'Exit $', 'P&L %', 'Entry Time', 'Exit Time']
            print(tabulate(formatted_trades, headers=headers, tablefmt='grid'))
        else:
            print("No recent trades")
            
        print("\n📉 Trading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average P&L: {avg_pnl:.2f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        
        conn.close()

    def analyze_market(self, symbol: str = "SHIB-USD", time_period_minutes: int = 1):
        """Analyze market and execute trades with enhanced risk management"""
        
        # Get current open positions count
        open_positions = self.get_open_positions()
        num_positions = len(open_positions)
        
        # Check if we've hit position limit
        if num_positions >= self.max_positions:
            print(f"\n⚠️ Maximum positions ({self.max_positions}) reached. Monitoring only.")
            
            # Create DataFrame for analysis
            prices = []
            timestamps = []
            spreads = []
            
            try:
                # Collect some price data for analysis
                for _ in range(6):  # 30 seconds of data
                    price_data = self.get_shib_price()
                    if price_data:
                        mid_price = price_data['price']
                        spread = price_data['spread']
                        spread_pct = (spread / mid_price) * 100
                        
                        prices.append(mid_price)
                        timestamps.append(datetime.now())
                        spreads.append(spread_pct)
                    time.sleep(5)
                
                # Create analysis DataFrame
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'price': prices,
                    'spread': spreads
                })
                df.set_index('timestamp', inplace=True)
                
                # Get trading signals and risk assessment
                signals = self.signal_aggregator.analyze(df, np.mean(spreads))
                risk = self.signal_aggregator.get_risk_assessment(df)
                
                # Monitor and potentially exit positions based on signals
                current_price = prices[-1]
                avg_spread = np.mean(spreads)
                
                print(f"\n🔍 Market Analysis:")
                print(f"Current Price: ${current_price:.8f}")
                print(f"Average Spread: {avg_spread:.2f}%")
                print(f"Signal: {signals['recommendation']}")
                print(f"Risk Level: {risk['risk_level']}")
                
                for pos in open_positions:
                    trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
                    entry_price = float(entry_price)
                    current_pnl = ((current_price - entry_price) / entry_price) * 100
                    
                    print(f"\n📈 Position {trade_id}:")
                    print(f"Entry: ${entry_price:.8f}")
                    print(f"Current: ${current_price:.8f}")
                    print(f"P&L: {current_pnl:+.2f}%")
                    
                    # Check if we should exit based on signals
                    if signals['recommendation'] in ['SELL', 'TAKE_PROFIT']:
                        if current_pnl > avg_spread:  # Only exit if profit covers spread
                            print(f"🎯 Taking profit based on market signals")
                            if self.close_position(trade_id, current_price, "signal_based_exit"):
                                print(f"✅ Position closed successfully with {current_pnl:+.2f}% profit")
                            else:
                                print(f"❌ Failed to close position")
                    
                    # Check stop loss and take profit levels
                    self.check_stop_loss_take_profit(current_price)
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
            
            self.display_trading_dashboard()
            return
            
        # Continue with market analysis for new positions
        prices = []
        timestamps = []
        spreads = []
        last_price = None
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🔄 Starting Market Analysis: {symbol}")
        print(f"{'='*80}")
        
        try:
            while time.time() - start_time < time_period_minutes * 60:
                try:
                    price_data = self.get_shib_price()
                    
                    if price_data:
                        mid_price = price_data['price']
                        spread = price_data['spread']
                        spread_pct = (spread / mid_price) * 100
                        
                        if self.reference_price is None:
                            self.reference_price = mid_price
                            self.last_price = mid_price
                            
                        if self.last_price:
                            price_change_pct = ((mid_price - self.reference_price) / self.reference_price) * 100
                        else:
                            price_change_pct = 0.0
                        
                        prices.append(mid_price)
                        timestamps.append(datetime.now())
                        spreads.append(spread_pct)
                        self.last_price = mid_price
                        
                        # Display current market info
                        print(f"\n{datetime.now().strftime('%H:%M:%S')} Market Update:")
                        print(f"Price: ${mid_price:.8f}")
                        print(f"Spread: {spread_pct:.2f}%")
                        print(f"Change from start: {price_change_pct:+.2f}%")
                        
                        # Display position updates
                        if open_positions:
                            print("\n📈 Open Positions:")
                            for pos in open_positions:
                                trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
                                pos_pnl = ((mid_price - float(entry_price)) / float(entry_price)) * 100
                                print(f"ID {trade_id}: Entry ${float(entry_price):.8f} | P&L: {pos_pnl:+.2f}%")
                        
                        # Check stop loss and take profit for open positions
                        self.check_stop_loss_take_profit(mid_price)
                    
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print("\n⚠️ Analysis interrupted")
            return
            
        # Create analysis DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'spread': spreads
        })
        df.set_index('timestamp', inplace=True)
        
        # Get trading signals and risk assessment
        signals = self.signal_aggregator.analyze(df, np.mean(spreads))
        risk = self.signal_aggregator.get_risk_assessment(df)
        
        # Execute trades if conditions are met
        avg_spread = np.mean(spreads)
        current_positions = len(self.get_open_positions())
        
        print("\n🤔 Trade Decision Analysis:")
        print(f"Current Spread: {avg_spread:.2f}% (Max allowed: {self.max_spread*100:.2f}%)")
        print(f"Open Positions: {current_positions}/{self.max_positions}")
        print(f"Signal Recommendation: {signals['recommendation']}")
        if signals['recommendation'] == 'BUY':
            print(f"Buy Sentiment: {signals['buy_sentiment']:.1f}% (Min required: 80%)")
        elif signals['recommendation'] == 'SELL':
            print(f"Sell Sentiment: {signals['sell_sentiment']:.1f}% (Min required: 80%)")
        print(f"Market Volatility: {risk['volatility']*100:.1f}%")
        
        # Calculate position size based on risk
        volatility_factor = 1 - risk['volatility']
        spread_factor = 1 - (avg_spread / (self.max_spread * 100))  # Convert max_spread to percentage
        risk_adjusted_size = self.position_size * min(volatility_factor, spread_factor)
        
        print(f"\nRisk Analysis:")
        print(f"Volatility Factor: {volatility_factor:.2f}")
        print(f"Spread Factor: {spread_factor:.2f}")
        print(f"Risk-Adjusted Position Size: ${risk_adjusted_size:.2f}")
        
        if current_positions >= self.max_positions:
            print("\n❌ No trade: Maximum positions reached")
            print(f"Currently holding {current_positions} positions. Wait for positions to close before new entries.")
        elif signals['recommendation'] == 'WAIT':
            print("\n❌ No trade: Market conditions not optimal")
            if signals['buy_sentiment'] < 60:
                print(f"Buy sentiment ({signals['buy_sentiment']:.1f}%) below minimum threshold (60%)")
                print("Technical indicators suggest weak upward momentum")
            if avg_spread > (self.max_spread * 100 * 0.8):
                print(f"Spread ({avg_spread:.2f}%) approaching maximum threshold ({self.max_spread*100:.2f}%)")
                print("High spread would eat into potential profits")
            if risk['volatility']*100 > 15:
                print(f"High market volatility ({risk['volatility']*100:.1f}%) suggesting caution")
                print("Waiting for more stable market conditions")
        elif signals['recommendation'] == 'HOLD':
            print("\n⏸️ No trade: Market in consolidation")
            print("Technical Analysis Summary:")
            if 'buy_sentiment' in signals:
                print(f"- Buy Sentiment: {signals['buy_sentiment']:.1f}% (Need > 80% for entry)")
            if 'sell_sentiment' in signals:
                print(f"- Sell Sentiment: {signals['sell_sentiment']:.1f}% (Need > 80% for exit)")
            print("- Market showing mixed signals")
            print("- Waiting for clearer directional movement")
        elif avg_spread > (self.max_spread * 100):  # Compare with percentage
            print("\n❌ No trade: Spread too high")
            print(f"Current spread ({avg_spread:.2f}%) exceeds maximum allowed ({self.max_spread*100:.2f}%)")
            print("Transaction costs would significantly impact potential returns")
        else:
            # Execute trades based on strong signals
            if signals['recommendation'] == 'BUY' and signals['buy_sentiment'] > 80:
                try:
                    current_price = prices[-1]
                    if current_price is None or current_price == 0:
                        print("\n❌ Error: Invalid price for BUY order")
                        return
                    quantity = risk_adjusted_size / float(current_price)
                    print(f"\n✅ Executing BUY:")
                    print(f"Quantity: {quantity:.2f} {symbol}")
                    print(f"Price: ${current_price:.8f}")
                    self.place_order(symbol, 'buy', quantity)
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    print(f"\n❌ Error calculating BUY order details: {str(e)}")
                    return
            
            elif signals['recommendation'] == 'SELL' and signals['sell_sentiment'] > 80:
                try:
                    current_price = prices[-1]
                    if current_price is None:
                        print("\n❌ Error: Invalid price for SELL order")
                        return
                    positions = self.get_open_positions()
                    for pos in positions:
                        if pos[1] == symbol and pos[2] == 'buy':
                            should_close, reason = self.should_close_position(pos, current_price, signals['recommendation'])
                            if should_close:
                                print(f"\n✅ Executing SELL:")
                                print(f"Price: ${current_price:.8f}")
                                self.close_position(pos[0], float(current_price), reason)
                            else:
                                print(f"\n❌ Not closing position: {reason}")
                except (TypeError, ValueError) as e:
                    print(f"\n❌ Error calculating SELL order details: {str(e)}")
                    return
            else:
                print("\n❌ No trade: Signal sentiment not strong enough")
        
        # Display comprehensive dashboard
        self.display_trading_dashboard()
        
        return {
            "signals": signals,
            "risk_assessment": risk,
            "average_spread": avg_spread,
            "current_price": prices[-1]
        }

    def get_shib_price(self) -> dict:
        """Get real-time SHIB price from Robinhood"""
        path = "/api/v1/crypto/marketdata/best_bid_ask/?symbol=SHIB-USD"
        headers = self.get_authorization_header("GET", path)
        
        try:
            response = requests.get(f"{self.api_url}{path}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and 'results' in data and data['results']:
                    result = data['results'][0]
                    return {
                        'bid': float(result['bid_inclusive_of_sell_spread']),
                        'ask': float(result['ask_inclusive_of_buy_spread']),
                        'price': float(result['price']),
                        'spread': float(result['ask_inclusive_of_buy_spread']) - float(result['bid_inclusive_of_sell_spread'])
                    }
            return None
        except Exception as e:
            print(f"Error getting SHIB price: {str(e)}")
            return None

    def get_authorization_header(self, method: str, path: str, body: str = "") -> dict:
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        message = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.signing_key.sign(message.encode("utf-8"))
        
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp)
        }

    def get_open_positions(self) -> list:
        """Get all open positions"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        try:
            c.execute('''SELECT id, symbol, side, entry_price, quantity, entry_time, status, 
                        stop_loss, take_profit 
                        FROM trades 
                        WHERE status = 'OPEN' 
                        ORDER BY entry_time DESC''')
            positions = c.fetchall()
            return positions
        except sqlite3.Error as e:
            print(f"Database error in get_open_positions: {str(e)}")
            return []
        finally:
            conn.close()

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for the given symbol"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM trades WHERE symbol = ? AND status = "OPEN"', (symbol,))
        count = c.fetchone()[0]
        
        conn.close()
        return count > 0

    def reset_stop_losses(self):
        """Reset stop losses for all open positions to be below entry price"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        try:
            c.execute('SELECT id, entry_price FROM trades WHERE status = "OPEN"')
            positions = c.fetchall()
            
            for trade_id, entry_price in positions:
                entry_price = float(entry_price)
                correct_stop_loss = round(entry_price * (1 - self.stop_loss_pct), 8)  # 10% below entry
                take_profit = round(entry_price * (1 + self.take_profit_pct), 8)  # 15% above entry
                c.execute('UPDATE trades SET stop_loss = ?, take_profit = ? WHERE id = ?', 
                         (correct_stop_loss, take_profit, trade_id))
            
            conn.commit()
            print("\n🔄 Reset stop losses and take profits to correct levels")
        except Exception as e:
            print(f"\n❌ Error resetting stop losses: {str(e)}")
        finally:
            conn.close()

    def display_trading_dashboard(self):
        """Display current trading status with enhanced metrics"""
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        # Get open positions with stop loss and take profit
        open_positions = self.get_open_positions()
        
        # Format open positions for display
        formatted_positions = []
        for pos in open_positions:
            trade_id, symbol, side, entry_price, quantity, entry_time, status, stop_loss, take_profit = pos
            formatted_positions.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"{quantity:.0f}",
                f"${format(float(stop_loss), '.8f')}",
                f"${format(float(take_profit), '.8f')}",
                entry_time
            ])
        
        # Get recent closed trades
        c.execute('''SELECT symbol, side, entry_price, exit_price, pnl_percentage,
                           entry_time, exit_time 
                    FROM trades WHERE status = "CLOSED" 
                    ORDER BY exit_time DESC LIMIT 5''')
        recent_trades = c.fetchall()
        
        # Format recent trades for display
        formatted_trades = []
        for trade in recent_trades:
            symbol, side, entry_price, exit_price, pnl_pct, entry_time, exit_time = trade
            formatted_trades.append([
                symbol,
                side,
                f"${format(float(entry_price), '.8f')}",
                f"${format(float(exit_price), '.8f')}",
                f"{pnl_pct:+.2f}%",
                entry_time,
                exit_time
            ])
        
        # Calculate trading metrics
        c.execute('SELECT COUNT(*), SUM(pnl), AVG(pnl_percentage) FROM trades WHERE status = "CLOSED"')
        total_trades, total_pnl, avg_pnl = c.fetchone()
        
        # Handle null values
        total_trades = total_trades or 0
        total_pnl = total_pnl or 0.0
        avg_pnl = avg_pnl or 0.0
        
        c.execute('SELECT COUNT(*) FROM trades WHERE status = "CLOSED" AND pnl > 0')
        winning_trades = c.fetchone()[0] or 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n📊 === Trading Dashboard ===")
        
        print("\n💼 Open Positions:")
        if formatted_positions:
            headers = ['Symbol', 'Side', 'Entry $', 'Qty', 'Stop Loss', 'Take Profit', 'Time']
            print(tabulate(formatted_positions, headers=headers, tablefmt='grid'))
        else:
            print("No open positions")
            
        print("\n📈 Recent Trades:")
        if formatted_trades:
            headers = ['Symbol', 'Side', 'Entry $', 'Exit $', 'P&L %', 'Entry Time', 'Exit Time']
            print(tabulate(formatted_trades, headers=headers, tablefmt='grid'))
        else:
            print("No recent trades")
            
        print("\n📉 Trading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average P&L: {avg_pnl:.2f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        
        conn.close()

def main():
    analyzer = CryptoAnalyzer()
    
    print("\n🤖 Starting SHIB Trading Bot")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            analyzer.analyze_market("SHIB-USD", 1)
            print("\n⏳ Waiting for next cycle...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    main()
