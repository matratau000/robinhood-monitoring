"""
Helper module for tracking and verifying Robinhood entry prices
"""
import requests
import sqlite3
from datetime import datetime

class RHEntryPriceTracker:
    def __init__(self, api_url, get_auth_header_func):
        """
        Initialize the entry price tracker
        
        Args:
            api_url (str): Base URL for Robinhood API
            get_auth_header_func (callable): Function to get authorization headers
        """
        self.api_url = api_url
        self.get_auth_header = get_auth_header_func
        
    def get_order_details(self, order_id):
        """Get order details from Robinhood API"""
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        headers = self.get_auth_header("GET", path)
        
        try:
            response = requests.get(f"{self.api_url}{path}", headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting order details: {e}")
            return None

    def update_trade_with_actual_price(self, trade_id, db_path='trades.db'):
        """Update trade with actual execution price from Robinhood"""
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Get trade details including order_id
            c.execute('SELECT order_id FROM trades WHERE id = ?', (trade_id,))
            result = c.fetchone()
            
            if result and result[0]:
                order_id = result[0]
                # Get actual execution details from Robinhood
                order_details = self.get_order_details(order_id)
                
                if order_details and order_details.get('state') == 'filled':
                    executed_price = float(order_details.get('executed_price', 0))
                    if executed_price > 0:
                        # Update trade with actual entry price
                        c.execute('''
                            UPDATE trades 
                            SET entry_price = ? 
                            WHERE id = ?
                        ''', (executed_price, trade_id))
                        conn.commit()
                        print(f"✅ Updated trade {trade_id} with actual entry price: ${executed_price:.8f}")
                    
            conn.close()
        except Exception as e:
            print(f"Error updating trade entry price: {e}")

    def store_initial_trade(self, trade_id, symbol, side, estimated_price, quantity, 
                          order_id, stop_loss_pct=0.10, take_profit_pct=0.15, 
                          db_path='trades.db'):
        """
        Store initial trade details with estimated price
        
        Args:
            trade_id (str): Unique trade identifier
            symbol (str): Trading symbol
            side (str): Buy/Sell
            estimated_price (float): Estimated entry price
            quantity (float): Trade quantity
            order_id (str): Robinhood order ID
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            db_path (str): Path to SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Calculate stop loss and take profit
            stop_loss = round(estimated_price * (1 - stop_loss_pct), 8)
            take_profit = round(estimated_price * (1 + take_profit_pct), 8)
            
            # Store trade with initial estimated price
            c.execute('''INSERT INTO trades (
                            id, symbol, side, entry_price, quantity, entry_time,
                            status, stop_loss, take_profit, order_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (trade_id, symbol, side, estimated_price, quantity, datetime.now(),
                      'OPEN', stop_loss, take_profit, order_id))
            
            conn.commit()
            conn.close()
            
            print(f"\n✅ Stored initial trade:")
            print(f"Trade ID: {trade_id}")
            print(f"Order ID: {order_id}")
            print(f"Estimated Entry: ${estimated_price:.8f}")
            
            return True
            
        except Exception as e:
            print(f"Error storing initial trade: {e}")
            return False
