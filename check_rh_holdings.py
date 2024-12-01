import os
import json
import sqlite3
from datetime import datetime
import base64
import requests
from nacl.signing import SigningKey
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RobinhoodHoldingsChecker:
    def __init__(self):
        self.api_key = os.getenv('ROBINHOOD_API_KEY')
        self.base64_private_key = os.getenv('ROBINHOOD_PRIVATE_KEY')
        if not self.api_key or not self.base64_private_key:
            raise ValueError("Please set ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY environment variables")
        
        # Initialize the private key for signing requests
        private_key_seed = base64.b64decode(self.base64_private_key)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"
        
        # Connect to SQLite database
        self.conn = sqlite3.connect('trades.db')
        self.cursor = self.conn.cursor()
        
        # Ensure tables exist
        self.setup_database()

    def setup_database(self):
        """Create necessary tables if they don't exist"""
        # Table for tracking trades
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS trades
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
        self.conn.commit()

    def get_authorization_header(self, method: str, path: str, body: str = "") -> dict:
        """Generate authorization headers for API requests"""
        timestamp = int(time.time())
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))
        
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def make_api_request(self, method: str, path: str, body: str = "") -> dict:
        """Make an API request to Robinhood"""
        headers = self.get_authorization_header(method, path, body)
        url = self.base_url + path
        
        try:
            print(f"\n🔍 Making {method} request to: {url}")
            print(f"🔑 Headers: {json.dumps(headers, indent=2)}")
            
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            else:
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            
            print(f"📡 Response status: {response.status_code}")
            print(f"📦 Response body: {json.dumps(response.json(), indent=2)}")
            
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Error making API request: {e}")
            return None

    def get_current_holdings(self):
        """Fetch current holdings from Robinhood"""
        print("\n📊 Fetching current holdings from Robinhood...")
        path = "/api/v1/crypto/trading/holdings/"
        return self.make_api_request("GET", path)

    def get_db_positions(self):
        """Get all open positions from database"""
        print("\n💾 Fetching positions from database...")
        self.cursor.execute("SELECT id, symbol, side, entry_price, quantity, stop_loss, take_profit, entry_time FROM trades WHERE status = 'OPEN'")
        positions = self.cursor.fetchall()
        print(f"📝 Found {len(positions)} open positions in database")
        for pos in positions:
            print(f"  - {pos}")
        return positions

    def remove_closed_position(self, position_id):
        """Remove a position from the database"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.cursor.execute("""
            UPDATE trades 
            SET status = 'CLOSED', 
                exit_time = ?, 
                exit_price = 0,
                pnl = 0,
                pnl_percentage = 0
            WHERE id = ?
        """, (current_time, position_id))
        self.conn.commit()
        print(f"Closed position {position_id} in database")

    def sync_holdings(self):
        """Sync Robinhood holdings with local database"""
        print("\n🔄 Syncing Robinhood holdings with local database...")
        
        # Get current holdings from Robinhood
        rh_holdings = self.get_current_holdings()
        if not rh_holdings or 'results' not in rh_holdings:
            print("❌ Failed to fetch Robinhood holdings")
            return
        
        print(f"\n📈 Processing {len(rh_holdings['results'])} holdings from Robinhood")
        
        # Create a map of current holdings
        current_holdings = {}
        for holding in rh_holdings['results']:
            try:
                quantity = holding.get('total_quantity', '0')
                if float(quantity) > 0:
                    current_holdings[holding['asset_code']] = {
                        'quantity': float(quantity),
                        'available': float(holding.get('quantity_available_for_trading', '0'))
                    }
                    print(f"✅ Found active holding: {holding['asset_code']} - {quantity} (Available: {holding.get('quantity_available_for_trading', '0')})")
            except (KeyError, ValueError) as e:
                print(f"⚠️ Warning: Could not process holding {holding}: {str(e)}")
                continue
        
        # Get database positions
        db_positions = self.get_db_positions()
        
        # Track total quantities per symbol in database
        db_quantities = {}
        
        # Check each database position
        for position in db_positions:
            position_id, symbol, side, entry_price, quantity, stop_loss, take_profit, timestamp = position
            asset_code = symbol.split('-')[0]  # Convert SHIB-USD to SHIB
            print(f"\n🔍 Checking position {position_id} for {symbol}")
            
            # Add to total quantity for this symbol
            db_quantities[asset_code] = db_quantities.get(asset_code, 0) + float(quantity)
            
            # Remove position if:
            # 1. We have no holdings for this asset in Robinhood
            # 2. Our total quantity in DB exceeds what we have in Robinhood
            should_remove = False
            
            if asset_code not in current_holdings:
                print(f"🚫 No holdings found for {asset_code} in Robinhood")
                should_remove = True
            elif db_quantities[asset_code] > current_holdings[asset_code]['quantity']:
                print(f"🚫 Database quantity ({db_quantities[asset_code]}) exceeds Robinhood quantity ({current_holdings[asset_code]['quantity']})")
                should_remove = True
                
            if should_remove:
                self.remove_closed_position(position_id)
                print(f"🚫 Removed {symbol} position {position_id}")
            else:
                print(f"✅ Position {position_id} is valid")
        
        print("\n✅ Holdings sync complete!")
        
        # Print current state
        print("\n📊 Current Holdings Summary:")
        if not current_holdings:
            print("No active holdings found in Robinhood")
        else:
            for asset, details in current_holdings.items():
                qty = details['quantity']
                if qty > 0:
                    print(f"{asset}: {qty:,.8f}")
                    if asset in db_quantities:
                        print(f"  DB Quantity: {db_quantities[asset]:,.8f}")

    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    try:
        checker = RobinhoodHoldingsChecker()
        checker.sync_holdings()
        checker.close()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY environment variables are set")

if __name__ == "__main__":
    main()
