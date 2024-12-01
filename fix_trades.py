import sqlite3
from datetime import datetime

def fix_trades():
    # Connect to the database
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()
    
    # Get all open trades
    cursor.execute("""
        SELECT id, entry_price, side, quantity
        FROM trades 
        WHERE status = 'OPEN'
    """)
    open_trades = cursor.fetchall()
    
    # Fix each open trade
    for trade_id, entry_price, side, quantity in open_trades:
        if entry_price:
            entry_price = float(entry_price)
            
            # For buy orders
            if side.lower() == 'buy':
                # Calculate stop loss price (0.5% below entry)
                stop_loss = round(entry_price * 0.995, 8)
                # Calculate take profit (1.5% above entry)
                take_profit = round(entry_price * 1.015, 8)
            
            # Update the trade
            cursor.execute("""
                UPDATE trades 
                SET 
                    stop_loss = ?,
                    take_profit = ?
                WHERE id = ?
            """, (stop_loss, take_profit, trade_id))
            print(f"Updated trade {trade_id}:")
            print(f"Entry Price: {entry_price}")
            print(f"Stop Loss: {stop_loss} (-0.5%)")
            print(f"Take Profit: {take_profit} (+1.5%)")
            print("---")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("\nSuccessfully fixed stop loss and take profit values for open trades")

if __name__ == "__main__":
    try:
        fix_trades()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
