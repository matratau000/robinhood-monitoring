import os
import base64
import requests
from datetime import datetime
from nacl.signing import SigningKey
from dotenv import load_dotenv
from dateutil import tz

def test_auth():
    load_dotenv()
    
    # Load credentials
    api_key = os.getenv('ROBINHOOD_API_KEY')
    private_key = SigningKey(base64.b64decode(os.getenv('ROBINHOOD_PRIVATE_KEY')))
    
    # API endpoint details
    base_url = "https://trading.robinhood.com"
    path = "/api/v1/crypto/trading/accounts/"
    
    # Generate timestamp
    timestamp = int(datetime.now(tz=tz.tzutc()).timestamp())
    
    # Create message and sign it
    message = f"{api_key}{timestamp}{path}GET"
    signed = private_key.sign(message.encode("utf-8"))
    
    # Prepare headers
    headers = {
        "x-api-key": api_key,
        "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
        "x-timestamp": str(timestamp),
        "Content-Type": "application/json; charset=utf-8"
    }
    
    # Print request details
    print("Request Details:")
    print(f"URL: {base_url}{path}")
    print("\nHeaders:")
    for key, value in headers.items():
        print(f"{key}: {value}")
    print(f"\nMessage being signed: {message}")
    
    # Make request
    try:
        response = requests.get(f"{base_url}{path}", headers=headers)
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_auth()
