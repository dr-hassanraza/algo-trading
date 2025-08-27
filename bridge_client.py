#!/usr/bin/env python3
"""
Bridge Client for Node.js Investing.com API
===========================================

Python client to communicate with the Node.js bridge server
for accessing investing.com data.
"""

import requests
import json
import time
import subprocess
import os
import signal
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class InvestingBridgeClient:
    """Client to communicate with Node.js investing.com bridge"""
    
    def __init__(self, bridge_url: str = "http://localhost:3001"):
        self.bridge_url = bridge_url
        self.bridge_process = None
        self.session = requests.Session()
        self.session.timeout = 30
    
    def start_bridge(self) -> bool:
        """Start the Node.js bridge server"""
        try:
            # Check if bridge is already running
            if self.is_bridge_running():
                logger.info("Bridge server already running")
                return True
            
            # Start the Node.js bridge
            bridge_script = os.path.join(os.path.dirname(__file__), 'nodejs_bridge.js')
            
            if not os.path.exists(bridge_script):
                logger.error(f"Bridge script not found: {bridge_script}")
                return False
            
            logger.info("Starting Node.js bridge server...")
            self.bridge_process = subprocess.Popen(
                ['node', bridge_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Wait for server to start
            for _ in range(10):  # Try for 10 seconds
                time.sleep(1)
                if self.is_bridge_running():
                    logger.info("Bridge server started successfully")
                    return True
            
            logger.error("Bridge server failed to start")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start bridge: {e}")
            return False
    
    def stop_bridge(self):
        """Stop the Node.js bridge server"""
        try:
            # Try graceful shutdown first
            response = self.session.get(f"{self.bridge_url}/shutdown", timeout=5)
            time.sleep(2)
        except:
            pass
        
        # Force kill if still running
        if self.bridge_process:
            try:
                os.killpg(os.getpgid(self.bridge_process.pid), signal.SIGTERM)
                self.bridge_process.wait(timeout=5)
            except:
                pass
            self.bridge_process = None
    
    def is_bridge_running(self) -> bool:
        """Check if bridge server is running"""
        try:
            response = self.session.get(f"{self.bridge_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_currency_data(self, pair: str, period: str = "P1M", interval: str = "P1D", points: int = 120) -> Optional[Dict]:
        """Get currency data from the bridge"""
        try:
            if not self.is_bridge_running():
                logger.warning("Bridge not running, attempting to start...")
                if not self.start_bridge():
                    return None
            
            url = f"{self.bridge_url}/currency/{pair}"
            params = {
                'period': period,
                'interval': interval,
                'points': points
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get currency data for {pair}: {e}")
            return None
    
    def get_pkr_rates(self) -> Dict[str, float]:
        """Get current PKR exchange rates"""
        rates = {}
        
        currencies = ['usd-pkr', 'eur-pkr', 'gbp-pkr', 'sar-pkr']
        
        for currency in currencies:
            try:
                data = self.get_currency_data(currency, period="P1D", points=1)
                if data and data.get('success') and data.get('meta'):
                    rates[currency] = data['meta']['current_rate']
            except Exception as e:
                logger.warning(f"Failed to get rate for {currency}: {e}")
        
        return rates
    
    def __enter__(self):
        """Context manager entry"""
        self.start_bridge()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_bridge()

# Integration function for the enhanced signal analyzer
def get_currency_context() -> Dict[str, float]:
    """Get currency context for PSX analysis"""
    
    try:
        with InvestingBridgeClient() as bridge:
            rates = bridge.get_pkr_rates()
            
            if rates:
                logger.info(f"Got currency rates: {rates}")
                return rates
            else:
                logger.warning("No currency rates available")
                return {}
                
    except Exception as e:
        logger.error(f"Failed to get currency context: {e}")
        return {}

# Test the bridge client
def test_bridge_client():
    """Test the investing bridge client"""
    
    print("🚀 Testing Investing.com Bridge Client")
    print("=" * 50)
    
    with InvestingBridgeClient() as bridge:
        
        # Test health check
        print("\\n🔍 Testing bridge health...")
        if bridge.is_bridge_running():
            print("   ✅ Bridge server is healthy")
        else:
            print("   ❌ Bridge server not responding")
            return
        
        # Test currency data
        print("\\n📊 Testing currency data...")
        currencies = ['usd-pkr', 'eur-pkr', 'gbp-pkr']
        
        for currency in currencies:
            print(f"\\n🔍 Testing {currency.upper()}...")
            data = bridge.get_currency_data(currency, period="P1W", points=7)
            
            if data and data.get('success'):
                meta = data.get('meta', {})
                historical = data.get('data', [])
                
                print(f"   ✅ Current rate: {meta.get('current_rate', 'N/A')} PKR")
                print(f"   📈 Change: {meta.get('change', 'N/A')} PKR")
                print(f"   📊 Historical points: {len(historical)}")
                
                if historical:
                    latest = historical[-1]
                    print(f"   📅 Latest: {latest['date']} - Close: {latest['close']:.2f}")
            else:
                print(f"   ❌ Failed to get data for {currency}")
        
        # Test PKR rates summary
        print("\\n💱 Testing PKR rates summary...")
        rates = bridge.get_pkr_rates()
        
        if rates:
            print("   ✅ PKR Exchange Rates:")
            for pair, rate in rates.items():
                currency = pair.split('-')[0].upper()
                print(f"      {currency}/PKR: {rate:.2f}")
        else:
            print("   ❌ No PKR rates available")
    
    print("\\n🏁 Bridge client test completed!")

if __name__ == "__main__":
    test_bridge_client()