"""
PSX Terminal API Integration
Professional implementation of the PSX Terminal API for quantitative trading
Supports both REST API and WebSocket streaming
"""

import requests
import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import threading
import time
import logging
from dataclasses import dataclass, asdict
import asyncio
import ssl

@dataclass
class MarketTick:
    """Market tick data structure"""
    symbol: str
    market: str
    status: str
    price: float
    change: float
    change_percent: float
    volume: int
    trades: int
    value: float
    high: float
    low: float
    bid: float
    ask: float
    bid_volume: int
    ask_volume: int
    timestamp: int

@dataclass
class KLineData:
    """K-line/Candlestick data structure"""
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    quote_volume: Optional[float] = None
    trades: Optional[int] = None

@dataclass
class DividendData:
    """Dividend data structure"""
    symbol: str
    ex_date: str
    payment_date: str
    record_date: str
    amount: float
    year: int

class PSXTerminalAPI:
    """
    Professional PSX Terminal API client with REST and WebSocket support
    """
    
    def __init__(self):
        self.base_url = "https://psxterminal.com"
        self.ws_url = "wss://psxterminal.com/"
        self.session = requests.Session()
        self.ws = None
        self.subscriptions = {}
        self.callbacks = {}
        self.client_id = None
        self.is_connected = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'PSX-Quant-Trading-System/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    # ============================================================================
    # REST API METHODS
    # ============================================================================
    
    def test_connectivity(self) -> Dict[str, Any]:
        """Test API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/api/status", timeout=10)
            response.raise_for_status()
            data = response.json()
            self.logger.info("✅ PSX Terminal API connectivity test successful")
            return data
        except Exception as e:
            self.logger.error(f"❌ PSX Terminal API connectivity failed: {str(e)}")
            raise

    def get_market_data(self, market_type: str, symbol: str) -> Optional[MarketTick]:
        """
        Get real-time market data for a specific symbol
        
        Args:
            market_type: Market type (REG, FUT, IDX, ODL, BNB)
            symbol: Symbol name (e.g., HUBC)
        """
        try:
            url = f"{self.base_url}/api/ticks/{market_type}/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                tick_data = data['data']
                return MarketTick(
                    symbol=tick_data['symbol'],
                    market=tick_data['market'],
                    status=tick_data['st'],
                    price=tick_data['price'],
                    change=tick_data['change'],
                    change_percent=tick_data['changePercent'],
                    volume=tick_data['volume'],
                    trades=tick_data['trades'],
                    value=tick_data['value'],
                    high=tick_data['high'],
                    low=tick_data['low'],
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    bid_volume=tick_data['bidVol'],
                    ask_volume=tick_data['askVol'],
                    timestamp=tick_data['timestamp']
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        try:
            response = self.session.get(f"{self.base_url}/api/symbols", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                symbols = data['data']
                self.logger.info(f"✅ Retrieved {len(symbols)} symbols from PSX Terminal")
                return symbols
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {str(e)}")
            return []

    def get_market_stats(self, stats_type: str) -> Optional[Dict[str, Any]]:
        """
        Get market statistics
        
        Args:
            stats_type: Stats type (REG, IDX, BNB, ODL, FUT, breadth, sectors)
        """
        try:
            response = self.session.get(f"{self.base_url}/api/stats/{stats_type}", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data['data']
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching {stats_type} stats: {str(e)}")
            return None

    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed company information"""
        try:
            response = self.session.get(f"{self.base_url}/api/companies/{symbol}", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data['data']
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental analysis data and financial ratios"""
        try:
            response = self.session.get(f"{self.base_url}/api/fundamentals/{symbol}", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                return data['data']
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return None

    def get_klines(self, symbol: str, timeframe: str, start: Optional[str] = None, 
                  end: Optional[str] = None, limit: Optional[int] = None) -> List[KLineData]:
        """
        Get historical candlestick/kline data
        
        Args:
            symbol: Symbol name
            timeframe: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start: Start timestamp (13-digit Unix timestamp in milliseconds)
            end: End timestamp (13-digit Unix timestamp in milliseconds)
            limit: Number of records to return (max: 100)
        """
        try:
            url = f"{self.base_url}/api/klines/{symbol}/{timeframe}"
            params = {}
            
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            if limit:
                params['limit'] = limit
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                klines = []
                for kline in data['data']:
                    klines.append(KLineData(
                        symbol=kline['symbol'],
                        timeframe=kline['timeframe'],
                        timestamp=kline['timestamp'],
                        open=kline['open'],
                        high=kline['high'],
                        low=kline['low'],
                        close=kline['close'],
                        volume=kline['volume'],
                        quote_volume=kline.get('quoteVolume'),
                        trades=kline.get('trades')
                    ))
                return klines
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {str(e)}")
            return []

    def get_kline_by_timestamp(self, symbol: str, timeframe: str, timestamp: str) -> Optional[KLineData]:
        """Get single k-line by exact timestamp"""
        try:
            url = f"{self.base_url}/api/klines/{symbol}/{timeframe}/{timestamp}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                kline = data['data']
                return KLineData(
                    symbol=kline['symbol'],
                    timeframe=kline['timeframe'],
                    timestamp=kline['timestamp'],
                    open=kline['open'],
                    high=kline['high'],
                    low=kline['low'],
                    close=kline['close'],
                    volume=kline['volume'],
                    quote_volume=kline.get('quoteVolume'),
                    trades=kline.get('trades')
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching kline for {symbol} at {timestamp}: {str(e)}")
            return None

    def get_dividends(self, symbol: str) -> List[DividendData]:
        """Get dividend history for a symbol"""
        try:
            response = self.session.get(f"{self.base_url}/api/dividends/{symbol}", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                dividends = []
                for div in data['data']:
                    dividends.append(DividendData(
                        symbol=div['symbol'],
                        ex_date=div['ex_date'],
                        payment_date=div['payment_date'],
                        record_date=div['record_date'],
                        amount=div['amount'],
                        year=div['year']
                    ))
                return dividends
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching dividends for {symbol}: {str(e)}")
            return []

    # ============================================================================
    # WEBSOCKET METHODS
    # ============================================================================
    
    def connect_websocket(self) -> bool:
        """Connect to PSX Terminal WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={'sslopt': {'cert_reqs': ssl.CERT_NONE}}
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            max_wait = 10
            wait_time = 0
            while not self.is_connected and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5
            
            if self.is_connected:
                self.logger.info("✅ PSX Terminal WebSocket connected")
                return True
            else:
                self.logger.error("❌ PSX Terminal WebSocket connection timeout")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ WebSocket connection error: {str(e)}")
            return False

    def _on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        self.logger.info("🔗 WebSocket connection established")

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'welcome':
                self.client_id = data.get('clientId')
                self.logger.info(f"📨 Welcome message received, Client ID: {self.client_id}")
                
            elif message_type == 'ping':
                # Respond to ping with pong
                pong_msg = {
                    'type': 'pong',
                    'timestamp': data.get('timestamp')
                }
                self.ws.send(json.dumps(pong_msg))
                
            elif message_type == 'tickUpdate':
                # Handle real-time tick updates
                self._handle_tick_update(data)
                
            elif message_type == 'kline':
                # Handle k-line updates
                self._handle_kline_update(data)
                
            elif message_type == 'subscribeResponse':
                # Handle subscription responses
                self._handle_subscription_response(data)
                
            elif message_type == 'error':
                self.logger.error(f"❌ WebSocket error: {data.get('message')}")
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")

    def _on_error(self, ws, error):
        """WebSocket error handler"""
        self.logger.error(f"❌ WebSocket error: {str(error)}")

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        self.is_connected = False
        self.logger.info("🔌 WebSocket connection closed")

    def _handle_tick_update(self, data):
        """Handle real-time tick updates"""
        symbol = data.get('symbol')
        tick = data.get('tick')
        
        if symbol in self.callbacks:
            callback = self.callbacks[symbol]
            if callback:
                # Convert to MarketTick object
                market_tick = MarketTick(
                    symbol=tick['s'],
                    market=tick['m'],
                    status=tick['st'],
                    price=tick['c'],
                    change=tick['ch'],
                    change_percent=tick['pch'],
                    volume=tick['v'],
                    trades=tick['tr'],
                    value=tick['val'],
                    high=tick['h'],
                    low=tick['l'],
                    bid=tick['bp'],
                    ask=tick['ap'],
                    bid_volume=tick['bv'],
                    ask_volume=tick['av'],
                    timestamp=tick['t']
                )
                callback(market_tick)

    def _handle_kline_update(self, data):
        """Handle k-line updates"""
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        kline_data = data.get('data', [])
        
        callback_key = f"{symbol}_{timeframe}_kline"
        if callback_key in self.callbacks:
            callback = self.callbacks[callback_key]
            if callback:
                klines = []
                for kline in kline_data:
                    klines.append(KLineData(
                        symbol=kline['symbol'],
                        timeframe=kline.get('interval', timeframe),
                        timestamp=kline['timestamp'],
                        open=kline['open'],
                        high=kline['high'],
                        low=kline['low'],
                        close=kline['close'],
                        volume=kline['volume'],
                        quote_volume=kline.get('quoteVolume'),
                        trades=kline.get('trades')
                    ))
                callback(klines)

    def _handle_subscription_response(self, data):
        """Handle subscription responses"""
        request_id = data.get('requestId')
        status = data.get('status')
        subscription_key = data.get('subscriptionKey')
        
        if status == 'success':
            self.subscriptions[request_id] = subscription_key
            self.logger.info(f"✅ Subscription successful: {subscription_key}")
        else:
            self.logger.error(f"❌ Subscription failed for request: {request_id}")

    def subscribe_market_data(self, market_type: str = "REG", symbol: Optional[str] = None, 
                            callback: Optional[Callable] = None) -> str:
        """
        Subscribe to real-time market data
        
        Args:
            market_type: Market filter (REG, FUT, IDX, ODL, BNB, all)
            symbol: Specific symbol filter (optional)
            callback: Callback function for updates
            
        Returns:
            Request ID
        """
        if not self.is_connected:
            raise Exception("WebSocket not connected")
        
        request_id = f"market_data_{int(time.time() * 1000)}"
        
        params = {"marketType": market_type}
        if symbol:
            params["symbol"] = symbol
            
        subscribe_msg = {
            "type": "subscribe",
            "subscriptionType": "marketData", 
            "params": params,
            "requestId": request_id
        }
        
        if callback and symbol:
            self.callbacks[symbol] = callback
            
        self.ws.send(json.dumps(subscribe_msg))
        return request_id

    def subscribe_klines(self, symbol: str, timeframe: str = "1m", 
                        callback: Optional[Callable] = None) -> str:
        """
        Subscribe to k-line updates (2-step process)
        
        Args:
            symbol: Symbol name
            timeframe: Time interval
            callback: Callback function for updates
            
        Returns:
            Request ID
        """
        if not self.is_connected:
            raise Exception("WebSocket not connected")
        
        # Step 1: Request historical data
        request_id = f"kline_{symbol}_{timeframe}_{int(time.time() * 1000)}"
        
        kline_request = {
            "type": "klines",
            "symbol": symbol,
            "timeframe": timeframe,
            "requestId": request_id
        }
        
        self.ws.send(json.dumps(kline_request))
        
        # Step 2: Set up callback for real-time updates
        if callback:
            callback_key = f"{symbol}_{timeframe}_kline"
            self.callbacks[callback_key] = callback
            
            # Subscribe for real-time updates
            subscribe_msg = {
                "type": "subscribe",
                "subscriptionType": "klines",
                "params": {
                    "symbol": symbol,
                    "timeframe": timeframe
                },
                "requestId": f"{request_id}_subscribe"
            }
            self.ws.send(json.dumps(subscribe_msg))
        
        return request_id

    def unsubscribe(self, subscription_key: str) -> bool:
        """Unsubscribe from a stream"""
        if not self.is_connected:
            return False
        
        request_id = f"unsub_{int(time.time() * 1000)}"
        
        unsubscribe_msg = {
            "type": "unsubscribe",
            "subscriptionKey": subscription_key,
            "requestId": request_id
        }
        
        self.ws.send(json.dumps(unsubscribe_msg))
        return True

    def disconnect_websocket(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.is_connected = False
            self.ws.close()
            self.logger.info("🔌 WebSocket disconnected")

    # ============================================================================
    # CONVENIENCE METHODS FOR TRADING SYSTEM
    # ============================================================================
    
    def get_multiple_market_data(self, symbols: List[str], market_type: str = "REG") -> Dict[str, MarketTick]:
        """Get market data for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                tick = self.get_market_data(market_type, symbol)
                if tick:
                    results[symbol] = tick
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        return results

    def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        overview = {}
        
        try:
            # Get regular market stats
            reg_stats = self.get_market_stats("REG")
            if reg_stats:
                overview['regular_market'] = reg_stats
            
            # Get market breadth
            breadth = self.get_market_stats("breadth")
            if breadth:
                overview['breadth'] = breadth
            
            # Get sector stats
            sectors = self.get_market_stats("sectors")
            if sectors:
                overview['sectors'] = sectors
            
            # Get index data
            idx_stats = self.get_market_stats("IDX")
            if idx_stats:
                overview['indices'] = idx_stats
                
        except Exception as e:
            self.logger.error(f"Error getting market overview: {str(e)}")
        
        return overview

    def get_enhanced_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data for a single symbol"""
        data = {}
        
        try:
            # Market data
            tick = self.get_market_data("REG", symbol)
            if tick:
                data['market_data'] = asdict(tick)
            
            # Company info
            company = self.get_company_info(symbol)
            if company:
                data['company_info'] = company
            
            # Fundamentals
            fundamentals = self.get_fundamentals(symbol)
            if fundamentals:
                data['fundamentals'] = fundamentals
            
            # Dividends
            dividends = self.get_dividends(symbol)
            if dividends:
                data['dividends'] = [asdict(div) for div in dividends]
            
            # Recent k-lines (1 hour data for last 24 hours)
            klines = self.get_klines(symbol, "1h", limit=24)
            if klines:
                data['recent_klines'] = [asdict(kline) for kline in klines]
                
        except Exception as e:
            self.logger.error(f"Error getting enhanced data for {symbol}: {str(e)}")
        
        return data

    def convert_to_dataframe(self, klines: List[KLineData]) -> pd.DataFrame:
        """Convert k-line data to pandas DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        data = []
        for kline in klines:
            data.append({
                'timestamp': pd.to_datetime(kline.timestamp, unit='ms'),
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

# Example usage and testing
if __name__ == "__main__":
    # Initialize API
    api = PSXTerminalAPI()
    
    # Test connectivity
    status = api.test_connectivity()
    print(f"API Status: {status}")
    
    # Get symbols
    symbols = api.get_all_symbols()
    print(f"Total symbols: {len(symbols)}")
    print(f"First 10 symbols: {symbols[:10]}")
    
    # Test market data
    if symbols:
        test_symbol = 'HBL'  # Use a liquid symbol
        tick = api.get_market_data("REG", test_symbol)
        if tick:
            print(f"\n{test_symbol} Market Data:")
            print(f"Price: {tick.price}")
            print(f"Change: {tick.change} ({tick.change_percent:.2%})")
            print(f"Volume: {tick.volume:,}")
            print(f"High: {tick.high}, Low: {tick.low}")
    
    # Test k-lines
    klines = api.get_klines(test_symbol, "1h", limit=5)
    if klines:
        print(f"\nRecent K-lines for {test_symbol}:")
        for kline in klines[-3:]:
            print(f"{datetime.fromtimestamp(kline.timestamp/1000)}: "
                  f"O:{kline.open} H:{kline.high} L:{kline.low} C:{kline.close} V:{kline.volume}")
    
    # Test market overview
    overview = api.get_market_overview()
    if overview:
        print(f"\nMarket Overview:")
        if 'regular_market' in overview:
            stats = overview['regular_market']
            print(f"Total Volume: {stats.get('totalVolume', 0):,}")
            print(f"Gainers: {stats.get('gainers', 0)}")
            print(f"Losers: {stats.get('losers', 0)}")