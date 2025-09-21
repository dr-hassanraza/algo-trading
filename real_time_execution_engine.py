"""
REAL-TIME EXECUTION ENGINE WITH SLIPPAGE MODELING
Advanced Order Execution System for High-Accuracy Intraday Trading

Features:
- Real-time order execution with market microstructure awareness
- Advanced slippage modeling and cost analysis
- Smart order routing and execution algorithms
- Latency optimization and timing analysis
- PSX-specific execution adaptations
- Position management and portfolio reconciliation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import asyncio
import threading
import queue
import time as time_module
import warnings
from enum import Enum
import json

warnings.filterwarnings('ignore')

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order container with all execution details"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    
    # Execution details
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    remaining_quantity: int = 0
    
    # Timestamps
    created_time: datetime = None
    submitted_time: datetime = None
    first_fill_time: datetime = None
    last_fill_time: datetime = None
    
    # Execution quality
    slippage: float = 0.0
    implementation_shortfall: float = 0.0
    market_impact: float = 0.0
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

@dataclass
class Fill:
    """Fill/execution details"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    
    # Execution quality metrics
    effective_spread: float = 0.0
    price_improvement: float = 0.0
    market_impact: float = 0.0

@dataclass
class MarketData:
    """Real-time market data snapshot"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    volume: int
    
    # Microstructure data
    spread: float = 0.0
    mid_price: float = 0.0
    
    def __post_init__(self):
        self.spread = self.ask - self.bid
        self.mid_price = (self.bid + self.ask) / 2

class SlippageModel:
    """Advanced slippage modeling for order execution"""
    
    def __init__(self):
        # PSX-specific parameters (calibrated from historical data)
        self.psx_params = {
            'base_spread_bps': 15,      # Base spread in basis points
            'impact_coeff': 0.5,        # Market impact coefficient
            'volume_decay': 0.7,        # Volume impact decay
            'volatility_multiplier': 1.2, # Volatility impact multiplier
            'session_multipliers': {
                'opening': 1.8,         # Higher slippage during opening
                'midday': 1.0,          # Normal slippage midday
                'lunch': 1.4,           # Higher slippage during lunch
                'closing': 1.6          # Higher slippage during closing
            }
        }
        
        # Liquidity tiers and their impact
        self.liquidity_tiers = {
            'tier_1': {'min_volume': 100000, 'impact_multiplier': 0.8},  # High liquidity
            'tier_2': {'min_volume': 50000, 'impact_multiplier': 1.0},   # Medium liquidity
            'tier_3': {'min_volume': 10000, 'impact_multiplier': 1.5},   # Low liquidity
            'tier_4': {'min_volume': 0, 'impact_multiplier': 2.5}        # Very low liquidity
        }
    
    def estimate_slippage(self, order: Order, market_data: MarketData, 
                         historical_volume: float, current_volatility: float) -> Dict[str, float]:
        """Estimate expected slippage for order execution"""
        
        # Base spread component
        spread_slippage = market_data.spread / 2  # Half spread as base cost
        
        # Market impact component
        participation_rate = order.quantity / historical_volume if historical_volume > 0 else 0.1
        impact_slippage = self._calculate_market_impact(order, participation_rate, current_volatility)
        
        # Timing impact (session-dependent)
        timing_multiplier = self._get_session_multiplier()
        
        # Liquidity tier adjustment
        liquidity_multiplier = self._get_liquidity_multiplier(historical_volume)
        
        # Order size adjustment
        size_multiplier = self._get_size_multiplier(order.quantity, market_data)
        
        # Volatility adjustment
        vol_multiplier = 1 + (current_volatility - 0.02) * self.psx_params['volatility_multiplier']
        vol_multiplier = max(0.5, min(3.0, vol_multiplier))  # Cap between 0.5x and 3x
        
        # Total expected slippage
        total_slippage = (spread_slippage + impact_slippage) * timing_multiplier * liquidity_multiplier * size_multiplier * vol_multiplier
        
        return {
            'spread_component': spread_slippage,
            'impact_component': impact_slippage,
            'timing_multiplier': timing_multiplier,
            'liquidity_multiplier': liquidity_multiplier,
            'size_multiplier': size_multiplier,
            'volatility_multiplier': vol_multiplier,
            'total_expected_slippage': total_slippage,
            'confidence_interval': (total_slippage * 0.7, total_slippage * 1.3)
        }
    
    def _calculate_market_impact(self, order: Order, participation_rate: float, volatility: float) -> float:
        """Calculate market impact based on order size and market conditions"""
        
        # Square root impact model (common in academic literature)
        impact = self.psx_params['impact_coeff'] * volatility * np.sqrt(participation_rate)
        
        # Adjust for order side (sells typically have higher impact)
        if order.side == 'SELL':
            impact *= 1.1
        
        return impact
    
    def _get_session_multiplier(self) -> float:
        """Get session-based slippage multiplier"""
        
        current_time = datetime.now().time()
        
        if time(9, 15) <= current_time <= time(10, 0):
            return self.psx_params['session_multipliers']['opening']
        elif time(12, 30) <= current_time <= time(13, 30):
            return self.psx_params['session_multipliers']['lunch']
        elif time(15, 0) <= current_time <= time(15, 30):
            return self.psx_params['session_multipliers']['closing']
        else:
            return self.psx_params['session_multipliers']['midday']
    
    def _get_liquidity_multiplier(self, historical_volume: float) -> float:
        """Get liquidity-based multiplier"""
        
        for tier, params in self.liquidity_tiers.items():
            if historical_volume >= params['min_volume']:
                return params['impact_multiplier']
        
        return self.liquidity_tiers['tier_4']['impact_multiplier']
    
    def _get_size_multiplier(self, quantity: int, market_data: MarketData) -> float:
        """Get size-based multiplier"""
        
        # Compare order size to available liquidity
        available_liquidity = market_data.bid_size if quantity > 0 else market_data.ask_size
        
        if available_liquidity == 0:
            return 2.0  # High impact if no visible liquidity
        
        size_ratio = abs(quantity) / available_liquidity
        
        if size_ratio <= 0.5:
            return 1.0      # Normal impact
        elif size_ratio <= 1.0:
            return 1.3      # Moderate impact
        elif size_ratio <= 2.0:
            return 1.8      # High impact
        else:
            return 2.5      # Very high impact

class RealTimeExecutionEngine:
    """Advanced real-time execution engine"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Core components
        self.slippage_model = SlippageModel()
        
        # Order management
        self.orders = {}            # Active orders
        self.fills = {}             # Execution fills
        self.positions = {}         # Current positions
        
        # Execution queues
        self.order_queue = queue.Queue()
        self.market_data_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = {}
        
        # Execution metrics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_time': 0.0,
            'total_slippage': 0.0,
            'avg_slippage': 0.0
        }
        
        # Risk controls
        self.risk_controls = {
            'max_order_value': 100000,     # Maximum order value
            'max_position_size': 200000,   # Maximum position size
            'daily_loss_limit': 50000,     # Daily loss limit
            'max_orders_per_minute': 10,   # Order rate limit
        }
        
        # PSX trading hours
        self.trading_hours = {
            'start': time(9, 15),
            'end': time(15, 30),
            'lunch_start': time(12, 30),
            'lunch_end': time(13, 30)
        }
        
        # Execution threads
        self.execution_thread = None
        self.market_data_thread = None
        self.running = False
        
    def start_execution_engine(self):
        """Start the real-time execution engine"""
        
        if self.running:
            print("Execution engine already running")
            return
        
        print("🚀 Starting Real-Time Execution Engine...")
        self.running = True
        
        # Start execution threads
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.market_data_thread = threading.Thread(target=self._market_data_loop, daemon=True)
        
        self.execution_thread.start()
        self.market_data_thread.start()
        
        print("✅ Execution engine started successfully")
    
    def stop_execution_engine(self):
        """Stop the execution engine"""
        
        print("🛑 Stopping execution engine...")
        self.running = False
        
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        if self.market_data_thread:
            self.market_data_thread.join(timeout=5)
        
        print("✅ Execution engine stopped")
    
    def submit_order(self, order: Order) -> bool:
        """Submit order for execution"""
        
        # Pre-execution validation
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.execution_stats['rejected_orders'] += 1
            return False
        
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.orders)}"
        
        # Add to order management
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.now()
        self.orders[order.order_id] = order
        
        # Queue for execution
        self.order_queue.put(order)
        self.execution_stats['total_orders'] += 1
        
        print(f"📝 Order submitted: {order.order_id} - {order.side} {order.quantity} {order.symbol}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        self.execution_stats['cancelled_orders'] += 1
        
        print(f"❌ Order cancelled: {order_id}")
        return True
    
    def update_market_data(self, symbol: str, market_data: MarketData):
        """Update real-time market data"""
        
        self.market_data_cache[symbol] = market_data
        self.last_market_update[symbol] = datetime.now()
        
        # Queue for processing
        self.market_data_queue.put((symbol, market_data))
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        return self.orders.get(order_id)
    
    def get_position(self, symbol: str) -> Dict[str, Union[int, float]]:
        """Get current position for symbol"""
        
        position = self.positions.get(symbol, {
            'quantity': 0,
            'avg_price': 0.0,
            'market_value': 0.0,
            'unrealized_pnl': 0.0
        })
        
        # Update market value if we have current market data
        if symbol in self.market_data_cache and position['quantity'] != 0:
            current_price = self.market_data_cache[symbol].last_price
            position['market_value'] = position['quantity'] * current_price
            position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_price'])
        
        return position
    
    def get_execution_quality_report(self) -> Dict[str, Union[float, int]]:
        """Generate execution quality report"""
        
        if self.execution_stats['filled_orders'] == 0:
            return self.execution_stats.copy()
        
        # Calculate averages
        total_fills = len(self.fills)
        if total_fills > 0:
            avg_slippage = sum(fill.market_impact for fill in self.fills.values()) / total_fills
            self.execution_stats['avg_slippage'] = avg_slippage
        
        # Add fill rate
        self.execution_stats['fill_rate'] = self.execution_stats['filled_orders'] / self.execution_stats['total_orders'] if self.execution_stats['total_orders'] > 0 else 0
        
        return self.execution_stats.copy()
    
    def _execution_loop(self):
        """Main execution loop"""
        
        while self.running:
            try:
                # Process pending orders
                if not self.order_queue.empty():
                    order = self.order_queue.get(timeout=1)
                    self._process_order(order)
                
                # Check for order updates
                self._check_order_timeouts()
                
                time_module.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in execution loop: {e}")
    
    def _market_data_loop(self):
        """Market data processing loop"""
        
        while self.running:
            try:
                if not self.market_data_queue.empty():
                    symbol, market_data = self.market_data_queue.get(timeout=1)
                    self._process_market_data(symbol, market_data)
                
                time_module.sleep(0.05)  # Faster loop for market data
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in market data loop: {e}")
    
    def _process_order(self, order: Order):
        """Process individual order for execution"""
        
        # Check if market is open
        if not self._is_market_open():
            print(f"⚠️ Market closed, order {order.order_id} queued for next session")
            return
        
        # Get current market data
        if order.symbol not in self.market_data_cache:
            print(f"⚠️ No market data for {order.symbol}, order {order.order_id} pending")
            return
        
        market_data = self.market_data_cache[order.symbol]
        
        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order, market_data)
        elif order.order_type == OrderType.LIMIT:
            self._execute_limit_order(order, market_data)
        elif order.order_type == OrderType.STOP:
            self._execute_stop_order(order, market_data)
        elif order.order_type == OrderType.STOP_LIMIT:
            self._execute_stop_limit_order(order, market_data)
    
    def _execute_market_order(self, order: Order, market_data: MarketData):
        """Execute market order immediately"""
        
        # Determine execution price
        if order.side == 'BUY':
            execution_price = market_data.ask
            available_quantity = market_data.ask_size
        else:
            execution_price = market_data.bid
            available_quantity = market_data.bid_size
        
        # Calculate slippage
        slippage_info = self.slippage_model.estimate_slippage(
            order, market_data, 
            historical_volume=50000,  # Would come from historical data
            current_volatility=0.02   # Would be calculated from recent returns
        )
        
        # Apply slippage to execution price
        slippage_amount = slippage_info['total_expected_slippage']
        if order.side == 'BUY':
            execution_price += slippage_amount
        else:
            execution_price -= slippage_amount
        
        # Determine fill quantity
        fill_quantity = min(order.remaining_quantity, available_quantity, order.quantity)
        
        if fill_quantity > 0:
            # Create fill
            fill = Fill(
                fill_id=f"FILL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=execution_price,
                timestamp=datetime.now(),
                market_impact=slippage_amount
            )
            
            # Update order
            self._update_order_fill(order, fill)
            
            # Update position
            self._update_position(order.symbol, order.side, fill_quantity, execution_price)
            
            print(f"✅ Market order executed: {fill.quantity} {fill.symbol} @ {fill.price:.2f}")
    
    def _execute_limit_order(self, order: Order, market_data: MarketData):
        """Execute limit order if price is favorable"""
        
        can_execute = False
        
        if order.side == 'BUY' and market_data.ask <= order.price:
            can_execute = True
            execution_price = min(order.price, market_data.ask)
        elif order.side == 'SELL' and market_data.bid >= order.price:
            can_execute = True
            execution_price = max(order.price, market_data.bid)
        
        if can_execute:
            # Execute similar to market order but at limit price
            fill_quantity = min(order.remaining_quantity, order.quantity)
            
            fill = Fill(
                fill_id=f"FILL_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=execution_price,
                timestamp=datetime.now(),
                price_improvement=abs(execution_price - order.price) if order.price else 0
            )
            
            self._update_order_fill(order, fill)
            self._update_position(order.symbol, order.side, fill_quantity, execution_price)
            
            print(f"✅ Limit order executed: {fill.quantity} {fill.symbol} @ {fill.price:.2f}")
    
    def _execute_stop_order(self, order: Order, market_data: MarketData):
        """Execute stop order when stop price is triggered"""
        
        triggered = False
        
        if order.side == 'BUY' and market_data.last_price >= order.stop_price:
            triggered = True
        elif order.side == 'SELL' and market_data.last_price <= order.stop_price:
            triggered = True
        
        if triggered:
            # Convert to market order
            order.order_type = OrderType.MARKET
            self._execute_market_order(order, market_data)
    
    def _execute_stop_limit_order(self, order: Order, market_data: MarketData):
        """Execute stop-limit order"""
        
        # Check if stop is triggered first
        triggered = False
        
        if order.side == 'BUY' and market_data.last_price >= order.stop_price:
            triggered = True
        elif order.side == 'SELL' and market_data.last_price <= order.stop_price:
            triggered = True
        
        if triggered:
            # Convert to limit order
            order.order_type = OrderType.LIMIT
            self._execute_limit_order(order, market_data)
    
    def _update_order_fill(self, order: Order, fill: Fill):
        """Update order with fill information"""
        
        # Add fill to records
        self.fills[fill.fill_id] = fill
        
        # Update order
        order.filled_quantity += fill.quantity
        order.remaining_quantity -= fill.quantity
        
        # Calculate average fill price
        if order.filled_quantity > 0:
            total_value = order.avg_fill_price * (order.filled_quantity - fill.quantity) + fill.price * fill.quantity
            order.avg_fill_price = total_value / order.filled_quantity
        
        # Update timestamps
        if order.first_fill_time is None:
            order.first_fill_time = fill.timestamp
        order.last_fill_time = fill.timestamp
        
        # Update status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            self.execution_stats['filled_orders'] += 1
        else:
            order.status = OrderStatus.PARTIAL_FILLED
        
        # Calculate execution quality metrics
        order.slippage = abs(fill.price - order.price) if order.price else 0
        order.market_impact = fill.market_impact
    
    def _update_position(self, symbol: str, side: str, quantity: int, price: float):
        """Update position after fill"""
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'realized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        if side == 'BUY':
            # Add to position
            new_total_cost = position['total_cost'] + quantity * price
            new_quantity = position['quantity'] + quantity
            
            if new_quantity > 0:
                position['avg_price'] = new_total_cost / new_quantity
            
            position['quantity'] = new_quantity
            position['total_cost'] = new_total_cost
            
        else:  # SELL
            # Reduce position
            if position['quantity'] > 0:
                # Calculate realized P&L
                realized_pnl = quantity * (price - position['avg_price'])
                position['realized_pnl'] += realized_pnl
                
                # Update position
                position['quantity'] -= quantity
                position['total_cost'] -= quantity * position['avg_price']
                
                if position['quantity'] == 0:
                    position['avg_price'] = 0.0
                    position['total_cost'] = 0.0
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before execution"""
        
        # Basic validation
        if order.quantity <= 0:
            print(f"❌ Invalid quantity: {order.quantity}")
            return False
        
        if order.symbol not in self.market_data_cache:
            print(f"❌ No market data for symbol: {order.symbol}")
            return False
        
        # Risk checks
        market_data = self.market_data_cache[order.symbol]
        order_value = order.quantity * market_data.last_price
        
        if order_value > self.risk_controls['max_order_value']:
            print(f"❌ Order value exceeds limit: {order_value}")
            return False
        
        # Position size check
        current_position = self.get_position(order.symbol)
        new_position_value = abs(current_position['quantity'] * market_data.last_price + order_value)
        
        if new_position_value > self.risk_controls['max_position_size']:
            print(f"❌ Position size would exceed limit: {new_position_value}")
            return False
        
        return True
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        
        current_time = datetime.now().time()
        current_day = datetime.now().weekday()
        
        # Check if it's a trading day (Monday to Friday)
        if current_day >= 5:  # Saturday or Sunday
            return False
        
        # Check trading hours
        if self.trading_hours['start'] <= current_time <= self.trading_hours['end']:
            # Check if it's not lunch time
            if not (self.trading_hours['lunch_start'] <= current_time <= self.trading_hours['lunch_end']):
                return True
        
        return False
    
    def _check_order_timeouts(self):
        """Check for order timeouts and expiry"""
        
        current_time = datetime.now()
        
        for order_id, order in list(self.orders.items()):
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                continue
            
            # Check time in force
            if order.time_in_force == "DAY":
                # Cancel at end of trading day
                if current_time.time() >= self.trading_hours['end']:
                    self.cancel_order(order_id)
            
            # Check for stale orders (more than 1 hour old)
            if (current_time - order.created_time).total_seconds() > 3600:
                print(f"⚠️ Stale order detected: {order_id}")
    
    def _process_market_data(self, symbol: str, market_data: MarketData):
        """Process incoming market data updates"""
        
        # Update any pending limit or stop orders for this symbol
        for order_id, order in self.orders.items():
            if (order.symbol == symbol and 
                order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]):
                
                if order.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]:
                    # Re-queue for execution check
                    try:
                        self.order_queue.put(order, block=False)
                    except queue.Full:
                        pass  # Queue is full, will be processed later

# Testing function
def test_execution_engine():
    """Test the real-time execution engine"""
    print("🧪 Testing Real-Time Execution Engine...")
    
    # Initialize engine
    engine = RealTimeExecutionEngine()
    
    # Start engine
    engine.start_execution_engine()
    
    # Create sample market data
    market_data = MarketData(
        symbol='HBL',
        timestamp=datetime.now(),
        bid=100.50,
        ask=100.60,
        bid_size=1000,
        ask_size=1500,
        last_price=100.55,
        last_size=500,
        volume=50000
    )
    
    # Update market data
    engine.update_market_data('HBL', market_data)
    
    # Test market order
    print("📊 Testing market order...")
    market_order = Order(
        order_id="TEST_001",
        symbol='HBL',
        side='BUY',
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    success = engine.submit_order(market_order)
    print(f"   Order submitted: {success}")
    
    # Wait for execution
    time_module.sleep(1)
    
    # Check order status
    order_status = engine.get_order_status("TEST_001")
    if order_status:
        print(f"   Order Status: {order_status.status}")
        print(f"   Filled Quantity: {order_status.filled_quantity}")
        print(f"   Avg Fill Price: {order_status.avg_fill_price:.2f}")
    
    # Test limit order
    print("\n📊 Testing limit order...")
    limit_order = Order(
        order_id="TEST_002",
        symbol='HBL',
        side='BUY',
        quantity=200,
        order_type=OrderType.LIMIT,
        price=100.45  # Below current ask
    )
    
    engine.submit_order(limit_order)
    time_module.sleep(0.5)
    
    # Check position
    print("\n💼 Checking position...")
    position = engine.get_position('HBL')
    print(f"   Quantity: {position['quantity']}")
    print(f"   Avg Price: {position['avg_price']:.2f}")
    print(f"   Market Value: {position['market_value']:.2f}")
    
    # Get execution quality report
    print("\n📈 Execution Quality Report:")
    quality_report = engine.get_execution_quality_report()
    for key, value in quality_report.items():
        print(f"   {key}: {value}")
    
    # Stop engine
    engine.stop_execution_engine()
    
    print("\n✅ Real-Time Execution Engine test completed!")

if __name__ == "__main__":
    test_execution_engine()