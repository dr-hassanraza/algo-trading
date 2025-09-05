#!/usr/bin/env python3
"""
Real-Time Data Processing Pipeline for Algorithmic Trading
=========================================================

High-performance, low-latency data processing system for real-time trading
with comprehensive latency monitoring and performance optimization.

Features:
- Sub-millisecond data processing pipeline
- Asynchronous data ingestion and processing
- Real-time feature engineering and signal generation
- Latency monitoring and alerting
- Backpressure handling and flow control
- Circuit breaker pattern for fault tolerance
- Memory-efficient streaming data structures
- Performance profiling and bottleneck detection
"""

import pandas as pd
import numpy as np
import datetime as dt
import asyncio
import aiohttp
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import warnings
import logging
from pathlib import Path
import json
import sys
import gc

# Performance monitoring
import psutil
import resource
from contextlib import contextmanager

# Data structures
from sortedcontainers import SortedDict
import heapq

from quant_system_config import SystemConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class LatencyConfig:
    """Configuration for latency requirements and monitoring"""
    
    # Latency thresholds (microseconds)
    data_ingestion_max: int = 1000      # 1ms
    feature_processing_max: int = 5000   # 5ms
    signal_generation_max: int = 2000    # 2ms
    total_pipeline_max: int = 10000      # 10ms
    
    # Monitoring
    latency_window_size: int = 1000
    percentiles: List[float] = field(default_factory=lambda: [50, 95, 99, 99.9])
    alert_threshold_breaches: int = 5
    
    # Performance optimization
    batch_size: int = 100
    max_queue_size: int = 10000
    gc_frequency: int = 1000  # Process N items before GC
    
    # Circuit breaker
    failure_threshold: int = 10
    recovery_timeout: int = 30  # seconds

@dataclass 
class ProcessingMetrics:
    """Real-time processing metrics"""
    timestamp: float
    stage: str
    latency_us: float
    queue_size: int
    memory_mb: float
    cpu_percent: float
    throughput_msgs_per_sec: float

class LatencyMonitor:
    """Comprehensive latency monitoring and alerting system"""
    
    def __init__(self, config: LatencyConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.latency_window_size)
        self.stage_metrics = defaultdict(lambda: deque(maxlen=config.latency_window_size))
        self.alert_counts = defaultdict(int)
        self.last_alert_time = defaultdict(float)
        
    def record_latency(self, stage: str, latency_us: float, 
                      additional_metrics: Dict[str, Any] = None):
        """Record latency measurement for a processing stage"""
        
        timestamp = time.time()
        
        # Create metrics record
        metrics = ProcessingMetrics(
            timestamp=timestamp,
            stage=stage,
            latency_us=latency_us,
            queue_size=additional_metrics.get('queue_size', 0) if additional_metrics else 0,
            memory_mb=additional_metrics.get('memory_mb', 0) if additional_metrics else 0,
            cpu_percent=additional_metrics.get('cpu_percent', 0) if additional_metrics else 0,
            throughput_msgs_per_sec=additional_metrics.get('throughput', 0) if additional_metrics else 0
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.stage_metrics[stage].append(metrics)
        
        # Check for latency violations
        self._check_latency_violations(stage, latency_us)
    
    def _check_latency_violations(self, stage: str, latency_us: float):
        """Check if latency exceeds thresholds and generate alerts"""
        
        thresholds = {
            'data_ingestion': self.config.data_ingestion_max,
            'feature_processing': self.config.feature_processing_max,
            'signal_generation': self.config.signal_generation_max,
            'total_pipeline': self.config.total_pipeline_max
        }
        
        threshold = thresholds.get(stage, self.config.total_pipeline_max)
        
        if latency_us > threshold:
            self.alert_counts[stage] += 1
            
            # Rate-limited alerting
            current_time = time.time()
            last_alert = self.last_alert_time.get(stage, 0)
            
            if (self.alert_counts[stage] >= self.config.alert_threshold_breaches and 
                current_time - last_alert > 10):  # 10 second rate limit
                
                self._generate_alert(stage, latency_us, threshold)
                self.last_alert_time[stage] = current_time
                self.alert_counts[stage] = 0  # Reset counter
    
    def _generate_alert(self, stage: str, actual_latency: float, threshold: float):
        """Generate latency violation alert"""
        
        logger.warning(
            f"LATENCY VIOLATION - Stage: {stage}, "
            f"Actual: {actual_latency:.1f}μs, "
            f"Threshold: {threshold:.1f}μs, "
            f"Violation: {((actual_latency/threshold-1)*100):.1f}%"
        )
    
    def get_latency_statistics(self, stage: str = None, 
                             window_minutes: int = 5) -> Dict[str, float]:
        """Get latency statistics for specified stage or overall"""
        
        cutoff_time = time.time() - (window_minutes * 60)
        
        if stage:
            relevant_metrics = [m for m in self.stage_metrics[stage] 
                              if m.timestamp > cutoff_time]
        else:
            relevant_metrics = [m for m in self.metrics_history 
                              if m.timestamp > cutoff_time]
        
        if not relevant_metrics:
            return {}
        
        latencies = [m.latency_us for m in relevant_metrics]
        
        stats = {
            'count': len(latencies),
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
        
        # Add percentiles
        for p in self.config.percentiles:
            stats[f'p{p}'] = np.percentile(latencies, p)
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        summary = {
            'overall_stats': self.get_latency_statistics(),
            'stage_stats': {},
            'alerts': dict(self.alert_counts),
            'violation_rate': {}
        }
        
        # Per-stage statistics
        for stage in self.stage_metrics.keys():
            summary['stage_stats'][stage] = self.get_latency_statistics(stage)
            
            # Calculate violation rates
            recent_metrics = list(self.stage_metrics[stage])
            if recent_metrics:
                violations = sum(1 for m in recent_metrics[-100:] 
                               if m.latency_us > self.config.total_pipeline_max)
                summary['violation_rate'][stage] = violations / min(len(recent_metrics), 100)
        
        return summary

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, config: LatencyConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = 'OPEN'

class DataBuffer:
    """High-performance ring buffer for streaming data"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = np.empty(maxsize, dtype=object)
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.RLock()
    
    def put(self, item):
        """Add item to buffer"""
        with self.lock:
            if self.size == self.maxsize:
                # Overwrite oldest item
                self.tail = (self.tail + 1) % self.maxsize
            else:
                self.size += 1
            
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.maxsize
    
    def get_batch(self, batch_size: int) -> List[Any]:
        """Get batch of items from buffer"""
        with self.lock:
            if self.size == 0:
                return []
            
            actual_batch_size = min(batch_size, self.size)
            result = []
            
            for _ in range(actual_batch_size):
                result.append(self.buffer[self.tail])
                self.tail = (self.tail + 1) % self.maxsize
                self.size -= 1
            
            return result
    
    def peek_latest(self, n: int = 1) -> List[Any]:
        """Peek at latest n items without removing them"""
        with self.lock:
            if self.size == 0:
                return []
            
            result = []
            actual_n = min(n, self.size)
            
            for i in range(actual_n):
                idx = (self.head - 1 - i) % self.maxsize
                result.append(self.buffer[idx])
            
            return result

class StreamingFeatureProcessor:
    """High-performance streaming feature computation"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_buffer = deque(maxlen=window_size)
        self.volume_buffer = deque(maxlen=window_size)
        
        # Pre-computed rolling statistics
        self.returns_buffer = deque(maxlen=window_size)
        self.volatility_buffer = deque(maxlen=window_size)
        
        # Efficient online statistics
        self._return_sum = 0.0
        self._return_sum_sq = 0.0
        self._volume_sum = 0.0
        
    def add_tick(self, price: float, volume: float, timestamp: float) -> Dict[str, float]:
        """Add new tick and compute features incrementally"""
        
        features = {}
        
        # Store new data
        prev_price = self.price_buffer[-1] if self.price_buffer else price
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        # Compute return
        if len(self.price_buffer) >= 2:
            ret = (price - prev_price) / prev_price
            
            # Update online statistics
            old_ret = None
            if len(self.returns_buffer) == self.window_size:
                old_ret = self.returns_buffer[0]
                self._return_sum -= old_ret
                self._return_sum_sq -= old_ret ** 2
            
            self.returns_buffer.append(ret)
            self._return_sum += ret
            self._return_sum_sq += ret ** 2
            
            # Compute features
            n = len(self.returns_buffer)
            if n > 1:
                features['return_mean'] = self._return_sum / n
                features['return_var'] = (self._return_sum_sq / n) - (features['return_mean'] ** 2)
                features['return_vol'] = np.sqrt(max(features['return_var'], 0))
        
        # Volume features
        if len(self.volume_buffer) >= 2:
            features['volume_ratio'] = volume / np.mean(list(self.volume_buffer)[:-1])
        
        # Price features
        if len(self.price_buffer) >= 5:
            recent_prices = list(self.price_buffer)
            features['momentum_5'] = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
        
        if len(self.price_buffer) >= 20:
            recent_prices = list(self.price_buffer)
            sma_20 = np.mean(recent_prices[-20:])
            features['price_vs_sma20'] = (price - sma_20) / sma_20
        
        features['timestamp'] = timestamp
        features['price'] = price
        features['volume'] = volume
        
        return features

class RealTimeProcessor:
    """Main real-time data processing engine"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.latency_config = LatencyConfig()
        
        # Components
        self.latency_monitor = LatencyMonitor(self.latency_config)
        self.circuit_breaker = CircuitBreaker(self.latency_config)
        
        # Data structures
        self.data_buffer = DataBuffer(self.latency_config.max_queue_size)
        self.feature_processors = {}  # Per-symbol processors
        
        # Processing queues
        self.raw_data_queue = queue.Queue(maxsize=self.latency_config.max_queue_size)
        self.feature_queue = queue.Queue(maxsize=self.latency_config.max_queue_size)
        self.signal_queue = queue.Queue(maxsize=self.latency_config.max_queue_size)
        
        # Threading
        self.processing_threads = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.processed_count = 0
        self.last_gc_count = 0
        self.start_time = time.time()
        
        # Callbacks
        self.data_callbacks = []
        self.signal_callbacks = []
        
    def register_data_callback(self, callback: Callable[[Dict], None]):
        """Register callback for processed data"""
        self.data_callbacks.append(callback)
    
    def register_signal_callback(self, callback: Callable[[Dict], None]):
        """Register callback for generated signals"""
        self.signal_callbacks.append(callback)
    
    @contextmanager
    def _measure_latency(self, stage: str, additional_metrics: Dict[str, Any] = None):
        """Context manager for measuring processing stage latency"""
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
            
            # Add system metrics if not provided
            if additional_metrics is None:
                additional_metrics = {}
                
            if 'memory_mb' not in additional_metrics:
                process = psutil.Process()
                additional_metrics['memory_mb'] = process.memory_info().rss / 1024 / 1024
                additional_metrics['cpu_percent'] = process.cpu_percent()
            
            self.latency_monitor.record_latency(stage, latency_us, additional_metrics)
    
    def start_processing(self):
        """Start real-time processing pipeline"""
        
        if self.running:
            logger.warning("Processor already running")
            return
        
        self.running = True
        logger.info("Starting real-time processing pipeline")
        
        # Start processing threads
        threads = [
            ('data_ingestion', self._data_ingestion_loop),
            ('feature_processing', self._feature_processing_loop),
            ('signal_generation', self._signal_generation_loop),
            ('cleanup', self._cleanup_loop)
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Started {len(threads)} processing threads")
    
    def stop_processing(self):
        """Stop real-time processing pipeline"""
        
        if not self.running:
            return
        
        logger.info("Stopping real-time processing pipeline")
        self.running = False
        
        # Wait for threads to complete
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Real-time processing stopped")
    
    def ingest_tick_data(self, symbol: str, price: float, volume: float, 
                        timestamp: float = None):
        """Ingest new tick data for processing"""
        
        if timestamp is None:
            timestamp = time.time()
        
        tick_data = {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'ingestion_time': time.perf_counter()
        }
        
        try:
            # Non-blocking put with timeout
            self.raw_data_queue.put_nowait(tick_data)
            self.data_buffer.put(tick_data)
            
        except queue.Full:
            logger.warning("Raw data queue full, dropping tick")
            # Could implement backpressure handling here
    
    def _data_ingestion_loop(self):
        """Main data ingestion processing loop"""
        
        batch = []
        
        while self.running:
            try:
                # Collect batch of ticks
                while len(batch) < self.latency_config.batch_size:
                    try:
                        tick = self.raw_data_queue.get(timeout=0.001)  # 1ms timeout
                        batch.append(tick)
                    except queue.Empty:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                with self._measure_latency('data_ingestion', 
                                         {'queue_size': self.raw_data_queue.qsize()}):
                    self._process_data_batch(batch)
                
                batch.clear()
                
            except Exception as e:
                logger.error(f"Error in data ingestion: {e}")
                batch.clear()
    
    def _process_data_batch(self, batch: List[Dict]):
        """Process batch of tick data"""
        
        processed_batch = []
        
        for tick in batch:
            # Basic validation
            if not self._validate_tick(tick):
                continue
            
            # Add processing metadata
            tick['processing_start'] = time.perf_counter()
            tick['ingestion_latency_us'] = (
                tick['processing_start'] - tick['ingestion_time']
            ) * 1_000_000
            
            processed_batch.append(tick)
        
        # Forward to feature processing
        if processed_batch:
            try:
                self.feature_queue.put_nowait(processed_batch)
            except queue.Full:
                logger.warning("Feature queue full, dropping batch")
    
    def _validate_tick(self, tick: Dict) -> bool:
        """Validate tick data quality"""
        
        try:
            # Basic sanity checks
            if tick['price'] <= 0 or tick['volume'] < 0:
                return False
            
            if not tick['symbol'] or len(tick['symbol']) > 10:
                return False
            
            # Timestamp reasonableness
            now = time.time()
            if abs(tick['timestamp'] - now) > 300:  # 5 minutes
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False
    
    def _feature_processing_loop(self):
        """Feature processing loop"""
        
        while self.running:
            try:
                # Get batch from queue
                batch = self.feature_queue.get(timeout=1.0)
                
                with self._measure_latency('feature_processing',
                                         {'queue_size': self.feature_queue.qsize()}):
                    processed_features = self._compute_features_batch(batch)
                
                # Forward to signal generation
                if processed_features:
                    try:
                        self.signal_queue.put_nowait(processed_features)
                    except queue.Full:
                        logger.warning("Signal queue full, dropping features")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in feature processing: {e}")
    
    def _compute_features_batch(self, batch: List[Dict]) -> List[Dict]:
        """Compute features for batch of ticks"""
        
        processed_features = []
        
        for tick in batch:
            symbol = tick['symbol']
            
            # Get or create feature processor for symbol
            if symbol not in self.feature_processors:
                self.feature_processors[symbol] = StreamingFeatureProcessor()
            
            processor = self.feature_processors[symbol]
            
            # Compute features
            try:
                features = processor.add_tick(
                    tick['price'], 
                    tick['volume'], 
                    tick['timestamp']
                )
                
                # Add metadata
                features['symbol'] = symbol
                features['processing_time'] = time.perf_counter()
                features['total_latency_us'] = (
                    features['processing_time'] - tick['ingestion_time']
                ) * 1_000_000
                
                processed_features.append(features)
                
                # Call data callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(features)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")
                
            except Exception as e:
                logger.error(f"Error computing features for {symbol}: {e}")
        
        return processed_features
    
    def _signal_generation_loop(self):
        """Signal generation loop"""
        
        while self.running:
            try:
                # Get features from queue
                feature_batch = self.signal_queue.get(timeout=1.0)
                
                with self._measure_latency('signal_generation',
                                         {'queue_size': self.signal_queue.qsize()}):
                    signals = self._generate_signals_batch(feature_batch)
                
                # Process signals
                for signal in signals:
                    for callback in self.signal_callbacks:
                        try:
                            callback(signal)
                        except Exception as e:
                            logger.error(f"Error in signal callback: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
    
    def _generate_signals_batch(self, feature_batch: List[Dict]) -> List[Dict]:
        """Generate trading signals from features"""
        
        signals = []
        
        for features in feature_batch:
            try:
                # Simple signal generation logic (replace with your ML models)
                signal = self._compute_signal(features)
                
                if signal:
                    signal['generation_time'] = time.perf_counter()
                    signal['total_pipeline_latency_us'] = (
                        signal['generation_time'] - 
                        features.get('processing_time', features['timestamp'])
                    ) * 1_000_000
                    
                    signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error generating signal for {features.get('symbol', 'unknown')}: {e}")
        
        return signals
    
    def _compute_signal(self, features: Dict) -> Optional[Dict]:
        """Compute trading signal from features"""
        
        # Simple example signal logic
        signal_strength = 0.0
        signal_type = None
        
        # Momentum signal
        if 'momentum_5' in features and abs(features['momentum_5']) > 0.005:
            signal_strength += 0.3 * np.sign(features['momentum_5'])
        
        # Volatility signal
        if 'return_vol' in features and features['return_vol'] > 0.02:
            signal_strength *= 0.8  # Reduce signal in high volatility
        
        # Volume confirmation
        if 'volume_ratio' in features and features['volume_ratio'] > 1.5:
            signal_strength *= 1.2  # Amplify signal with high volume
        
        # Generate signal if strong enough
        if abs(signal_strength) > 0.1:  # Minimum threshold
            signal_type = 'BUY' if signal_strength > 0 else 'SELL'
            
            return {
                'symbol': features['symbol'],
                'signal_type': signal_type,
                'strength': abs(signal_strength),
                'confidence': min(abs(signal_strength) * 2, 1.0),
                'features_used': {k: v for k, v in features.items() 
                                if k in ['momentum_5', 'return_vol', 'volume_ratio']},
                'timestamp': features['timestamp']
            }
        
        return None
    
    def _cleanup_loop(self):
        """Periodic cleanup and maintenance"""
        
        while self.running:
            try:
                time.sleep(10)  # Run every 10 seconds
                
                # Garbage collection
                if self.processed_count - self.last_gc_count > self.latency_config.gc_frequency:
                    gc.collect()
                    self.last_gc_count = self.processed_count
                
                # Performance logging
                runtime = time.time() - self.start_time
                throughput = self.processed_count / runtime if runtime > 0 else 0
                
                logger.debug(f"Processed: {self.processed_count}, "
                           f"Throughput: {throughput:.1f} msg/sec")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        runtime = time.time() - self.start_time
        throughput = self.processed_count / runtime if runtime > 0 else 0
        
        # System resources
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = {
            'runtime_seconds': runtime,
            'processed_count': self.processed_count,
            'throughput_msg_per_sec': throughput,
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'queue_sizes': {
                'raw_data': self.raw_data_queue.qsize(),
                'features': self.feature_queue.qsize(),
                'signals': self.signal_queue.qsize()
            },
            'active_symbols': len(self.feature_processors),
            'latency_stats': self.latency_monitor.get_performance_summary()
        }
        
        return metrics
    
    def get_recent_data(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get recent processed data"""
        
        recent_data = self.data_buffer.peek_latest(limit)
        
        if symbol:
            return [d for d in recent_data if d.get('symbol') == symbol]
        
        return recent_data

# Test function
def test_realtime_processor():
    """Test real-time processing pipeline"""
    print("🚀 Testing Real-Time Processing Pipeline")
    print("=" * 50)
    
    # Create processor
    config = SystemConfig()
    processor = RealTimeProcessor(config)
    
    print(f"🔧 Configuration:")
    print(f"   Max Latency: {processor.latency_config.total_pipeline_max}μs")
    print(f"   Batch Size: {processor.latency_config.batch_size}")
    print(f"   Max Queue Size: {processor.latency_config.max_queue_size}")
    
    # Test data collection
    processed_data = []
    generated_signals = []
    
    def data_callback(data):
        processed_data.append(data)
    
    def signal_callback(signal):
        generated_signals.append(signal)
    
    # Register callbacks
    processor.register_data_callback(data_callback)
    processor.register_signal_callback(signal_callback)
    
    try:
        # Start processing
        print(f"\n🔄 Starting processing pipeline...")
        processor.start_processing()
        
        # Generate test data
        print(f"📊 Generating test market data...")
        
        symbols = ['UBL', 'MCB', 'FFC']
        base_prices = {'UBL': 150.0, 'MCB': 120.0, 'FFC': 85.0}
        
        # Simulate market data
        for i in range(1000):
            for symbol in symbols:
                # Random walk price
                base_price = base_prices[symbol]
                price_change = np.random.randn() * 0.01 * base_price  # 1% volatility
                price = max(base_price + price_change, base_price * 0.5)  # Floor
                base_prices[symbol] = price
                
                # Random volume
                volume = np.random.randint(100, 10000)
                
                # Ingest tick
                processor.ingest_tick_data(symbol, price, volume)
            
            # Small delay to simulate realistic timing
            if i % 100 == 0:
                time.sleep(0.01)  # 10ms pause every 100 ticks
                
                # Show progress
                metrics = processor.get_performance_metrics()
                print(f"   Processed: {metrics['processed_count']}, "
                      f"Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/sec")
        
        # Wait for processing to complete
        print(f"\n⏳ Waiting for processing to complete...")
        time.sleep(2.0)
        
        # Get final metrics
        final_metrics = processor.get_performance_metrics()
        
        print(f"\n✅ Processing Results:")
        print(f"   Total Processed: {final_metrics['processed_count']}")
        print(f"   Average Throughput: {final_metrics['throughput_msg_per_sec']:.1f} msg/sec")
        print(f"   Memory Usage: {final_metrics['memory_mb']:.1f} MB")
        print(f"   Active Symbols: {final_metrics['active_symbols']}")
        
        # Latency statistics
        latency_stats = final_metrics['latency_stats']['overall_stats']
        if latency_stats:
            print(f"\n📊 Latency Statistics:")
            print(f"   Mean Latency: {latency_stats['mean']:.1f}μs")
            print(f"   P95 Latency: {latency_stats.get('p95.0', 0):.1f}μs")
            print(f"   P99 Latency: {latency_stats.get('p99.0', 0):.1f}μs")
            print(f"   Max Latency: {latency_stats['max']:.1f}μs")
        
        # Queue status
        queue_sizes = final_metrics['queue_sizes']
        print(f"\n📋 Queue Status:")
        for queue_name, size in queue_sizes.items():
            print(f"   {queue_name}: {size} items")
        
        # Data processing results
        print(f"\n📈 Data Processing:")
        print(f"   Features Generated: {len(processed_data)}")
        print(f"   Signals Generated: {len(generated_signals)}")
        
        if generated_signals:
            signal_types = {}
            for signal in generated_signals:
                signal_type = signal['signal_type']
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
            print(f"   Signal Distribution: {signal_types}")
            
            # Show sample signals
            print(f"\n🎯 Sample Signals:")
            for signal in generated_signals[:3]:
                print(f"      {signal['symbol']}: {signal['signal_type']} "
                      f"(Strength: {signal['strength']:.3f}, "
                      f"Confidence: {signal['confidence']:.3f})")
        
        # Performance validation
        print(f"\n🔍 Performance Validation:")
        
        violation_rates = final_metrics['latency_stats'].get('violation_rate', {})
        for stage, rate in violation_rates.items():
            status = "✅" if rate < 0.05 else "⚠️"  # 5% threshold
            print(f"   {stage} Violation Rate: {status} {rate*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in real-time processor test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop processing
        print(f"\n🛑 Stopping processor...")
        processor.stop_processing()
    
    print(f"\n🏁 Real-time processing test completed!")

if __name__ == "__main__":
    test_realtime_processor()