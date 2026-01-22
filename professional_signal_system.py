"""
Professional Trading Signal System
===================================
Investment-grade signal generation with:
- Backtesting validation
- Signal tracking & performance history
- Multi-timeframe confirmation
- Advanced risk management

Author: PSX Trading System
Version: 2.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os
import requests


@dataclass
class SignalRecord:
    """Record of a generated signal for tracking"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: str
    reasons: List[str]
    timeframes_aligned: int  # How many timeframes agreed
    # Outcome tracking (filled later)
    outcome: str = "PENDING"  # PENDING, WIN, LOSS, EXPIRED
    exit_price: float = 0.0
    exit_timestamp: str = ""
    pnl_percent: float = 0.0


@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    equity_curve: List[float]
    trades: List[Dict]


class PSXDataFetcher:
    """Fetch historical and real-time data from PSX"""

    def __init__(self):
        self.base_url = "https://dps.psx.com.pk"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PSX-Professional-System/2.0',
            'Accept': 'application/json'
        })

    def get_intraday_data(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fetch intraday tick data"""
        try:
            url = f"{self.base_url}/timeseries/int/{symbol}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'], columns=['timestamp', 'price', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                    df = df.dropna().sort_values('timestamp').reset_index(drop=True)
                    return df.tail(limit)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

    def get_ohlcv_from_ticks(self, df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """Convert tick data to OHLCV candles"""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.set_index('timestamp', inplace=True)

        ohlcv = df['price'].resample(timeframe).ohlc()
        ohlcv['volume'] = df['volume'].resample(timeframe).sum()
        ohlcv = ohlcv.dropna().reset_index()
        ohlcv.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        return ohlcv


class TechnicalAnalyzer:
    """Calculate technical indicators"""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 30:
            return df

        df = df.copy()
        price = df[price_col]

        # Moving Averages
        df['sma_5'] = price.rolling(5).mean()
        df['sma_10'] = price.rolling(10).mean()
        df['sma_20'] = price.rolling(20).mean()
        df['sma_50'] = price.rolling(50).mean() if len(df) >= 50 else price.rolling(20).mean()
        df['ema_12'] = price.ewm(span=12).mean()
        df['ema_26'] = price.ewm(span=26).mean()

        # RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = price.rolling(20).mean()
        bb_std = price.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range)
        if 'high' in df.columns and 'low' in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - price.shift())
            low_close = abs(df['low'] - price.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_percent'] = (df['atr'] / price) * 100
        else:
            # Estimate ATR from price volatility
            df['atr'] = price.rolling(14).std() * 1.5
            df['atr_percent'] = (df['atr'] / price) * 100

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Momentum
        df['momentum'] = price.pct_change(10) * 100
        df['roc'] = ((price - price.shift(10)) / price.shift(10)) * 100

        # Stochastic
        low_14 = price.rolling(14).min()
        high_14 = price.rolling(14).max()
        df['stoch_k'] = ((price - low_14) / (high_14 - low_14)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ADX (Simplified)
        df['adx'] = abs(df['momentum']).rolling(14).mean()

        return df


class MultiTimeframeAnalyzer:
    """Analyze signals across multiple timeframes"""

    def __init__(self, data_fetcher: PSXDataFetcher):
        self.fetcher = data_fetcher
        self.analyzer = TechnicalAnalyzer()

    def analyze_timeframe(self, df: pd.DataFrame) -> Dict:
        """Analyze a single timeframe and return signal components"""
        if df.empty or len(df) < 20:
            return {'trend': 0, 'momentum': 0, 'signal': 'NEUTRAL', 'strength': 0}

        df = self.analyzer.calculate_all_indicators(df)
        latest = df.iloc[-1]

        score = 0

        # Trend Analysis
        if latest['sma_5'] > latest['sma_20']:
            score += 2
        else:
            score -= 2

        # RSI
        rsi = latest['rsi']
        if rsi < 30:
            score += 2
        elif rsi < 40:
            score += 1
        elif rsi > 70:
            score -= 2
        elif rsi > 60:
            score -= 1

        # MACD
        if latest['macd'] > latest['macd_signal']:
            score += 1
        else:
            score -= 1

        # Determine signal
        if score >= 3:
            signal = 'STRONG_BUY'
        elif score >= 1:
            signal = 'BUY'
        elif score <= -3:
            signal = 'STRONG_SELL'
        elif score <= -1:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        return {
            'trend': 1 if latest['sma_5'] > latest['sma_20'] else -1,
            'momentum': score,
            'signal': signal,
            'strength': abs(score),
            'rsi': rsi,
            'price': latest['close'] if 'close' in latest else latest['price']
        }

    def get_multi_timeframe_signal(self, symbol: str) -> Dict:
        """Get signal confirmed across multiple timeframes"""
        # Fetch raw data
        raw_data = self.fetcher.get_intraday_data(symbol, limit=1000)

        if raw_data.empty:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'timeframes_aligned': 0,
                'analysis': {}
            }

        # Analyze different timeframes
        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1H': '1H'
        }

        analyses = {}
        for tf_name, tf_resample in timeframes.items():
            try:
                ohlcv = self.fetcher.get_ohlcv_from_ticks(raw_data, tf_resample)
                if not ohlcv.empty and len(ohlcv) >= 20:
                    analyses[tf_name] = self.analyze_timeframe(ohlcv)
            except:
                continue

        if not analyses:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'timeframes_aligned': 0,
                'analysis': {}
            }

        # Count aligned signals
        buy_count = sum(1 for a in analyses.values() if 'BUY' in a['signal'])
        sell_count = sum(1 for a in analyses.values() if 'SELL' in a['signal'])
        total_timeframes = len(analyses)

        # Determine final signal based on alignment
        if buy_count >= 3:
            signal = 'STRONG_BUY'
            confidence = min(85, 50 + (buy_count * 10))
            aligned = buy_count
        elif buy_count >= 2:
            signal = 'BUY'
            confidence = min(70, 40 + (buy_count * 10))
            aligned = buy_count
        elif sell_count >= 3:
            signal = 'STRONG_SELL'
            confidence = min(85, 50 + (sell_count * 10))
            aligned = sell_count
        elif sell_count >= 2:
            signal = 'SELL'
            confidence = min(70, 40 + (sell_count * 10))
            aligned = sell_count
        else:
            signal = 'HOLD'
            confidence = 30
            aligned = 0

        # Get current price
        current_price = list(analyses.values())[-1].get('price', 0)

        return {
            'signal': signal,
            'confidence': confidence,
            'timeframes_aligned': aligned,
            'total_timeframes': total_timeframes,
            'analysis': analyses,
            'price': current_price
        }


class BacktestEngine:
    """Backtest trading signals against historical data"""

    def __init__(self, data_fetcher: PSXDataFetcher):
        self.fetcher = data_fetcher
        self.analyzer = TechnicalAnalyzer()

    def generate_historical_signals(self, df: pd.DataFrame,
                                    stop_loss_pct: float = 0.05,
                                    take_profit_pct: float = 0.05) -> List[Dict]:
        """Generate signals for historical data"""
        if df.empty or len(df) < 50:
            return []

        df = self.analyzer.calculate_all_indicators(df)
        signals = []

        for i in range(30, len(df) - 10):  # Leave room for outcome
            row = df.iloc[i]
            prev = df.iloc[i-1]

            score = 0
            reasons = []

            # RSI
            if row['rsi'] < 30:
                score += 2
                reasons.append("RSI oversold")
            elif row['rsi'] > 70:
                score -= 2
                reasons.append("RSI overbought")

            # Trend
            if row['sma_5'] > row['sma_20']:
                score += 1
                reasons.append("Bullish trend")
            else:
                score -= 1
                reasons.append("Bearish trend")

            # MACD crossover
            if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                score += 2
                reasons.append("MACD bullish crossover")
            elif row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                score -= 2
                reasons.append("MACD bearish crossover")

            # Volume confirmation
            if row['volume_ratio'] > 1.5:
                if score > 0:
                    score += 1
                elif score < 0:
                    score -= 1
                reasons.append("High volume")

            # Generate signal
            price = row['close'] if 'close' in row else row['price']

            if score >= 3:
                signal_type = 'BUY'
                stop = price * (1 - stop_loss_pct)
                target = price * (1 + take_profit_pct)
            elif score <= -3:
                signal_type = 'SELL'
                stop = price * (1 + stop_loss_pct)
                target = price * (1 - take_profit_pct)
            else:
                continue  # Skip HOLD signals

            signals.append({
                'index': i,
                'timestamp': row['timestamp'] if 'timestamp' in row else row.name,
                'signal': signal_type,
                'entry_price': price,
                'stop_loss': stop,
                'take_profit': target,
                'score': score,
                'reasons': reasons
            })

        return signals

    def evaluate_signal_outcome(self, df: pd.DataFrame, signal: Dict,
                                 max_holding_periods: int = 20) -> Dict:
        """Evaluate the outcome of a signal"""
        start_idx = signal['index']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        signal_type = signal['signal']

        for i in range(start_idx + 1, min(start_idx + max_holding_periods + 1, len(df))):
            price = df.iloc[i]['close'] if 'close' in df.iloc[i] else df.iloc[i]['price']

            if signal_type == 'BUY':
                if price >= take_profit:
                    return {
                        'outcome': 'WIN',
                        'exit_price': price,
                        'pnl_percent': ((price - entry_price) / entry_price) * 100,
                        'holding_periods': i - start_idx
                    }
                elif price <= stop_loss:
                    return {
                        'outcome': 'LOSS',
                        'exit_price': price,
                        'pnl_percent': ((price - entry_price) / entry_price) * 100,
                        'holding_periods': i - start_idx
                    }
            else:  # SELL
                if price <= take_profit:
                    return {
                        'outcome': 'WIN',
                        'exit_price': price,
                        'pnl_percent': ((entry_price - price) / entry_price) * 100,
                        'holding_periods': i - start_idx
                    }
                elif price >= stop_loss:
                    return {
                        'outcome': 'LOSS',
                        'exit_price': price,
                        'pnl_percent': ((entry_price - price) / entry_price) * 100,
                        'holding_periods': i - start_idx
                    }

        # Expired without hitting stop or target
        final_price = df.iloc[min(start_idx + max_holding_periods, len(df) - 1)]['close']
        if signal_type == 'BUY':
            pnl = ((final_price - entry_price) / entry_price) * 100
        else:
            pnl = ((entry_price - final_price) / entry_price) * 100

        return {
            'outcome': 'WIN' if pnl > 0 else 'LOSS',
            'exit_price': final_price,
            'pnl_percent': pnl,
            'holding_periods': max_holding_periods
        }

    def run_backtest(self, symbol: str, initial_capital: float = 1000000) -> BacktestResult:
        """Run full backtest for a symbol"""
        # Fetch data
        raw_data = self.fetcher.get_intraday_data(symbol, limit=2000)

        if raw_data.empty or len(raw_data) < 100:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0, total_return=0,
                max_drawdown=0, sharpe_ratio=0, avg_win=0, avg_loss=0,
                best_trade=0, worst_trade=0, avg_holding_period=0,
                equity_curve=[initial_capital], trades=[]
            )

        # Convert to OHLCV
        ohlcv = self.fetcher.get_ohlcv_from_ticks(raw_data, '15min')

        if ohlcv.empty or len(ohlcv) < 50:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0, total_return=0,
                max_drawdown=0, sharpe_ratio=0, avg_win=0, avg_loss=0,
                best_trade=0, worst_trade=0, avg_holding_period=0,
                equity_curve=[initial_capital], trades=[]
            )

        # Generate signals
        signals = self.generate_historical_signals(ohlcv)

        if not signals:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0, total_return=0,
                max_drawdown=0, sharpe_ratio=0, avg_win=0, avg_loss=0,
                best_trade=0, worst_trade=0, avg_holding_period=0,
                equity_curve=[initial_capital], trades=[]
            )

        # Evaluate each signal
        trades = []
        wins = []
        losses = []
        holding_periods = []

        for signal in signals:
            outcome = self.evaluate_signal_outcome(ohlcv, signal)
            trade = {**signal, **outcome}
            trades.append(trade)

            if outcome['outcome'] == 'WIN':
                wins.append(outcome['pnl_percent'])
            else:
                losses.append(outcome['pnl_percent'])
            holding_periods.append(outcome['holding_periods'])

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 and avg_loss > 0 else 0

        # Calculate equity curve
        equity = initial_capital
        equity_curve = [equity]
        peak = equity
        max_dd = 0

        for trade in trades:
            pnl = trade['pnl_percent'] / 100
            position_size = equity * 0.05  # 5% position size
            trade_pnl = position_size * pnl
            equity += trade_pnl
            equity_curve.append(equity)

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        total_return = ((equity - initial_capital) / initial_capital) * 100

        # Sharpe ratio (simplified)
        returns = [t['pnl_percent'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=max(wins) if wins else 0,
            worst_trade=min(losses) if losses else 0,
            avg_holding_period=np.mean(holding_periods) if holding_periods else 0,
            equity_curve=equity_curve,
            trades=trades
        )


class SignalTracker:
    """Track signals and their outcomes for performance history"""

    def __init__(self, storage_path: str = "signal_history.json"):
        self.storage_path = storage_path
        self.signals: List[SignalRecord] = []
        self.load_history()

    def load_history(self):
        """Load signal history from file"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord(**s) for s in data]
            except:
                self.signals = []

    def save_history(self):
        """Save signal history to file"""
        try:
            with open(self.storage_path, 'w') as f:
                data = [vars(s) for s in self.signals]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving signal history: {e}")

    def record_signal(self, symbol: str, signal_type: str, entry_price: float,
                      stop_loss: float, take_profit: float, confidence: float,
                      reasons: List[str], timeframes_aligned: int = 1) -> str:
        """Record a new signal"""
        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            reasons=reasons,
            timeframes_aligned=timeframes_aligned
        )

        self.signals.append(record)
        self.save_history()
        return signal_id

    def update_signal_outcome(self, signal_id: str, outcome: str,
                               exit_price: float, pnl_percent: float):
        """Update a signal with its outcome"""
        for signal in self.signals:
            if signal.signal_id == signal_id:
                signal.outcome = outcome
                signal.exit_price = exit_price
                signal.exit_timestamp = datetime.now().isoformat()
                signal.pnl_percent = pnl_percent
                self.save_history()
                return True
        return False

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        completed = [s for s in self.signals if s.outcome != 'PENDING']

        if not completed:
            return {
                'total_signals': len(self.signals),
                'completed': 0,
                'pending': len(self.signals),
                'win_rate': 0,
                'avg_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        wins = [s for s in completed if s.outcome == 'WIN']
        losses = [s for s in completed if s.outcome == 'LOSS']

        pnls = [s.pnl_percent for s in completed]

        return {
            'total_signals': len(self.signals),
            'completed': len(completed),
            'pending': len(self.signals) - len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(completed) * 100) if completed else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_confidence': np.mean([s.confidence for s in self.signals])
        }

    def get_recent_signals(self, limit: int = 20) -> List[SignalRecord]:
        """Get recent signals"""
        return sorted(self.signals, key=lambda x: x.timestamp, reverse=True)[:limit]


class ProfessionalSignalGenerator:
    """Professional-grade signal generator with all enhancements"""

    def __init__(self):
        self.fetcher = PSXDataFetcher()
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.fetcher)
        self.backtest_engine = BacktestEngine(self.fetcher)
        self.signal_tracker = SignalTracker()
        self.analyzer = TechnicalAnalyzer()

    def generate_signal(self, symbol: str, track: bool = True) -> Dict:
        """Generate a professional-grade signal"""

        # Get multi-timeframe analysis
        mtf_result = self.mtf_analyzer.get_multi_timeframe_signal(symbol)

        if mtf_result['signal'] == 'HOLD' or mtf_result['confidence'] < 40:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': mtf_result['confidence'],
                'entry_price': mtf_result.get('price', 0),
                'stop_loss': 0,
                'take_profit': 0,
                'reasons': ['Insufficient timeframe alignment'],
                'timeframes_aligned': mtf_result['timeframes_aligned'],
                'grade': 'D',
                'investable': False
            }

        # Calculate entry/exit levels
        price = mtf_result['price']
        atr_pct = 0.03  # Default 3% ATR estimate

        if 'BUY' in mtf_result['signal']:
            stop_loss = price * (1 - max(0.02, atr_pct * 1.5))
            take_profit = price * (1 + max(0.04, atr_pct * 3))
        else:
            stop_loss = price * (1 + max(0.02, atr_pct * 1.5))
            take_profit = price * (1 - max(0.04, atr_pct * 3))

        # Determine grade
        aligned = mtf_result['timeframes_aligned']
        confidence = mtf_result['confidence']

        if aligned >= 4 and confidence >= 75:
            grade = 'A'
            investable = True
        elif aligned >= 3 and confidence >= 60:
            grade = 'B'
            investable = True
        elif aligned >= 2 and confidence >= 50:
            grade = 'C'
            investable = False  # Needs confirmation
        else:
            grade = 'D'
            investable = False

        # Build reasons
        reasons = []
        for tf, analysis in mtf_result.get('analysis', {}).items():
            if analysis['signal'] != 'NEUTRAL':
                reasons.append(f"{tf}: {analysis['signal']} (RSI: {analysis.get('rsi', 50):.0f})")

        result = {
            'symbol': symbol,
            'signal': mtf_result['signal'],
            'confidence': confidence,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasons': reasons[:5],
            'timeframes_aligned': aligned,
            'total_timeframes': mtf_result.get('total_timeframes', 4),
            'grade': grade,
            'investable': investable,
            'analysis': mtf_result.get('analysis', {})
        }

        # Track signal if requested
        if track and result['signal'] != 'HOLD':
            self.signal_tracker.record_signal(
                symbol=symbol,
                signal_type=result['signal'],
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                reasons=reasons,
                timeframes_aligned=aligned
            )

        return result

    def get_backtest_summary(self, symbol: str) -> Dict:
        """Get backtest summary for a symbol"""
        result = self.backtest_engine.run_backtest(symbol)

        return {
            'symbol': symbol,
            'total_trades': result.total_trades,
            'win_rate': f"{result.win_rate:.1f}%",
            'profit_factor': f"{result.profit_factor:.2f}",
            'total_return': f"{result.total_return:.1f}%",
            'max_drawdown': f"{result.max_drawdown:.1f}%",
            'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
            'avg_win': f"{result.avg_win:.2f}%",
            'avg_loss': f"{result.avg_loss:.2f}%",
            'equity_curve': result.equity_curve,
            'is_profitable': result.total_return > 0 and result.win_rate > 50
        }

    def get_performance_history(self) -> Dict:
        """Get signal tracking performance history"""
        return self.signal_tracker.get_performance_stats()

    def scan_market(self, symbols: List[str], min_grade: str = 'C') -> List[Dict]:
        """Scan market for investable opportunities"""
        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        min_grade_value = grade_order.get(min_grade, 2)

        opportunities = []

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol, track=False)

                if signal['signal'] != 'HOLD':
                    signal_grade_value = grade_order.get(signal['grade'], 0)
                    if signal_grade_value >= min_grade_value:
                        opportunities.append(signal)
            except Exception as e:
                continue

        # Sort by grade and confidence
        opportunities.sort(key=lambda x: (grade_order.get(x['grade'], 0), x['confidence']), reverse=True)

        return opportunities


# Export main class
__all__ = ['ProfessionalSignalGenerator', 'BacktestResult', 'SignalRecord']
