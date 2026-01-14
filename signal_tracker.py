"""
SIGNAL PERFORMANCE TRACKER - Track and Analyze Signal Quality
===============================================================
Measures actual performance of signals to continuously improve the system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class SignalRecord:
    """Record of a generated signal"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    quality_grade: str  # A, B, C, D
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: str

    # Outcome (filled in later)
    outcome: str = ""  # WIN, LOSS, PENDING, EXPIRED
    exit_price: float = 0
    exit_timestamp: str = ""
    actual_return_pct: float = 0
    max_favorable_pct: float = 0  # Best price reached
    max_adverse_pct: float = 0    # Worst price reached
    hit_target: bool = False
    hit_stop: bool = False
    days_to_outcome: int = 0


class SignalTracker:
    """
    Track signal performance to measure and improve accuracy
    """

    def __init__(self, data_file: str = 'signal_history.json'):
        self.data_file = data_file
        self.signals: Dict[str, SignalRecord] = {}
        self.performance_stats: Dict = {}
        self._load_data()

    def record_signal(self, symbol: str, signal_type: str, confidence: float,
                     quality_grade: str, entry_price: float,
                     stop_loss: float, take_profit: float) -> str:
        """
        Record a new signal for tracking
        Returns signal_id
        """
        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            quality_grade=quality_grade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now().isoformat()
        )

        self.signals[signal_id] = record
        self._save_data()

        return signal_id

    def update_signal_outcome(self, signal_id: str, current_price: float) -> Optional[str]:
        """
        Update signal with current price and check for outcome
        Returns outcome if signal is resolved (WIN/LOSS)
        """
        if signal_id not in self.signals:
            return None

        signal = self.signals[signal_id]

        if signal.outcome in ['WIN', 'LOSS']:
            return None  # Already resolved

        entry = signal.entry_price

        # Calculate current metrics
        if signal.signal_type == 'BUY':
            current_return = (current_price - entry) / entry * 100
            favorable = (current_price - entry) / entry * 100
            adverse = min(0, favorable)
        else:  # SELL
            current_return = (entry - current_price) / entry * 100
            favorable = (entry - current_price) / entry * 100
            adverse = min(0, favorable)

        # Update max favorable/adverse
        signal.max_favorable_pct = max(signal.max_favorable_pct, favorable)
        signal.max_adverse_pct = min(signal.max_adverse_pct, adverse)

        # Check for stop loss
        if signal.signal_type == 'BUY' and current_price <= signal.stop_loss:
            signal.outcome = 'LOSS'
            signal.hit_stop = True
            signal.exit_price = current_price
            signal.exit_timestamp = datetime.now().isoformat()
            signal.actual_return_pct = current_return

        elif signal.signal_type == 'SELL' and current_price >= signal.stop_loss:
            signal.outcome = 'LOSS'
            signal.hit_stop = True
            signal.exit_price = current_price
            signal.exit_timestamp = datetime.now().isoformat()
            signal.actual_return_pct = current_return

        # Check for take profit
        elif signal.signal_type == 'BUY' and current_price >= signal.take_profit:
            signal.outcome = 'WIN'
            signal.hit_target = True
            signal.exit_price = current_price
            signal.exit_timestamp = datetime.now().isoformat()
            signal.actual_return_pct = current_return

        elif signal.signal_type == 'SELL' and current_price <= signal.take_profit:
            signal.outcome = 'WIN'
            signal.hit_target = True
            signal.exit_price = current_price
            signal.exit_timestamp = datetime.now().isoformat()
            signal.actual_return_pct = current_return

        # Check for expiry (7 days max)
        signal_date = datetime.fromisoformat(signal.timestamp)
        days_elapsed = (datetime.now() - signal_date).days
        signal.days_to_outcome = days_elapsed

        if days_elapsed >= 7 and signal.outcome not in ['WIN', 'LOSS']:
            signal.outcome = 'EXPIRED'
            signal.exit_price = current_price
            signal.exit_timestamp = datetime.now().isoformat()
            signal.actual_return_pct = current_return

        self._save_data()
        return signal.outcome if signal.outcome else None

    def get_performance_stats(self) -> Dict:
        """
        Calculate comprehensive performance statistics
        """
        resolved_signals = [s for s in self.signals.values()
                          if s.outcome in ['WIN', 'LOSS', 'EXPIRED']]

        if not resolved_signals:
            return {
                'total_signals': len(self.signals),
                'resolved_signals': 0,
                'pending_signals': len(self.signals),
                'message': 'No resolved signals yet'
            }

        # Overall stats
        wins = [s for s in resolved_signals if s.outcome == 'WIN']
        losses = [s for s in resolved_signals if s.outcome == 'LOSS']
        expired = [s for s in resolved_signals if s.outcome == 'EXPIRED']

        win_rate = len(wins) / len(resolved_signals) * 100 if resolved_signals else 0

        avg_win = np.mean([s.actual_return_pct for s in wins]) if wins else 0
        avg_loss = np.mean([s.actual_return_pct for s in losses]) if losses else 0

        # Profit factor
        total_wins = sum(s.actual_return_pct for s in wins) if wins else 0
        total_losses = abs(sum(s.actual_return_pct for s in losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # By confidence level
        high_conf = [s for s in resolved_signals if s.confidence >= 75]
        med_conf = [s for s in resolved_signals if 60 <= s.confidence < 75]
        low_conf = [s for s in resolved_signals if s.confidence < 60]

        high_conf_win_rate = len([s for s in high_conf if s.outcome == 'WIN']) / len(high_conf) * 100 if high_conf else 0
        med_conf_win_rate = len([s for s in med_conf if s.outcome == 'WIN']) / len(med_conf) * 100 if med_conf else 0
        low_conf_win_rate = len([s for s in low_conf if s.outcome == 'WIN']) / len(low_conf) * 100 if low_conf else 0

        # By quality grade
        grade_stats = {}
        for grade in ['A', 'B', 'C', 'D']:
            grade_signals = [s for s in resolved_signals if s.quality_grade == grade]
            if grade_signals:
                grade_wins = len([s for s in grade_signals if s.outcome == 'WIN'])
                grade_stats[grade] = {
                    'count': len(grade_signals),
                    'win_rate': grade_wins / len(grade_signals) * 100,
                    'avg_return': np.mean([s.actual_return_pct for s in grade_signals])
                }

        # By signal type
        buy_signals = [s for s in resolved_signals if s.signal_type == 'BUY']
        sell_signals = [s for s in resolved_signals if s.signal_type == 'SELL']

        buy_win_rate = len([s for s in buy_signals if s.outcome == 'WIN']) / len(buy_signals) * 100 if buy_signals else 0
        sell_win_rate = len([s for s in sell_signals if s.outcome == 'WIN']) / len(sell_signals) * 100 if sell_signals else 0

        # Recent performance (last 20 signals)
        recent = sorted(resolved_signals, key=lambda x: x.timestamp, reverse=True)[:20]
        recent_win_rate = len([s for s in recent if s.outcome == 'WIN']) / len(recent) * 100 if recent else 0

        return {
            'total_signals': len(self.signals),
            'resolved_signals': len(resolved_signals),
            'pending_signals': len(self.signals) - len(resolved_signals),

            'overall': {
                'win_rate': round(win_rate, 1),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'wins': len(wins),
                'losses': len(losses),
                'expired': len(expired)
            },

            'by_confidence': {
                'high_75+': {'count': len(high_conf), 'win_rate': round(high_conf_win_rate, 1)},
                'medium_60-75': {'count': len(med_conf), 'win_rate': round(med_conf_win_rate, 1)},
                'low_<60': {'count': len(low_conf), 'win_rate': round(low_conf_win_rate, 1)}
            },

            'by_quality': grade_stats,

            'by_type': {
                'BUY': {'count': len(buy_signals), 'win_rate': round(buy_win_rate, 1)},
                'SELL': {'count': len(sell_signals), 'win_rate': round(sell_win_rate, 1)}
            },

            'recent_20': {
                'win_rate': round(recent_win_rate, 1),
                'signals': len(recent)
            }
        }

    def get_symbol_performance(self, symbol: str) -> Dict:
        """Get performance stats for a specific symbol"""
        symbol_signals = [s for s in self.signals.values()
                        if s.symbol == symbol and s.outcome in ['WIN', 'LOSS', 'EXPIRED']]

        if not symbol_signals:
            return {'symbol': symbol, 'signals': 0, 'message': 'No resolved signals'}

        wins = len([s for s in symbol_signals if s.outcome == 'WIN'])

        return {
            'symbol': symbol,
            'signals': len(symbol_signals),
            'win_rate': round(wins / len(symbol_signals) * 100, 1),
            'avg_return': round(np.mean([s.actual_return_pct for s in symbol_signals]), 2),
            'best_trade': round(max(s.actual_return_pct for s in symbol_signals), 2),
            'worst_trade': round(min(s.actual_return_pct for s in symbol_signals), 2)
        }

    def get_pending_signals(self) -> List[SignalRecord]:
        """Get all pending (unresolved) signals"""
        return [s for s in self.signals.values() if s.outcome == '' or s.outcome == 'PENDING']

    def get_recent_signals(self, days: int = 7) -> List[SignalRecord]:
        """Get signals from the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [s for s in self.signals.values()
                if datetime.fromisoformat(s.timestamp) > cutoff]

    def generate_report(self) -> str:
        """Generate a text report of signal performance"""
        stats = self.get_performance_stats()

        if stats.get('message'):
            return f"Signal Tracker Report\n{'-'*40}\n{stats['message']}"

        report = f"""
{'='*50}
       SIGNAL PERFORMANCE REPORT
{'='*50}

OVERALL STATISTICS
------------------
Total Signals: {stats['total_signals']}
Resolved: {stats['resolved_signals']} | Pending: {stats['pending_signals']}

Win Rate: {stats['overall']['win_rate']}%
Wins: {stats['overall']['wins']} | Losses: {stats['overall']['losses']} | Expired: {stats['overall']['expired']}
Avg Win: +{stats['overall']['avg_win']}% | Avg Loss: {stats['overall']['avg_loss']}%
Profit Factor: {stats['overall']['profit_factor']}

PERFORMANCE BY CONFIDENCE
-------------------------
High (75%+): {stats['by_confidence']['high_75+']['win_rate']}% win rate ({stats['by_confidence']['high_75+']['count']} signals)
Medium (60-75%): {stats['by_confidence']['medium_60-75']['win_rate']}% win rate ({stats['by_confidence']['medium_60-75']['count']} signals)
Low (<60%): {stats['by_confidence']['low_<60']['win_rate']}% win rate ({stats['by_confidence']['low_<60']['count']} signals)

PERFORMANCE BY QUALITY GRADE
----------------------------"""

        for grade, data in stats.get('by_quality', {}).items():
            report += f"\nGrade {grade}: {data['win_rate']}% win rate ({data['count']} signals, avg {data['avg_return']:+.1f}%)"

        report += f"""

PERFORMANCE BY SIGNAL TYPE
--------------------------
BUY Signals: {stats['by_type']['BUY']['win_rate']}% win rate ({stats['by_type']['BUY']['count']} signals)
SELL Signals: {stats['by_type']['SELL']['win_rate']}% win rate ({stats['by_type']['SELL']['count']} signals)

RECENT PERFORMANCE (Last 20)
----------------------------
Win Rate: {stats['recent_20']['win_rate']}%

{'='*50}
"""
        return report

    def cleanup_old_signals(self, days: int = 30):
        """Remove signals older than N days"""
        cutoff = datetime.now() - timedelta(days=days)
        old_ids = [sid for sid, s in self.signals.items()
                   if datetime.fromisoformat(s.timestamp) < cutoff]

        for sid in old_ids:
            del self.signals[sid]

        self._save_data()
        return len(old_ids)

    def _save_data(self):
        """Save signals to file"""
        data = {sid: asdict(s) for sid, s in self.signals.items()}
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving signal data: {e}")

    def _load_data(self):
        """Load signals from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)

                for sid, s_data in data.items():
                    self.signals[sid] = SignalRecord(**s_data)
        except Exception as e:
            print(f"Error loading signal data: {e}")
            self.signals = {}


# Singleton instance
_tracker_instance = None

def get_tracker() -> SignalTracker:
    """Get the singleton tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SignalTracker()
    return _tracker_instance


if __name__ == "__main__":
    print("Testing Signal Tracker...")

    tracker = SignalTracker('test_signals.json')

    # Record some test signals
    test_signals = [
        ('HBL', 'BUY', 78, 'A', 250, 242, 270),
        ('LUCK', 'BUY', 65, 'B', 850, 820, 900),
        ('FFC', 'SELL', 72, 'B', 120, 128, 110),
        ('PSO', 'BUY', 55, 'C', 400, 385, 430),
    ]

    for symbol, sig_type, conf, grade, entry, sl, tp in test_signals:
        sid = tracker.record_signal(symbol, sig_type, conf, grade, entry, sl, tp)
        print(f"Recorded: {sid}")

    # Simulate some outcomes
    tracker.signals[list(tracker.signals.keys())[0]].outcome = 'WIN'
    tracker.signals[list(tracker.signals.keys())[0]].actual_return_pct = 5.2
    tracker.signals[list(tracker.signals.keys())[1]].outcome = 'LOSS'
    tracker.signals[list(tracker.signals.keys())[1]].actual_return_pct = -3.1

    # Print report
    print(tracker.generate_report())

    # Cleanup
    os.remove('test_signals.json')
