"""
PSX MARKET FILTER - Pakistan Stock Exchange Specific Rules
===========================================================
Applies PSX-specific trading rules and filters
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
import requests


class PSXMarketFilter:
    """
    Filter stocks based on PSX-specific rules and market conditions
    """

    def __init__(self):
        # PSX Trading Hours (Pakistan Standard Time)
        self.pre_open_start = time(9, 15)
        self.market_open = time(9, 30)
        self.market_close = time(15, 30)

        # PSX Circuit Breaker Limits
        self.circuit_limits = {
            'LEVEL_1': 0.05,   # 5% - First halt
            'LEVEL_2': 0.075,  # 7.5% - Extended halt
        }

        # Minimum trading requirements
        self.min_volume = 10000        # Minimum daily volume
        self.min_avg_volume = 25000    # Minimum 20-day average volume
        self.min_price = 1.0           # Minimum price (PKR)
        self.max_spread_pct = 2.0      # Maximum bid-ask spread %

        # KSE-100 Index components (top liquid stocks)
        self.kse100_symbols = [
            'HBL', 'UBL', 'MCB', 'ABL', 'NBP', 'BAFL', 'BAHL', 'MEBL',
            'LUCK', 'DGKC', 'MLCF', 'FCCL', 'KOHC', 'PIOC', 'CHCC',
            'FFC', 'EFERT', 'FFBL', 'ENGRO', 'FATIMA',
            'PSO', 'OGDC', 'PPL', 'POL', 'MARI', 'SSGC', 'SNGP',
            'HUBC', 'KEL', 'NCPL', 'KAPCO',
            'TRG', 'SYS', 'NETSOL',
            'INDU', 'PSMC', 'HCAR', 'MTL',
            'SEARL', 'GLAXO', 'FEROZ',
            'NESTLE', 'ICI', 'EPCL', 'LOTCHEM',
            'NML', 'NCL', 'UNITY', 'FFL'
        ]

        # High volatility stocks (require extra caution)
        self.high_volatility_stocks = [
            'TRG', 'SYS', 'NETSOL',  # Tech stocks
            'HCAR', 'PSMC',          # Autos
        ]

        # Stocks with frequent circuit breakers
        self.circuit_prone_stocks = []

    def is_market_open(self) -> bool:
        """Check if PSX market is currently open"""
        now = datetime.now()

        # Check if weekday (PSX closed on Saturday and Sunday)
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return self.market_open <= current_time <= self.market_close

    def get_trading_session(self) -> Dict:
        """
        Get current trading session with recommendations
        """
        now = datetime.now()
        current_time = now.time()

        session = {
            'name': '',
            'is_market_open': self.is_market_open(),
            'recommendation': '',
            'position_size_modifier': 1.0,
            'avoid_new_positions': False,
        }

        if now.weekday() >= 5:
            session['name'] = 'WEEKEND'
            session['recommendation'] = 'Market closed. Use time for analysis and planning.'
            session['avoid_new_positions'] = True

        elif current_time < self.pre_open_start:
            session['name'] = 'PRE_MARKET'
            session['recommendation'] = 'Prepare watchlist and review overnight news.'
            session['avoid_new_positions'] = True

        elif current_time < self.market_open:
            session['name'] = 'PRE_OPEN_AUCTION'
            session['recommendation'] = 'Monitor pre-open prices for gaps.'
            session['avoid_new_positions'] = True

        elif self.market_open <= current_time < time(10, 0):
            session['name'] = 'OPENING_SESSION'
            session['recommendation'] = 'High volatility. Wait for price discovery. Reduce position sizes.'
            session['position_size_modifier'] = 0.6
            session['avoid_new_positions'] = False

        elif time(10, 0) <= current_time < time(12, 30):
            session['name'] = 'MORNING_SESSION'
            session['recommendation'] = 'Prime trading hours. Normal position sizing.'
            session['position_size_modifier'] = 1.0
            session['avoid_new_positions'] = False

        elif time(12, 30) <= current_time < time(13, 30):
            session['name'] = 'LUNCH_SESSION'
            session['recommendation'] = 'Lower volume period. Smaller positions recommended.'
            session['position_size_modifier'] = 0.7
            session['avoid_new_positions'] = False

        elif time(13, 30) <= current_time < time(15, 0):
            session['name'] = 'AFTERNOON_SESSION'
            session['recommendation'] = 'Good momentum period. Normal trading.'
            session['position_size_modifier'] = 0.9
            session['avoid_new_positions'] = False

        elif time(15, 0) <= current_time <= self.market_close:
            session['name'] = 'CLOSING_SESSION'
            session['recommendation'] = 'Avoid new positions. Close intraday trades.'
            session['position_size_modifier'] = 0.5
            session['avoid_new_positions'] = True

        else:
            session['name'] = 'AFTER_HOURS'
            session['recommendation'] = 'Market closed. Review performance and prepare for tomorrow.'
            session['avoid_new_positions'] = True

        return session

    def filter_tradable_stocks(self, stocks_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter stocks based on PSX trading requirements

        Args:
            stocks_data: Dict of {symbol: {price, volume, avg_volume, spread, etc}}

        Returns:
            Filtered dict of tradable stocks with filter reasons
        """
        tradable = {}

        for symbol, data in stocks_data.items():
            filters_passed = []
            filters_failed = []

            # Price filter
            if data.get('price', 0) >= self.min_price:
                filters_passed.append('MIN_PRICE')
            else:
                filters_failed.append(f'Price below {self.min_price} PKR')

            # Volume filter
            if data.get('volume', 0) >= self.min_volume:
                filters_passed.append('MIN_VOLUME')
            else:
                filters_failed.append(f'Volume below {self.min_volume}')

            # Average volume filter
            if data.get('avg_volume', 0) >= self.min_avg_volume:
                filters_passed.append('AVG_VOLUME')
            else:
                filters_failed.append(f'Avg volume below {self.min_avg_volume}')

            # Spread filter (if available)
            spread = data.get('spread_pct', 0)
            if spread <= self.max_spread_pct or spread == 0:
                filters_passed.append('SPREAD')
            else:
                filters_failed.append(f'Spread too wide: {spread:.1f}%')

            # Circuit breaker check
            if not data.get('circuit_hit', False):
                filters_passed.append('NO_CIRCUIT')
            else:
                filters_failed.append('Circuit breaker hit')

            # Determine if tradable
            is_tradable = len(filters_failed) == 0

            tradable[symbol] = {
                **data,
                'is_tradable': is_tradable,
                'filters_passed': filters_passed,
                'filters_failed': filters_failed,
                'is_kse100': symbol in self.kse100_symbols,
                'is_high_volatility': symbol in self.high_volatility_stocks,
            }

        return tradable

    def check_circuit_breaker(self, symbol: str, current_price: float,
                            reference_price: float) -> Dict:
        """
        Check if stock has hit or is near circuit breaker

        Args:
            symbol: Stock symbol
            current_price: Current trading price
            reference_price: Previous day's closing price

        Returns:
            Dict with circuit status
        """
        if reference_price <= 0:
            return {'status': 'UNKNOWN', 'message': 'No reference price'}

        change_pct = (current_price - reference_price) / reference_price

        result = {
            'symbol': symbol,
            'change_pct': change_pct * 100,
            'status': 'NORMAL',
            'warning': None,
            'can_trade': True,
        }

        # Check upper circuit
        if change_pct >= self.circuit_limits['LEVEL_2']:
            result['status'] = 'UPPER_CIRCUIT_L2'
            result['warning'] = 'Stock at upper circuit (7.5%)'
            result['can_trade'] = False

        elif change_pct >= self.circuit_limits['LEVEL_1']:
            result['status'] = 'UPPER_CIRCUIT_L1'
            result['warning'] = 'Stock at upper circuit (5%)'
            result['can_trade'] = False

        elif change_pct >= self.circuit_limits['LEVEL_1'] * 0.9:
            result['status'] = 'NEAR_UPPER_CIRCUIT'
            result['warning'] = 'Approaching upper circuit - be cautious'
            result['can_trade'] = True

        # Check lower circuit
        elif change_pct <= -self.circuit_limits['LEVEL_2']:
            result['status'] = 'LOWER_CIRCUIT_L2'
            result['warning'] = 'Stock at lower circuit (7.5%)'
            result['can_trade'] = False

        elif change_pct <= -self.circuit_limits['LEVEL_1']:
            result['status'] = 'LOWER_CIRCUIT_L1'
            result['warning'] = 'Stock at lower circuit (5%)'
            result['can_trade'] = False

        elif change_pct <= -self.circuit_limits['LEVEL_1'] * 0.9:
            result['status'] = 'NEAR_LOWER_CIRCUIT'
            result['warning'] = 'Approaching lower circuit - be cautious'
            result['can_trade'] = True

        return result

    def get_lot_size(self, symbol: str, price: float) -> int:
        """
        Get standard lot size for a stock
        PSX uses different lot sizes based on price
        """
        if price >= 100:
            return 500
        elif price >= 50:
            return 500
        elif price >= 25:
            return 500
        elif price >= 10:
            return 500
        else:
            return 500  # Default lot size

    def round_to_lot_size(self, shares: int, symbol: str, price: float) -> int:
        """Round shares to nearest lot size"""
        lot_size = self.get_lot_size(symbol, price)
        return max(lot_size, (shares // lot_size) * lot_size)

    def calculate_t_plus_2_settlement(self, trade_date: datetime = None) -> datetime:
        """
        Calculate T+2 settlement date (PSX standard)
        """
        if trade_date is None:
            trade_date = datetime.now()

        settlement_date = trade_date
        business_days = 0

        while business_days < 2:
            settlement_date += timedelta(days=1)
            # Skip weekends
            if settlement_date.weekday() < 5:
                business_days += 1

        return settlement_date

    def get_market_news_impact(self) -> Dict:
        """
        Analyze market-wide news impact (placeholder for news API integration)
        """
        return {
            'overall_sentiment': 'NEUTRAL',
            'key_events': [],
            'sector_impacts': {},
            'recommendation': 'No significant market-moving news detected.'
        }

    def get_kse100_trend(self, kse100_data: pd.DataFrame = None) -> Dict:
        """
        Analyze KSE-100 index trend for market direction
        """
        if kse100_data is None or kse100_data.empty:
            return {
                'trend': 'UNKNOWN',
                'strength': 0,
                'recommendation': 'Unable to determine market trend'
            }

        # Calculate trend indicators
        if len(kse100_data) >= 20:
            sma_5 = kse100_data['close'].tail(5).mean()
            sma_20 = kse100_data['close'].tail(20).mean()
            current = kse100_data['close'].iloc[-1]

            if current > sma_5 > sma_20:
                trend = 'BULLISH'
                strength = 80
                recommendation = 'Market trending up. Favor long positions.'
            elif current < sma_5 < sma_20:
                trend = 'BEARISH'
                strength = 80
                recommendation = 'Market trending down. Be cautious with longs.'
            elif sma_5 > sma_20:
                trend = 'SLIGHTLY_BULLISH'
                strength = 60
                recommendation = 'Slight bullish bias. Normal trading.'
            elif sma_5 < sma_20:
                trend = 'SLIGHTLY_BEARISH'
                strength = 60
                recommendation = 'Slight bearish bias. Reduce position sizes.'
            else:
                trend = 'NEUTRAL'
                strength = 50
                recommendation = 'Market consolidating. Wait for direction.'
        else:
            trend = 'UNKNOWN'
            strength = 0
            recommendation = 'Insufficient data'

        return {
            'trend': trend,
            'strength': strength,
            'recommendation': recommendation
        }


# PSX Holiday Calendar (add actual holidays)
PSX_HOLIDAYS_2024 = [
    datetime(2024, 2, 5),   # Kashmir Day
    datetime(2024, 3, 23),  # Pakistan Day
    datetime(2024, 5, 1),   # Labor Day
    # Add more holidays as needed
]


def is_psx_holiday(date: datetime = None) -> bool:
    """Check if given date is a PSX holiday"""
    if date is None:
        date = datetime.now()

    return date.date() in [h.date() for h in PSX_HOLIDAYS_2024]


if __name__ == "__main__":
    print("Testing PSX Market Filter...")

    filter = PSXMarketFilter()

    # Test session
    session = filter.get_trading_session()
    print(f"\nCurrent Session: {session['name']}")
    print(f"Market Open: {session['is_market_open']}")
    print(f"Recommendation: {session['recommendation']}")
    print(f"Position Modifier: {session['position_size_modifier']}")

    # Test circuit breaker
    circuit = filter.check_circuit_breaker('HBL', 262, 250)
    print(f"\nCircuit Check for HBL:")
    print(f"  Change: {circuit['change_pct']:.1f}%")
    print(f"  Status: {circuit['status']}")
    print(f"  Can Trade: {circuit['can_trade']}")

    # Test settlement
    settlement = filter.calculate_t_plus_2_settlement()
    print(f"\nT+2 Settlement Date: {settlement.strftime('%Y-%m-%d')}")
