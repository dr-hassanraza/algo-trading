#!/usr/bin/env python3
"""
Portfolio Management Module
===========================

Handles portfolio tracking, position management, and P&L calculations
for the algorithmic trading chatbot.
"""

import json
import datetime as dt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class Position:
    """Represents a single position in the portfolio"""
    symbol: str
    quantity: int
    avg_price: float
    date_bought: str
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    notes: str = ""


@dataclass
class Transaction:
    """Represents a buy/sell transaction"""
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    date: str
    fees: float = 0.0
    notes: str = ""


class PortfolioManager:
    """Manages portfolio positions, transactions, and P&L calculations"""
    
    def __init__(self, portfolio_file: str = 'portfolio.json'):
        self.portfolio_file = portfolio_file
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        self.cash_balance: float = 0.0
        self.load_portfolio()
    
    def add_position(self, symbol: str, quantity: int, price: float, 
                    date: Optional[str] = None, notes: str = "") -> str:
        """Add a new position or update existing position"""
        
        if date is None:
            date = dt.datetime.now().strftime('%Y-%m-%d')
        
        # Create transaction record
        transaction = Transaction(
            symbol=symbol,
            action='buy',
            quantity=quantity,
            price=price,
            date=date,
            notes=notes
        )
        self.transactions.append(transaction)
        
        # Update position
        if symbol in self.positions:
            # Average down/up existing position
            existing = self.positions[symbol]
            total_cost = (existing.quantity * existing.avg_price) + (quantity * price)
            total_quantity = existing.quantity + quantity
            new_avg_price = total_cost / total_quantity
            
            existing.quantity = total_quantity
            existing.avg_price = new_avg_price
            existing.notes += f" | Added {quantity}@{price} on {date}"
            
            message = f"Updated {symbol}: {total_quantity} shares @ {new_avg_price:.2f} avg"
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                date_bought=date,
                notes=notes
            )
            message = f"Added new position: {symbol} {quantity} shares @ {price:.2f}"
        
        # Update cash balance
        total_cost = quantity * price
        self.cash_balance -= total_cost
        
        self.save_portfolio()
        return message
    
    def sell_position(self, symbol: str, quantity: int, price: float,
                     date: Optional[str] = None, notes: str = "") -> str:
        """Sell part or all of a position"""
        
        if symbol not in self.positions:
            return f"Error: No position found for {symbol}"
        
        if date is None:
            date = dt.datetime.now().strftime('%Y-%m-%d')
        
        position = self.positions[symbol]
        
        if quantity > position.quantity:
            return f"Error: Cannot sell {quantity} shares, only have {position.quantity}"
        
        # Create transaction record
        transaction = Transaction(
            symbol=symbol,
            action='sell',
            quantity=quantity,
            price=price,
            date=date,
            notes=notes
        )
        self.transactions.append(transaction)
        
        # Calculate P&L
        cost_basis = quantity * position.avg_price
        proceeds = quantity * price
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        # Update position
        if quantity == position.quantity:
            # Selling entire position
            del self.positions[symbol]
            message = f"Sold entire {symbol} position: {quantity}@{price:.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.1f}%)"
        else:
            # Partial sale
            position.quantity -= quantity
            position.notes += f" | Sold {quantity}@{price} on {date}"
            message = f"Sold {quantity} {symbol}@{price:.2f} | Remaining: {position.quantity} | P&L: {pnl:+.2f} ({pnl_pct:+.1f}%)"
        
        # Update cash balance
        self.cash_balance += proceeds
        
        self.save_portfolio()
        return message
    
    def get_position_pnl(self, symbol: str, current_price: float) -> Dict:
        """Calculate P&L for a specific position"""
        
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        current_value = position.quantity * current_price
        cost_basis = position.quantity * position.avg_price
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
        
        return {
            'symbol': symbol,
            'quantity': position.quantity,
            'avg_price': position.avg_price,
            'current_price': current_price,
            'cost_basis': cost_basis,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'date_bought': position.date_bought
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get complete portfolio summary with P&L"""
        
        total_cost_basis = 0
        total_current_value = 0
        position_details = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_price)
            pnl_data = self.get_position_pnl(symbol, current_price)
            
            if pnl_data:
                position_details.append(pnl_data)
                total_cost_basis += pnl_data['cost_basis']
                total_current_value += pnl_data['current_value']
        
        total_unrealized_pnl = total_current_value - total_cost_basis
        total_unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        # Calculate realized P&L from transactions
        realized_pnl = self.calculate_realized_pnl()
        
        return {
            'positions': position_details,
            'total_cost_basis': total_cost_basis,
            'total_current_value': total_current_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': total_unrealized_pnl_pct,
            'realized_pnl': realized_pnl,
            'cash_balance': self.cash_balance,
            'total_portfolio_value': total_current_value + self.cash_balance
        }
    
    def calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from completed transactions"""
        
        realized_pnl = 0.0
        symbol_positions = {}  # Track cost basis per symbol
        
        for transaction in sorted(self.transactions, key=lambda x: x.date):
            symbol = transaction.symbol
            
            if symbol not in symbol_positions:
                symbol_positions[symbol] = {'quantity': 0, 'total_cost': 0}
            
            if transaction.action == 'buy':
                symbol_positions[symbol]['quantity'] += transaction.quantity
                symbol_positions[symbol]['total_cost'] += transaction.quantity * transaction.price
            
            elif transaction.action == 'sell':
                if symbol_positions[symbol]['quantity'] >= transaction.quantity:
                    # Calculate avg cost basis for sold shares
                    avg_cost = symbol_positions[symbol]['total_cost'] / symbol_positions[symbol]['quantity']
                    cost_of_sold = transaction.quantity * avg_cost
                    proceeds = transaction.quantity * transaction.price
                    
                    realized_pnl += proceeds - cost_of_sold
                    
                    # Update remaining position
                    remaining_quantity = symbol_positions[symbol]['quantity'] - transaction.quantity
                    symbol_positions[symbol]['quantity'] = remaining_quantity
                    symbol_positions[symbol]['total_cost'] = remaining_quantity * avg_cost
        
        return realized_pnl
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> str:
        """Set stop loss for a position"""
        
        if symbol not in self.positions:
            return f"Error: No position found for {symbol}"
        
        self.positions[symbol].stop_loss = stop_price
        self.save_portfolio()
        return f"Set stop loss for {symbol} at {stop_price:.2f}"
    
    def set_target_price(self, symbol: str, target_price: float) -> str:
        """Set target price for a position"""
        
        if symbol not in self.positions:
            return f"Error: No position found for {symbol}"
        
        self.positions[symbol].target_price = target_price
        self.save_portfolio()
        return f"Set target price for {symbol} at {target_price:.2f}"
    
    def get_transactions_history(self, symbol: Optional[str] = None, limit: int = 20) -> List[Transaction]:
        """Get transaction history, optionally filtered by symbol"""
        
        transactions = self.transactions
        
        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]
        
        # Sort by date (newest first) and limit results
        transactions = sorted(transactions, key=lambda x: x.date, reverse=True)
        return transactions[:limit]
    
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate portfolio performance metrics"""
        
        summary = self.get_portfolio_summary(current_prices)
        
        # Calculate additional metrics
        win_rate = self.calculate_win_rate()
        avg_win, avg_loss = self.calculate_avg_win_loss()
        
        return {
            **summary,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len([t for t in self.transactions if t.action == 'sell']),
            'number_of_positions': len(self.positions)
        }
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades"""
        
        completed_trades = self.get_completed_trades()
        if not completed_trades:
            return 0.0
        
        wins = sum(1 for trade in completed_trades if trade['pnl'] > 0)
        return (wins / len(completed_trades)) * 100
    
    def calculate_avg_win_loss(self) -> Tuple[float, float]:
        """Calculate average win and average loss"""
        
        completed_trades = self.get_completed_trades()
        
        wins = [trade['pnl'] for trade in completed_trades if trade['pnl'] > 0]
        losses = [trade['pnl'] for trade in completed_trades if trade['pnl'] < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    def get_completed_trades(self) -> List[Dict]:
        """Get list of completed trades with P&L"""
        
        # This is a simplified version - in practice, you'd want more sophisticated
        # trade matching (FIFO, LIFO, etc.)
        completed_trades = []
        
        # Group transactions by symbol
        symbol_transactions = {}
        for transaction in self.transactions:
            if transaction.symbol not in symbol_transactions:
                symbol_transactions[transaction.symbol] = []
            symbol_transactions[transaction.symbol].append(transaction)
        
        # For each symbol, match buys with sells
        for symbol, transactions in symbol_transactions.items():
            transactions = sorted(transactions, key=lambda x: x.date)
            
            position_quantity = 0
            position_cost = 0.0
            
            for transaction in transactions:
                if transaction.action == 'buy':
                    position_quantity += transaction.quantity
                    position_cost += transaction.quantity * transaction.price
                
                elif transaction.action == 'sell':
                    if position_quantity >= transaction.quantity:
                        avg_cost = position_cost / position_quantity
                        cost_basis = transaction.quantity * avg_cost
                        proceeds = transaction.quantity * transaction.price
                        pnl = proceeds - cost_basis
                        
                        completed_trades.append({
                            'symbol': symbol,
                            'quantity': transaction.quantity,
                            'buy_price': avg_cost,
                            'sell_price': transaction.price,
                            'pnl': pnl,
                            'sell_date': transaction.date
                        })
                        
                        # Update remaining position
                        position_quantity -= transaction.quantity
                        position_cost = position_quantity * avg_cost
        
        return completed_trades
    
    def save_portfolio(self):
        """Save portfolio to JSON file"""
        
        data = {
            'positions': {symbol: asdict(position) for symbol, position in self.positions.items()},
            'transactions': [asdict(transaction) for transaction in self.transactions],
            'cash_balance': self.cash_balance,
            'last_updated': dt.datetime.now().isoformat()
        }
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_portfolio(self):
        """Load portfolio from JSON file"""
        
        try:
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
            
            # Load positions
            self.positions = {}
            for symbol, pos_data in data.get('positions', {}).items():
                self.positions[symbol] = Position(**pos_data)
            
            # Load transactions
            self.transactions = []
            for trans_data in data.get('transactions', []):
                self.transactions.append(Transaction(**trans_data))
            
            # Load cash balance
            self.cash_balance = data.get('cash_balance', 0.0)
            
        except FileNotFoundError:
            # Initialize empty portfolio
            self.positions = {}
            self.transactions = []
            self.cash_balance = 0.0
    
    def add_cash(self, amount: float, notes: str = "Cash deposit") -> str:
        """Add cash to portfolio"""
        
        self.cash_balance += amount
        self.save_portfolio()
        return f"Added {amount:.2f} PKR to cash balance. New balance: {self.cash_balance:.2f}"
    
    def remove_cash(self, amount: float, notes: str = "Cash withdrawal") -> str:
        """Remove cash from portfolio"""
        
        if amount > self.cash_balance:
            return f"Error: Insufficient cash balance. Available: {self.cash_balance:.2f}"
        
        self.cash_balance -= amount
        self.save_portfolio()
        return f"Withdrew {amount:.2f} PKR. Remaining balance: {self.cash_balance:.2f}"