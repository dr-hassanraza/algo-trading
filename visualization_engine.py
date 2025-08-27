#!/usr/bin/env python3
"""
Professional Visualization Engine
=================================

Advanced charting and visualization capabilities for technical analysis,
signal visualization, and portfolio tracking.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from config_manager import get_config

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Visualization features disabled.")

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive charts disabled.")

class TechnicalChart:
    """Professional technical analysis charting"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.colors = {
            'green': '#26a69a',
            'red': '#ef5350',
            'blue': '#1e88e5',
            'orange': '#ff9800',
            'purple': '#9c27b0',
            'background': '#fafafa',
            'grid': '#e0e0e0'
        }
    
    def create_comprehensive_chart(self, df: pd.DataFrame, symbol: str, 
                                 signals: Dict = None, patterns: Dict = None) -> str:
        """
        Create comprehensive technical analysis chart
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for charting")
            return ""
        
        # Create subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        self._plot_price_with_indicators(ax1, df, symbol, signals, patterns)
        
        # Volume chart
        ax2 = fig.add_subplot(gs[1])
        self._plot_volume(ax2, df)
        
        # RSI chart
        ax3 = fig.add_subplot(gs[2])
        self._plot_rsi(ax3, df)
        
        # MACD chart
        ax4 = fig.add_subplot(gs[3])
        self._plot_macd(ax4, df)
        
        # Style and save
        plt.tight_layout()
        
        # Save chart
        chart_dir = Path(get_config('output.reports_directory', 'scan_reports')) / 'charts'
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol.replace('.', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png"
        filepath = chart_dir / filename
        
        plt.savefig(
            filepath, 
            dpi=get_config('output.chart_dpi', 130),
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        
        logger.info(f"Chart saved: {filepath}")
        return str(filepath)
    
    def _plot_price_with_indicators(self, ax, df: pd.DataFrame, symbol: str, 
                                   signals: Dict = None, patterns: Dict = None):
        """Plot main price chart with indicators"""
        
        dates = df['Date']
        
        # Candlestick chart
        self._plot_candlesticks(ax, df)
        
        # Moving averages
        if 'MA44' in df.columns:
            ax.plot(dates, df['MA44'], label='MA44', color=self.colors['blue'], linewidth=2)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_up', 'BB_lo', 'BB_mid']):
            ax.plot(dates, df['BB_mid'], label='BB Mid', color=self.colors['orange'], alpha=0.7)
            ax.fill_between(dates, df['BB_lo'], df['BB_up'], alpha=0.1, color=self.colors['orange'])
        
        # Support/Resistance levels
        if signals and 'support_resistance' in signals:
            sr = signals['support_resistance']
            if sr.get('nearest_support'):
                ax.axhline(y=sr['nearest_support'], color=self.colors['green'], 
                          linestyle='--', alpha=0.7, label='Support')
            if sr.get('nearest_resistance'):
                ax.axhline(y=sr['nearest_resistance'], color=self.colors['red'], 
                          linestyle='--', alpha=0.7, label='Resistance')
        
        # Signal markers
        if signals and 'signal_strength' in signals:
            signal_grade = signals['signal_strength']['grade']
            last_price = df['Close'].iloc[-1]
            last_date = dates.iloc[-1]
            
            color = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 
                    'D': 'orange', 'F': 'red'}.get(signal_grade, 'gray')
            
            ax.scatter([last_date], [last_price], s=100, c=color, 
                      marker='o', edgecolor='black', linewidth=2, 
                      label=f'Signal: {signal_grade}', zorder=5)
        
        # Candlestick patterns
        if patterns:
            self._mark_patterns(ax, df, patterns)
        
        ax.set_title(f'{symbol} - Technical Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (PKR)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        
        for i, (_, row) in enumerate(df.iterrows()):
            date = row['Date']
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            # Color based on direction
            color = self.colors['green'] if close > open_price else self.colors['red']
            
            # High-Low line
            ax.plot([date, date], [low, high], color='black', linewidth=0.5)
            
            # Body rectangle
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                           0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    
    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume chart"""
        
        dates = df['Date']
        volumes = df['Volume']
        
        # Color bars based on price direction
        colors = [self.colors['green'] if df['Close'].iloc[i] > df['Open'].iloc[i] 
                 else self.colors['red'] for i in range(len(df))]
        
        ax.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
        
        # Volume moving average
        if len(df) > 20:
            vol_ma = df['Volume'].rolling(20).mean()
            ax.plot(dates, vol_ma, color=self.colors['blue'], linewidth=2, label='Vol MA(20)')
        
        ax.set_ylabel('Volume', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rsi(self, ax, df: pd.DataFrame):
        """Plot RSI indicator"""
        
        if 'RSI' not in df.columns:
            return
        
        dates = df['Date']
        rsi = df['RSI']
        
        ax.plot(dates, rsi, color=self.colors['purple'], linewidth=2, label='RSI(14)')
        
        # Overbought/Oversold lines
        ax.axhline(y=70, color=self.colors['red'], linestyle='--', alpha=0.7, label='Overbought')
        ax.axhline(y=30, color=self.colors['green'], linestyle='--', alpha=0.7, label='Oversold')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        # Fill overbought/oversold areas
        ax.fill_between(dates, 70, 100, alpha=0.1, color=self.colors['red'])
        ax.fill_between(dates, 0, 30, alpha=0.1, color=self.colors['green'])
        
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_macd(self, ax, df: pd.DataFrame):
        """Plot MACD indicator"""
        
        if not all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            return
        
        dates = df['Date']
        
        # MACD lines
        ax.plot(dates, df['MACD'], color=self.colors['blue'], linewidth=2, label='MACD')
        ax.plot(dates, df['MACD_Signal'], color=self.colors['red'], linewidth=2, label='Signal')
        
        # Histogram
        colors = [self.colors['green'] if h > 0 else self.colors['red'] for h in df['MACD_Hist']]
        ax.bar(dates, df['MACD_Hist'], color=colors, alpha=0.7, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('MACD', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _mark_patterns(self, ax, df: pd.DataFrame, patterns: Dict):
        """Mark candlestick patterns on chart"""
        
        active_patterns = [name for name, active in patterns.items() if active]
        if not active_patterns:
            return
        
        last_date = df['Date'].iloc[-1]
        last_high = df['High'].iloc[-1]
        
        # Add pattern annotation
        pattern_text = ', '.join(active_patterns[:3])  # Limit to 3 patterns
        ax.annotate(f'Patterns: {pattern_text}', 
                   xy=(last_date, last_high), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

class PortfolioVisualizer:
    """Portfolio performance visualization"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_portfolio_dashboard(self, portfolio_data: Dict, performance_data: pd.DataFrame = None) -> str:
        """Create comprehensive portfolio dashboard"""
        
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Dashboard', fontsize=16, fontweight='bold')
        
        # Portfolio allocation pie chart
        self._plot_allocation(axes[0, 0], portfolio_data)
        
        # Performance over time
        if performance_data is not None:
            self._plot_performance(axes[0, 1], performance_data)
        
        # Risk metrics
        self._plot_risk_metrics(axes[1, 0], portfolio_data)
        
        # Sector allocation
        self._plot_sector_allocation(axes[1, 1], portfolio_data)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_dir = Path(get_config('output.reports_directory', 'scan_reports')) / 'dashboards'
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"portfolio_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png"
        filepath = dashboard_dir / filename
        
        plt.savefig(filepath, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)
    
    def _plot_allocation(self, ax, portfolio_data: Dict):
        """Plot portfolio allocation pie chart"""
        
        positions = portfolio_data.get('positions', [])
        if not positions:
            ax.text(0.5, 0.5, 'No positions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Portfolio Allocation')
            return
        
        symbols = [pos['symbol'] for pos in positions]
        values = [pos['current_value'] for pos in positions]
        
        ax.pie(values, labels=symbols, autopct='%1.1f%%', startangle=90, colors=self.colors)
        ax.set_title('Portfolio Allocation')
    
    def _plot_performance(self, ax, performance_data: pd.DataFrame):
        """Plot portfolio performance over time"""
        
        if 'date' in performance_data.columns and 'total_value' in performance_data.columns:
            ax.plot(performance_data['date'], performance_data['total_value'], 
                   linewidth=2, color='#1f77b4')
            ax.set_title('Portfolio Value Over Time')
            ax.set_ylabel('Value (PKR)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Portfolio Performance')
    
    def _plot_risk_metrics(self, ax, portfolio_data: Dict):
        """Plot risk metrics"""
        
        metrics = ['Total Risk %', 'Concentration %', 'Diversification']
        values = [
            portfolio_data.get('total_risk_pct', 0),
            portfolio_data.get('concentration_risk', 0),
            portfolio_data.get('diversification_score', 0)
        ]
        
        bars = ax.bar(metrics, values, color=['red', 'orange', 'green'])
        ax.set_title('Risk Metrics')
        ax.set_ylabel('Percentage')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_sector_allocation(self, ax, portfolio_data: Dict):
        """Plot sector allocation"""
        
        positions = portfolio_data.get('positions', [])
        sectors = {}
        
        for pos in positions:
            sector = pos.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + pos['current_value']
        
        if sectors:
            ax.pie(sectors.values(), labels=sectors.keys(), autopct='%1.1f%%', 
                  startangle=90, colors=self.colors)
        else:
            ax.text(0.5, 0.5, 'No sector data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Sector Allocation')

class DataExporter:
    """Export trading data to various formats"""
    
    def __init__(self):
        self.output_dir = Path(get_config('output.reports_directory', 'scan_reports'))
        self.export_formats = get_config('output.export_formats', ['csv', 'json'])
    
    def export_scan_results(self, results: pd.DataFrame, symbol: str, date: str) -> List[str]:
        """Export scan results to multiple formats"""
        
        exported_files = []
        
        # Create output directory
        output_dir = self.output_dir / date
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV export
        if 'csv' in self.export_formats:
            csv_file = output_dir / f'{symbol}_scan_results.csv'
            results.to_csv(csv_file, index=False)
            exported_files.append(str(csv_file))
            logger.info(f"Exported CSV: {csv_file}")
        
        # JSON export
        if 'json' in self.export_formats:
            json_file = output_dir / f'{symbol}_scan_results.json'
            results.to_json(json_file, orient='records', indent=2)
            exported_files.append(str(json_file))
            logger.info(f"Exported JSON: {json_file}")
        
        # Excel export (if openpyxl available)
        if 'excel' in self.export_formats:
            try:
                excel_file = output_dir / f'{symbol}_scan_results.xlsx'
                results.to_excel(excel_file, index=False)
                exported_files.append(str(excel_file))
                logger.info(f"Exported Excel: {excel_file}")
            except ImportError:
                logger.warning("openpyxl not available for Excel export")
        
        return exported_files
    
    def export_portfolio_report(self, portfolio_data: Dict, transactions: List[Dict]) -> str:
        """Export comprehensive portfolio report"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        report_dir = self.output_dir / 'portfolio_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report = {
            'timestamp': timestamp,
            'portfolio_summary': portfolio_data,
            'transactions': transactions,
            'performance_metrics': self._calculate_performance_metrics(transactions),
            'risk_analysis': self._analyze_portfolio_risk(portfolio_data)
        }
        
        # Export as JSON
        report_file = report_dir / f'portfolio_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Portfolio report exported: {report_file}")
        return str(report_file)
    
    def _calculate_performance_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate portfolio performance metrics"""
        
        if not transactions:
            return {}
        
        # Basic metrics calculation
        total_invested = sum(t['quantity'] * t['price'] for t in transactions if t['action'] == 'buy')
        total_proceeds = sum(t['quantity'] * t['price'] for t in transactions if t['action'] == 'sell')
        
        return {
            'total_invested': total_invested,
            'total_proceeds': total_proceeds,
            'realized_pnl': total_proceeds - total_invested,
            'number_of_trades': len(transactions)
        }
    
    def _analyze_portfolio_risk(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio risk metrics"""
        
        return {
            'total_risk_percentage': portfolio_data.get('total_risk_pct', 0),
            'concentration_risk': portfolio_data.get('concentration_risk', 0),
            'diversification_score': portfolio_data.get('diversification_score', 0),
            'risk_level': portfolio_data.get('risk_level', 'unknown')
        }

# Global instances
technical_chart = TechnicalChart()
portfolio_visualizer = PortfolioVisualizer()
data_exporter = DataExporter()

# Convenience functions
def create_chart(df: pd.DataFrame, symbol: str, signals: Dict = None, patterns: Dict = None) -> str:
    """Create technical analysis chart"""
    return technical_chart.create_comprehensive_chart(df, symbol, signals, patterns)

def create_portfolio_dashboard(portfolio_data: Dict, performance_data: pd.DataFrame = None) -> str:
    """Create portfolio dashboard"""
    return portfolio_visualizer.create_portfolio_dashboard(portfolio_data, performance_data)

def export_results(results: pd.DataFrame, symbol: str, date: str) -> List[str]:
    """Export scan results"""
    return data_exporter.export_scan_results(results, symbol, date)

# Test function
if __name__ == '__main__':
    print("🧪 Testing Visualization Engine...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.normal(0, 1, 100).cumsum(),
        'High': 100 + np.random.normal(2, 1, 100).cumsum(),
        'Low': 100 + np.random.normal(-2, 1, 100).cumsum(),
        'Close': 100 + np.random.normal(0, 1.5, 100).cumsum(),
        'Volume': np.random.randint(10000, 50000, 100),
        'RSI': np.random.uniform(20, 80, 100),
        'MA44': 100 + np.random.normal(0, 0.5, 100).cumsum()
    })
    
    if MATPLOTLIB_AVAILABLE:
        # Test chart creation
        chart_file = create_chart(sample_data, 'TEST.KAR')
        print(f"Chart created: {chart_file}")
        
        # Test data export
        exported_files = export_results(sample_data, 'TEST', '2024-08-19')
        print(f"Exported files: {exported_files}")
    else:
        print("⚠️ Matplotlib not available - skipping chart tests")
    
    print("✅ Visualization engine test complete!")