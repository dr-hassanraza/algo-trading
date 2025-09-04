#!/usr/bin/env python3
"""
Professional PSX Trading System - Complete Integration
=====================================================

Enterprise-grade algorithmic trading system with all professional enhancements:
- Advanced technical indicators (MACD, Stochastic, ADX, RSI)
- Candlestick pattern recognition
- Professional risk management with dynamic position sizing
- Multi-timeframe analysis
- Comprehensive logging and configuration
- Data export and visualization
- Portfolio management with performance tracking

Usage:
    python professional_trading_system.py --symbol UBL --analysis enhanced
    python professional_trading_system.py --portfolio-report
    python professional_trading_system.py --web-interface
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import json

# Import our enhanced modules
from config_manager import get_config, logger
from enhanced_signal_analyzer import enhanced_signal_analysis
from advanced_indicators import detect_candlestick_patterns, macd, stochastic, adx
from risk_manager import calculate_position_size, multi_timeframe_check, risk_manager
from visualization_engine import create_chart, create_portfolio_dashboard, export_results
from portfolio_manager import PortfolioManager
from quant_system_config import create_default_config

class ProfessionalTradingSystem:
    """Complete professional trading system"""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        self.logger = logging.getLogger(__name__)
        
        # Load system configuration
        self.config = create_default_config()
        
        # Common PSX symbols from configuration
        self.common_symbols = self.config.universe.common_symbols
        
        self.logger.info("Professional Trading System initialized")
    
    def enhanced_analysis(self, symbol: str, create_charts: bool = True, export: bool = True) -> Dict:
        """
        Perform comprehensive enhanced analysis
        """
        self.logger.info(f"Starting enhanced analysis for {symbol}")
        
        try:
            # Run enhanced signal analysis
            result = enhanced_signal_analysis(symbol)
            
            if 'error' in result:
                self.logger.error(f"Analysis failed for {symbol}: {result['error']}")
                return result
            
            # Add additional analysis
            signal_strength = result['signal_strength']
            
            # Calculate position sizing recommendation
            current_price = result['price']
            stop_loss = result['risk_management']['stop_loss']
            
            position_size = calculate_position_size(current_price, stop_loss)
            result['position_sizing'] = {
                'shares': position_size.shares,
                'investment_amount': position_size.investment_amount,
                'risk_amount': position_size.risk_amount,
                'risk_percentage': position_size.risk_percentage,
                'warnings': position_size.warnings
            }
            
            # Create visualizations
            if create_charts:
                try:
                    # This would create comprehensive charts
                    self.logger.info(f"Creating charts for {symbol}")
                    # chart_path = create_chart(enhanced_data, symbol, result)
                    # result['chart_path'] = chart_path
                except Exception as e:
                    self.logger.warning(f"Chart creation failed: {e}")
            
            # Export results
            if export:
                try:
                    self._export_analysis_report(symbol, result)
                except Exception as e:
                    self.logger.warning(f"Export failed: {e}")
            
            self.logger.info(f"Enhanced analysis completed for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis error for {symbol}: {e}")
            return {'error': str(e)}
    
    def portfolio_analysis(self) -> Dict:
        """
        Comprehensive portfolio analysis
        """
        self.logger.info("Starting portfolio analysis")
        
        if not self.portfolio_manager.positions:
            return {'message': 'No positions in portfolio'}
        
        # Get current prices (simplified - in production, fetch from API)
        current_prices = {}
        for symbol in self.portfolio_manager.positions.keys():
            # TODO: Replace with real-time price feed from PSX API
            current_prices[symbol] = self.portfolio_manager.positions[symbol].avg_price
        
        # Portfolio summary
        summary = self.portfolio_manager.get_portfolio_summary(current_prices)
        
        # Portfolio risk analysis
        portfolio_positions = []
        for symbol, position in self.portfolio_manager.positions.items():
            portfolio_positions.append({
                'symbol': symbol,
                'value': position.quantity * current_prices[symbol],
                'risk_amount': position.quantity * current_prices[symbol] * 0.02,  # 2% risk
                'sector': 'Financial'  # TODO: Fetch sector from market data API
            })
        
        portfolio_risk = risk_manager.calculate_portfolio_risk(portfolio_positions)
        
        # Performance metrics
        metrics = self.portfolio_manager.get_portfolio_metrics(current_prices)
        
        # Combine all analysis
        portfolio_analysis = {
            'summary': summary,
            'risk_analysis': portfolio_risk,
            'performance_metrics': metrics,
            'recommendations': self._get_portfolio_recommendations(summary, portfolio_risk)
        }
        
        self.logger.info("Portfolio analysis completed")
        return portfolio_analysis
    
    def scan_multiple_symbols(self, symbols: List[str], analysis_type: str = 'enhanced') -> Dict:
        """
        Scan multiple symbols and provide comparative analysis
        """
        self.logger.info(f"Scanning {len(symbols)} symbols with {analysis_type} analysis")
        
        results = {}
        
        for symbol in symbols:
            try:
                if analysis_type == 'enhanced':
                    result = self.enhanced_analysis(symbol, create_charts=False, export=False)
                else:
                    # TODO: Implement comprehensive multi-timeframe analysis
                    result = enhanced_signal_analysis(symbol)
                
                results[symbol] = result
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        # Comparative analysis
        comparative = self._create_comparative_analysis(results)
        
        return {
            'individual_results': results,
            'comparative_analysis': comparative,
            'scan_summary': self._create_scan_summary(results)
        }
    
    def risk_assessment(self, symbol: str, quantity: int, current_price: float) -> Dict:
        """
        Comprehensive risk assessment for a potential trade
        """
        self.logger.info(f"Risk assessment for {symbol}: {quantity} shares at {current_price}")
        
        try:
            # Enhanced analysis for risk factors
            analysis = enhanced_signal_analysis(symbol)
            
            if 'error' in analysis:
                return analysis
            
            # Position sizing
            stop_loss = analysis['risk_management']['stop_loss']
            position_size = calculate_position_size(current_price, stop_loss)
            
            # Risk metrics
            investment_amount = quantity * current_price
            risk_amount = quantity * abs(current_price - stop_loss)
            
            # Portfolio impact
            current_portfolio_value = sum(
                pos.quantity * pos.avg_price 
                for pos in self.portfolio_manager.positions.values()
            )
            
            portfolio_impact = (investment_amount / (current_portfolio_value + investment_amount)) * 100
            
            # Stress testing
            stress_scenarios = risk_manager.stress_test(investment_amount)
            
            risk_assessment = {
                'symbol': symbol,
                'quantity': quantity,
                'current_price': current_price,
                'investment_amount': investment_amount,
                'risk_amount': risk_amount,
                'risk_percentage': (risk_amount / investment_amount) * 100,
                'portfolio_impact': portfolio_impact,
                'recommended_position_size': position_size.shares,
                'signal_strength': analysis['signal_strength'],
                'stress_test': stress_scenarios,
                'recommendation': self._get_risk_recommendation(analysis, portfolio_impact)
            }
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'error': str(e)}
    
    def generate_trading_report(self, symbol: str = None) -> str:
        """
        Generate comprehensive trading report
        """
        self.logger.info("Generating trading report")
        
        report_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_status': 'operational',
            'portfolio_analysis': self.portfolio_analysis()
        }
        
        if symbol:
            report_data['symbol_analysis'] = self.enhanced_analysis(symbol)
        else:
            # Scan top symbols
            top_symbols = self.common_symbols[:5]
            report_data['market_scan'] = self.scan_multiple_symbols(top_symbols)
        
        # Export report
        report_path = self._export_trading_report(report_data)
        
        self.logger.info(f"Trading report generated: {report_path}")
        return report_path
    
    def _export_analysis_report(self, symbol: str, analysis: Dict):
        """Export individual analysis report"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path('scan_reports') / 'analysis_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f'{symbol}_analysis_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report exported: {report_file}")
    
    def _export_trading_report(self, report_data: Dict) -> str:
        """Export comprehensive trading report"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path('scan_reports') / 'trading_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f'trading_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(report_file)
    
    def _create_comparative_analysis(self, results: Dict) -> Dict:
        """Create comparative analysis of multiple symbols"""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results for comparison'}
        
        # Extract scores and grades
        scores = []
        grades = []
        symbols = []
        
        for symbol, result in valid_results.items():
            if 'signal_strength' in result:
                symbols.append(symbol)
                scores.append(result['signal_strength']['score'])
                grades.append(result['signal_strength']['grade'])
        
        if not scores:
            return {'error': 'No signal strength data available'}
        
        # Ranking
        symbol_scores = list(zip(symbols, scores, grades))
        symbol_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        return {
            'top_performers': symbol_scores[:3],
            'bottom_performers': symbol_scores[-3:],
            'average_score': sum(scores) / len(scores),
            'grade_distribution': {grade: grades.count(grade) for grade in set(grades)},
            'recommendations': [
                f"Top pick: {symbol_scores[0][0]} (Grade {symbol_scores[0][2]}, Score {symbol_scores[0][1]:.0f})" if symbol_scores else "No recommendations available"
            ]
        }
    
    def _create_scan_summary(self, results: Dict) -> Dict:
        """Create summary of scan results"""
        
        total_symbols = len(results)
        valid_results = sum(1 for result in results.values() if 'error' not in result)
        error_count = total_symbols - valid_results
        
        # Count signal grades
        grades = {}
        strong_buys = 0
        
        for result in results.values():
            if 'signal_strength' in result:
                grade = result['signal_strength']['grade']
                grades[grade] = grades.get(grade, 0) + 1
                
                if result['signal_strength']['recommendation'] in ['STRONG BUY', 'BUY']:
                    strong_buys += 1
        
        return {
            'total_symbols_scanned': total_symbols,
            'successful_analyses': valid_results,
            'errors': error_count,
            'grade_distribution': grades,
            'buy_candidates': strong_buys,
            'scan_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _get_portfolio_recommendations(self, summary: Dict, risk_analysis: Dict) -> List[str]:
        """Generate portfolio recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_analysis['risk_level'] == 'high':
            recommendations.append("🔴 High portfolio risk detected - consider reducing position sizes")
        
        if risk_analysis['concentration_risk'] > 30:
            recommendations.append("⚠️ High concentration risk - diversify across more positions")
        
        # Performance recommendations
        if summary['total_unrealized_pnl_pct'] < -10:
            recommendations.append("📉 Portfolio down >10% - review stop losses and risk management")
        
        # Position count recommendations
        position_count = len(summary['positions'])
        if position_count < 3:
            recommendations.append("📊 Consider adding more positions to improve diversification")
        elif position_count > 15:
            recommendations.append("🎯 Consider consolidating positions to reduce complexity")
        
        if not recommendations:
            recommendations.append("✅ Portfolio appears well-balanced")
        
        return recommendations
    
    def _get_risk_recommendation(self, analysis: Dict, portfolio_impact: float) -> str:
        """Get risk-based trading recommendation"""
        
        signal_grade = analysis['signal_strength']['grade']
        signal_score = analysis['signal_strength']['score']
        
        if portfolio_impact > 25:
            return "AVOID - Position too large for portfolio"
        elif signal_grade in ['A', 'B'] and signal_score > 70:
            return "STRONG BUY - High probability setup"
        elif signal_grade in ['B', 'C'] and signal_score > 50:
            return "BUY - Moderate probability setup"
        elif signal_grade == 'D':
            return "HOLD - Wait for better setup"
        else:
            return "AVOID - Low probability setup"

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description='Professional PSX Trading System')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to analyze')
    parser.add_argument('--analysis', '-a', choices=['basic', 'enhanced'], default='enhanced',
                       help='Analysis type')
    parser.add_argument('--portfolio-report', '-p', action='store_true',
                       help='Generate portfolio report')
    parser.add_argument('--scan-symbols', '-m', nargs='+', 
                       help='Scan multiple symbols')
    parser.add_argument('--risk-assessment', '-r', action='store_true',
                       help='Perform risk assessment')
    parser.add_argument('--quantity', '-q', type=int, default=100,
                       help='Quantity for risk assessment')
    parser.add_argument('--price', type=float, help='Price for risk assessment')
    parser.add_argument('--web-interface', '-w', action='store_true',
                       help='Launch web interface')
    parser.add_argument('--export', action='store_true', default=True,
                       help='Export results')
    parser.add_argument('--charts', action='store_true', default=True,
                       help='Generate charts')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ProfessionalTradingSystem()
    
    try:
        if args.web_interface:
            print("🚀 Launching web interface...")
            
            # Check if streamlit_app.py exists
            streamlit_files = ['streamlit_app.py', 'streamlit_professional_dashboard.py']
            streamlit_file = None
            
            for file in streamlit_files:
                if os.path.exists(file):
                    streamlit_file = file
                    break
            
            if streamlit_file:
                print(f"📊 Starting Streamlit dashboard: {streamlit_file}")
                try:
                    # Launch Streamlit in a subprocess
                    subprocess.run(["streamlit", "run", streamlit_file], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to launch Streamlit: {e}")
                    print("💡 Make sure Streamlit is installed: pip install streamlit")
                    print(f"📝 Fallback: Run manually with 'streamlit run {streamlit_file}'")
                except FileNotFoundError:
                    print("❌ Streamlit not found in PATH")
                    print("💡 Install Streamlit: pip install streamlit")
                    print(f"📝 Fallback: Run manually with 'streamlit run {streamlit_file}'")
            else:
                print("❌ No Streamlit app file found")
                print("💡 Looking for: streamlit_app.py or streamlit_professional_dashboard.py")
                print("📝 Please ensure the Streamlit app file exists in the current directory")
            
        elif args.portfolio_report:
            print("📊 Generating portfolio report...")
            report_path = system.generate_trading_report()
            print(f"✅ Report generated: {report_path}")
            
        elif args.scan_symbols:
            print(f"🔍 Scanning symbols: {args.scan_symbols}")
            results = system.scan_multiple_symbols(args.scan_symbols, args.analysis)
            
            print("\n📈 Scan Results:")
            for symbol, result in results['individual_results'].items():
                if 'error' not in result:
                    signal = result['signal_strength']
                    print(f"  {symbol}: Grade {signal['grade']} ({signal['score']:.0f}/100) - {signal['recommendation']}")
                else:
                    print(f"  {symbol}: Error - {result['error']}")
            
            print(f"\n📊 Summary: {results['scan_summary']['buy_candidates']} buy candidates found")
            
        elif args.symbol:
            if args.risk_assessment:
                price = args.price or 100.0
                print(f"🎯 Risk assessment for {args.symbol}")
                risk_result = system.risk_assessment(args.symbol, args.quantity, price)
                
                if 'error' not in risk_result:
                    print(f"Investment: {risk_result['investment_amount']:,.0f} PKR")
                    print(f"Risk: {risk_result['risk_amount']:,.0f} PKR ({risk_result['risk_percentage']:.1f}%)")
                    print(f"Recommendation: {risk_result['recommendation']}")
                else:
                    print(f"Error: {risk_result['error']}")
            else:
                print(f"📊 Analyzing {args.symbol} with {args.analysis} analysis...")
                result = system.enhanced_analysis(args.symbol, args.charts, args.export)
                
                if 'error' not in result:
                    signal = result['signal_strength']
                    print(f"\n🎯 Analysis Results:")
                    print(f"  Grade: {signal['grade']} ({signal['score']:.0f}/100)")
                    print(f"  Price: {result['price']:.2f} PKR")
                    print(f"  Recommendation: {signal['recommendation']}")
                    print(f"  RSI: {result['technical_data']['rsi']:.1f}")
                    print(f"  Volume: {result['technical_data']['volume_ratio']:.1f}x average")
                    
                    risk = result['risk_management']
                    print(f"\n🛡️ Risk Management:")
                    print(f"  Stop Loss: {risk['stop_loss']:.2f} (-{risk['stop_loss_pct']:.1f}%)")
                    print(f"  Target: {risk['target1']:.2f} (+{risk['target1_pct']:.1f}%)")
                    print(f"  Risk/Reward: {risk['risk_reward_ratio']:.2f}")
                    
                    if 'position_sizing' in result:
                        pos = result['position_sizing']
                        print(f"\n📏 Position Sizing:")
                        print(f"  Recommended shares: {pos['shares']}")
                        print(f"  Investment: {pos['investment_amount']:,.0f} PKR")
                        print(f"  Risk: {pos['risk_amount']:,.0f} PKR ({pos['risk_percentage']:.1f}%)")
                else:
                    print(f"❌ Analysis failed: {result['error']}")
        else:
            # Default: generate comprehensive report
            print("📋 Generating comprehensive trading report...")
            report_path = system.generate_trading_report()
            print(f"✅ Report generated: {report_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}", exc_info=True)

if __name__ == '__main__':
    main()