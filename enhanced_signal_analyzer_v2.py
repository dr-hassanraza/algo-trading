#!/usr/bin/env python3
"""
Enhanced Signal Analyzer V2 - Multi-Source Integration
=====================================================

Integrates multiple data sources for comprehensive PSX analysis:
1. EODHD API (primary historical data)
2. yfinance (backup historical data) 
3. Multi-source real-time data
4. Currency context via Node.js bridge
5. Enhanced fundamental analysis with sector insights

This version provides the most comprehensive analysis available.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import datetime as dt
import logging

# Import our enhanced modules
from enhanced_data_fetcher import EnhancedDataFetcher
from bridge_client import get_currency_context
from enhanced_signal_analyzer import (
    get_sector_info, get_market_sentiment, analyze_news_sentiment,
    estimate_fundamentals, calculate_risk_management,
    enhanced_indicators, volume_analysis, find_support_resistance
)

logger = logging.getLogger(__name__)

class EnhancedSignalAnalyzerV2:
    """Enhanced signal analyzer with multi-source data integration"""
    
    def __init__(self):
        self.data_fetcher = EnhancedDataFetcher()
        self.currency_cache = {}
        self.currency_cache_time = 0
        self.cache_expiry = 1800  # 30 minutes
    
    def analyze_with_multi_source(self, symbol: str, days: int = 260) -> Dict:
        """Comprehensive analysis using all available data sources"""
        
        try:
            logger.info(f"Starting multi-source analysis for {symbol}")
            
            # Get base symbol for various APIs
            base_symbol = symbol.upper().split('.')[0]
            formatted_symbol = f"{base_symbol}.KAR"
            
            # Step 1: Get historical price data
            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=days+40)
            
            try:
                price_data = self.data_fetcher.fetch(formatted_symbol, start_date, end_date)
                if price_data.empty:
                    return {'error': f'No historical data available for {symbol}'}
            except Exception as e:
                return {'error': f'Failed to fetch historical data: {str(e)}'}
            
            # Step 2: Get real-time data for current price verification
            real_time_data = self.data_fetcher.get_real_time_data(formatted_symbol)
            
            # Step 3: Calculate enhanced technical indicators
            enhanced_data = enhanced_indicators(price_data)
            if enhanced_data.empty:
                return {'error': 'Insufficient data for technical analysis'}
            
            latest = enhanced_data.iloc[-1]
            
            # Use real-time price if available and recent
            if real_time_data and real_time_data.price > 0:
                current_price = real_time_data.price
                # Update the latest row with real-time data
                latest = latest.copy()
                latest['Close'] = current_price
            else:
                current_price = float(latest['Close'])
            
            # Step 4: Get contextual data
            sector_info = get_sector_info(base_symbol)
            market_sentiment = get_market_sentiment()
            news_sentiment = analyze_news_sentiment(base_symbol)
            
            # Step 5: Get currency context
            currency_context = self._get_currency_context()
            
            # Step 6: Enhanced fundamental analysis
            fundamentals = self._get_enhanced_fundamentals(formatted_symbol, sector_info, currency_context)
            
            # Step 7: Volume and support/resistance analysis
            volume_data = volume_analysis(enhanced_data)
            support_resistance = find_support_resistance(enhanced_data)
            
            # Step 8: Calculate enhanced signal strength
            signal_strength = self._calculate_enhanced_signal_strength(
                latest, volume_data, support_resistance, fundamentals,
                market_sentiment, news_sentiment, sector_info, currency_context
            )
            
            # Step 9: Risk management with currency consideration
            risk_mgmt = self._calculate_enhanced_risk_management(
                latest, support_resistance, currency_context
            )
            
            # Step 10: Compile comprehensive result
            enhanced_signal = {
                'symbol': formatted_symbol,
                'base_symbol': base_symbol,
                'date': latest['Date'].strftime('%Y-%m-%d') if 'Date' in latest else dt.date.today().strftime('%Y-%m-%d'),
                'price': current_price,
                'signal_strength': signal_strength,
                'risk_management': risk_mgmt,
                'technical_data': {
                    'ma44': float(latest['MA44']),
                    'bb_pctb': float(latest['BB_pctB']),
                    'rsi': float(latest['RSI']),
                    'atr': float(latest['ATR']),
                    'macd_hist': float(latest.get('MACD_Hist', 0)),
                    'stoch': float(latest.get('Stoch', 50)),
                    'adx': float(latest.get('ADX', 0)),
                    'volume_ratio': volume_data['volume_ratio'],
                    'ma44_trend': latest.get('ma44_trend', 'unknown'),
                    'ma44_slope': float(latest.get('MA44_slope10', 0)),
                    'trend_ma44_up': bool(latest.get('trend_ma44_up', False)),
                    'close_gt_ma44': bool(latest.get('close_gt_ma44', False))
                },
                'fundamentals': fundamentals,
                'support_resistance': support_resistance,
                'volume_analysis': volume_data,
                'sector_analysis': sector_info,
                'market_sentiment': market_sentiment,
                'news_sentiment': news_sentiment,
                'currency_context': currency_context,
                'data_sources': self._get_data_source_info(),
                'analysis_summary': {
                    'total_factors_analyzed': len(signal_strength.get('factors', [])),
                    'data_sources_used': len([s for s in self.data_fetcher.get_source_reliability().keys() if self.data_fetcher.source_success_rates[s]['total'] > 0]),
                    'currency_data_available': bool(currency_context),
                    'real_time_data_available': real_time_data is not None,
                    'analysis_version': 'v2_multi_source',
                    'enhanced_features': [
                        'multi_source_data', 'currency_context', 'enhanced_fundamentals', 
                        'sector_analysis', 'market_sentiment', 'news_sentiment', 'real_time_verification'
                    ]
                }
            }
            
            logger.info(f\"Multi-source analysis completed for {symbol}\")
            return enhanced_signal
            
        except Exception as e:
            logger.error(f\"Multi-source analysis failed for {symbol}: {e}\")
            return {'error': str(e)}
    
    def _get_currency_context(self) -> Dict:
        \"\"\"Get currency context with caching\"\"\"
        
        import time
        current_time = time.time()
        
        # Use cached data if available and fresh
        if self.currency_cache and (current_time - self.currency_cache_time) < self.cache_expiry:
            return self.currency_cache
        
        try:
            currency_context = get_currency_context()
            if currency_context:
                self.currency_cache = {
                    'usd_pkr_rate': currency_context.get('usd-pkr', 280.0),
                    'eur_pkr_rate': currency_context.get('eur-pkr', 310.0),
                    'gbp_pkr_rate': currency_context.get('gbp-pkr', 360.0),
                    'currency_strength': self._assess_currency_strength(currency_context),
                    'last_updated': dt.datetime.now().isoformat()
                }
                self.currency_cache_time = current_time
            else:
                # Fallback to estimated rates
                self.currency_cache = {
                    'usd_pkr_rate': 280.0,
                    'eur_pkr_rate': 310.0,
                    'gbp_pkr_rate': 360.0,
                    'currency_strength': 'neutral',
                    'last_updated': dt.datetime.now().isoformat(),
                    'note': 'estimated_rates'
                }
                
        except Exception as e:
            logger.warning(f\"Failed to get currency context: {e}\")
            self.currency_cache = {}
        
        return self.currency_cache
    
    def _assess_currency_strength(self, rates: Dict[str, float]) -> str:
        \"\"\"Assess PKR strength based on rates\"\"\"
        
        # Historical reference rates (approximate)
        ref_rates = {
            'usd-pkr': 280.0,
            'eur-pkr': 310.0,
            'gbp-pkr': 360.0
        }
        
        strength_score = 0
        
        for pair, current_rate in rates.items():
            if pair in ref_rates:
                ref_rate = ref_rates[pair]
                # Lower PKR rate = stronger PKR
                if current_rate < ref_rate * 0.98:
                    strength_score += 1  # PKR strengthening
                elif current_rate > ref_rate * 1.02:
                    strength_score -= 1  # PKR weakening
        
        if strength_score >= 2:
            return 'strong'
        elif strength_score >= 1:
            return 'strengthening'
        elif strength_score <= -2:
            return 'weak'
        elif strength_score <= -1:
            return 'weakening'
        else:
            return 'neutral'
    
    def _get_enhanced_fundamentals(self, symbol: str, sector_info: Dict, currency_context: Dict) -> Dict:
        \"\"\"Get enhanced fundamental analysis with currency adjustment\"\"\"
        
        # Get base fundamentals
        fundamentals = estimate_fundamentals(symbol.split('.')[0])
        
        # Enhance with currency context
        if currency_context:
            usd_pkr = currency_context.get('usd_pkr_rate', 280.0)
            currency_strength = currency_context.get('currency_strength', 'neutral')
            
            # Adjust fundamentals based on currency strength
            if currency_strength in ['weak', 'weakening']:
                # Weak PKR benefits exporters
                if sector_info.get('sector') in ['Textiles', 'Cement', 'Fertilizer']:
                    fundamentals['currency_adjustment'] = 'positive_export_boost'
                    fundamentals['adjusted_pe'] = fundamentals.get('PE', 10) * 0.9  # Lower PE due to earnings boost
                else:
                    fundamentals['currency_adjustment'] = 'negative_import_cost'
            elif currency_strength in ['strong', 'strengthening']:
                # Strong PKR benefits importers
                if sector_info.get('sector') in ['Oil & Gas', 'Automobiles']:
                    fundamentals['currency_adjustment'] = 'positive_import_savings'
                else:
                    fundamentals['currency_adjustment'] = 'negative_export_pressure'
            else:
                fundamentals['currency_adjustment'] = 'neutral'
        
        return fundamentals
    
    def _calculate_enhanced_signal_strength(self, row: pd.Series, volume_data: Dict, support_resistance: Dict, 
                                          fundamentals: Dict, market_sentiment: Dict, news_sentiment: Dict, 
                                          sector_info: Dict, currency_context: Dict) -> Dict:
        \"\"\"Enhanced signal strength calculation with currency factors\"\"\"
        
        # Start with base calculation (from enhanced_signal_analyzer)
        from enhanced_signal_analyzer import calculate_signal_strength
        
        base_signal = calculate_signal_strength(
            row, volume_data, support_resistance, fundamentals,
            market_sentiment, news_sentiment, sector_info
        )
        
        # Add currency-specific factors
        score = base_signal['score']
        factors = base_signal['factors'].copy()
        
        # Currency impact (5 points)
        if currency_context:
            currency_strength = currency_context.get('currency_strength', 'neutral')
            sector = sector_info.get('sector', 'Unknown')
            
            export_sectors = ['Textiles', 'Cement', 'Fertilizer', 'Food']
            import_sectors = ['Oil & Gas', 'Automobiles', 'Technology']
            
            if currency_strength in ['weak', 'weakening'] and sector in export_sectors:
                score += 5
                factors.append(f\"Weak PKR benefits {sector} exports (+5)\")
            elif currency_strength in ['strong', 'strengthening'] and sector in import_sectors:
                score += 3
                factors.append(f\"Strong PKR reduces {sector} costs (+3)\")
            elif currency_strength in ['weak', 'weakening'] and sector in import_sectors:
                score -= 3
                factors.append(f\"Weak PKR increases {sector} costs (-3)\")
            elif currency_strength in ['strong', 'strengthening'] and sector in export_sectors:
                score -= 2
                factors.append(f\"Strong PKR pressures {sector} exports (-2)\")
        
        # Multi-source data quality bonus (2 points)
        data_quality = self._assess_data_quality()
        if data_quality >= 0.8:
            score += 2
            factors.append(f\"High data quality confidence (+2)\")
        elif data_quality < 0.5:
            score -= 1
            factors.append(f\"Limited data quality (-1)\")
        
        # Normalize and update
        final_score = max(0, min(100, score))
        
        # Update recommendation based on enhanced score
        if final_score >= 85:
            grade = \"A+\"
            recommendation = \"STRONG BUY\"
        elif final_score >= 75:
            grade = \"A\"
            recommendation = \"STRONG BUY\"
        elif final_score >= 65:
            grade = \"B+\"
            recommendation = \"BUY\"
        elif final_score >= 55:
            grade = \"B\"
            recommendation = \"BUY\"
        elif final_score >= 45:
            grade = \"C+\"
            recommendation = \"WEAK BUY\"
        elif final_score >= 35:
            grade = \"C\"
            recommendation = \"WEAK BUY\"
        elif final_score >= 25:
            grade = \"D\"
            recommendation = \"HOLD\"
        else:
            grade = \"F\"
            recommendation = \"AVOID\"
        
        return {
            'score': final_score,
            'grade': grade,
            'recommendation': recommendation,
            'factors': factors,
            'rsi': base_signal['rsi'],
            'volume_ratio': base_signal['volume_ratio'],
            'market_sentiment': base_signal['market_sentiment'],
            'news_sentiment': base_signal['news_sentiment'],
            'sector': base_signal['sector'],
            'currency_impact': currency_context.get('currency_strength', 'neutral'),
            'data_quality_score': data_quality
        }
    
    def _calculate_enhanced_risk_management(self, row: pd.Series, support_resistance: Dict, currency_context: Dict) -> Dict:
        \"\"\"Enhanced risk management with currency considerations\"\"\"
        
        # Get base risk management
        base_risk = calculate_risk_management(row, support_resistance)
        
        # Adjust for currency volatility
        if currency_context:
            currency_strength = currency_context.get('currency_strength', 'neutral')
            
            # Increase risk for currency-sensitive scenarios
            if currency_strength in ['weak', 'strong']:  # High volatility
                base_risk['currency_risk_adjustment'] = 0.5  # Additional 0.5% risk
                base_risk['adjusted_stop_loss_pct'] = base_risk['stop_loss_pct'] + 0.5
            else:
                base_risk['currency_risk_adjustment'] = 0.0
                base_risk['adjusted_stop_loss_pct'] = base_risk['stop_loss_pct']
        
        return base_risk
    
    def _assess_data_quality(self) -> float:
        \"\"\"Assess overall data quality from multiple sources\"\"\"
        
        reliability = self.data_fetcher.get_source_reliability()
        if not reliability:
            return 0.5
        
        # Weight sources by importance
        weights = {'eodhd': 0.5, 'yfinance': 0.3, 'multi_source': 0.2}
        
        weighted_score = 0
        total_weight = 0
        
        for source, rate in reliability.items():
            if source in weights:
                weighted_score += rate * weights[source]
                total_weight += weights[source]
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _get_data_source_info(self) -> Dict:
        \"\"\"Get information about data sources used\"\"\"
        
        reliability = self.data_fetcher.get_source_reliability()
        cache_info = self.data_fetcher.get_cache_info()
        
        return {
            'source_reliability': reliability,
            'cache_info': cache_info,
            'preferred_source': max(reliability.items(), key=lambda x: x[1])[0] if reliability else 'unknown'
        }

# Main analysis function for easy integration
def enhanced_signal_analysis_v2(symbol: str, days: int = 260) -> Dict:
    \"\"\"Enhanced signal analysis with multi-source data integration\"\"\"
    
    analyzer = EnhancedSignalAnalyzerV2()
    return analyzer.analyze_with_multi_source(symbol, days)

# Test the enhanced analyzer
if __name__ == '__main__':
    import time
    
    print(\"🚀 Testing Enhanced Signal Analyzer V2\")
    print(\"=\" * 50)
    
    test_symbols = ['UBL', 'MCB', 'LUCK']
    
    for symbol in test_symbols:
        print(f\"\
📊 Analyzing {symbol} with multi-source integration...\")
        start_time = time.time()
        
        result = enhanced_signal_analysis_v2(symbol)
        
        analysis_time = time.time() - start_time
        
        if 'error' not in result:
            signal = result['signal_strength']
            summary = result['analysis_summary']
            
            print(f\"   ✅ {symbol}: Grade {signal['grade']} ({signal['score']:.1f}/100) - {signal['recommendation']}\")
            print(f\"   💰 Price: {result['price']:.2f} PKR\")
            print(f\"   🏢 Sector: {result['sector_analysis']['sector']}\")
            print(f\"   💱 Currency Impact: {signal['currency_impact']}\")
            print(f\"   📊 Data Sources: {summary['data_sources_used']} sources used\")
            print(f\"   ⏱️  Analysis Time: {analysis_time:.2f}s\")
            print(f\"   🎯 Top Factor: {signal['factors'][0] if signal['factors'] else 'N/A'}\")
        else:
            print(f\"   ❌ {symbol}: {result['error']}\")
    
    print(f\"\
🏁 Enhanced V2 analysis completed!\")