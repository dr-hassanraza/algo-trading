#!/usr/bin/env python3
"""
Advanced Feature Engineering Pipeline for PSX Quantitative Trading
==================================================================

Professional-grade feature engineering system implementing institutional
best practices for cross-sectional equity analysis.

Features:
- Price-based momentum and mean reversion
- Volatility-adjusted returns
- Fundamental value, quality, and growth metrics
- Cross-sectional ranking and sector neutralization
- Robust outlier handling and winsorization
- Time-series feature stability validation
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import logging

from quant_system_config import SystemConfig, FeatureConfig
from enhanced_data_fetcher import EnhancedDataFetcher

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for cross-sectional equity analysis"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.feature_config = self.config.features
        self.data_fetcher = EnhancedDataFetcher()
        
        # Feature metadata tracking
        self.feature_definitions = {}
        self.feature_importance_history = {}
        self.feature_stability_scores = {}
        
        # Scalers for different feature types
        self.price_scaler = RobustScaler()
        self.fundamental_scaler = QuantileTransformer(n_quantiles=100)
        self.cross_sectional_scaler = RobustScaler()
        
        # Sector mapping (PSX sectors)
        self.psx_sectors = {
            # Banking & Finance
            'UBL': 'Banking', 'MCB': 'Banking', 'HBL': 'Banking', 'ABL': 'Banking',
            'NBP': 'Banking', 'BOP': 'Banking', 'BAFL': 'Banking', 'MEBL': 'Banking',
            
            # Oil & Gas
            'PPL': 'Oil_Gas', 'OGDC': 'Oil_Gas', 'POL': 'Oil_Gas', 'MARI': 'Oil_Gas',
            'MPCL': 'Oil_Gas', 'PSO': 'Oil_Gas', 'SNGP': 'Oil_Gas', 'SSGC': 'Oil_Gas',
            
            # Cement
            'LUCK': 'Cement', 'DGKC': 'Cement', 'FCCL': 'Cement', 'CHCC': 'Cement',
            'MLCF': 'Cement', 'PIOC': 'Cement', 'KOHC': 'Cement',
            
            # Chemicals & Fertilizer
            'ENGRO': 'Chemical', 'FFC': 'Chemical', 'FATIMA': 'Chemical', 'ICI': 'Chemical',
            
            # Power & Energy
            'KTML': 'Power', 'KAPCO': 'Power', 'HUBCO': 'Power', 'KEL': 'Power',
            
            # Technology & Telecom
            'PTCL': 'Telecom', 'TRG': 'Technology', 'SYSTEMS': 'Technology',
            
            # Textiles
            'ARPL': 'Textile', 'AIRLINK': 'Textile', 'WTL': 'Textile',
            
            # Others
            'NESTLE': 'FMCG', 'LOTTE': 'FMCG', 'PTC': 'FMCG'
        }
    
    def create_price_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based momentum and mean reversion features"""
        
        features_df = pd.DataFrame(index=prices_df.index)
        
        # Momentum features (1m, 3m, 6m)
        for period in self.feature_config.momentum_periods:
            # Raw momentum
            momentum = prices_df['Close'].pct_change(period)
            features_df[f'momentum_{period}d'] = momentum
            
            # Volatility-adjusted momentum
            vol = prices_df['Close'].pct_change().rolling(period).std()
            features_df[f'momentum_{period}d_vol_adj'] = momentum / (vol + 1e-8)
            
            # Risk-adjusted momentum (Sharpe-like)
            returns = prices_df['Close'].pct_change()
            rolling_mean = returns.rolling(period).mean()
            rolling_std = returns.rolling(period).std()
            features_df[f'momentum_{period}d_sharpe'] = rolling_mean / (rolling_std + 1e-8)
        
        # Mean reversion features (5d, 10d)
        for period in self.feature_config.reversal_periods:
            # Short-term reversal
            reversal = -prices_df['Close'].pct_change(period)
            features_df[f'reversal_{period}d'] = reversal
            
            # RSI-like mean reversion
            price_changes = prices_df['Close'].diff()
            gains = price_changes.where(price_changes > 0, 0).rolling(period).mean()
            losses = -price_changes.where(price_changes < 0, 0).rolling(period).mean()
            rs = gains / (losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features_df[f'rsi_{period}d'] = (rsi - 50) / 50  # Center around 0
        
        # Volatility features
        for period in self.feature_config.volatility_periods:
            # Realized volatility
            returns = prices_df['Close'].pct_change()
            vol = returns.rolling(period).std() * np.sqrt(252)  # Annualized
            features_df[f'volatility_{period}d'] = vol
            
            # Volatility of volatility
            vol_of_vol = vol.rolling(period//2).std()
            features_df[f'vol_of_vol_{period}d'] = vol_of_vol
            
            # Downside deviation
            negative_returns = returns.where(returns < 0, 0)
            downside_vol = negative_returns.rolling(period).std() * np.sqrt(252)
            features_df[f'downside_vol_{period}d'] = downside_vol
        
        # Technical indicators
        if self.feature_config.enable_rsi:
            # 14-day RSI
            delta = prices_df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features_df['rsi_14'] = (rsi - 50) / 50
        
        if self.feature_config.enable_macd:
            # MACD
            ema12 = prices_df['Close'].ewm(span=12).mean()
            ema26 = prices_df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features_df['macd'] = macd / prices_df['Close']  # Normalize
            features_df['macd_signal'] = signal / prices_df['Close']
            features_df['macd_histogram'] = (macd - signal) / prices_df['Close']
        
        if self.feature_config.enable_bollinger:
            # Bollinger Bands
            sma20 = prices_df['Close'].rolling(20).mean()
            std20 = prices_df['Close'].rolling(20).std()
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            
            # %B (position within bands)
            features_df['bb_percent_b'] = (prices_df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            
            # Bandwidth
            features_df['bb_bandwidth'] = (bb_upper - bb_lower) / sma20
        
        if self.feature_config.enable_atr:
            # Average True Range
            high_low = prices_df['High'] - prices_df['Low']
            high_close = np.abs(prices_df['High'] - prices_df['Close'].shift())
            low_close = np.abs(prices_df['Low'] - prices_df['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean()
            features_df['atr_14'] = atr / prices_df['Close']  # Normalize by price
        
        # Volume-based features
        if 'Volume' in prices_df.columns:
            # Volume momentum
            vol_sma = prices_df['Volume'].rolling(20).mean()
            features_df['volume_ratio'] = prices_df['Volume'] / (vol_sma + 1)
            
            # Price-Volume relationship
            price_change = prices_df['Close'].pct_change()
            volume_change = prices_df['Volume'].pct_change()
            features_df['price_volume_corr'] = price_change.rolling(20).corr(volume_change)
            
            # On-Balance Volume
            obv = (np.sign(price_change) * prices_df['Volume']).fillna(0).cumsum()
            obv_sma = obv.rolling(20).mean()
            features_df['obv_ratio'] = obv / (obv_sma + 1)
        
        # Price pattern features
        # Gap features
        if all(col in prices_df.columns for col in ['Open', 'Close']):
            prev_close = prices_df['Close'].shift(1)
            gap = (prices_df['Open'] - prev_close) / prev_close
            features_df['gap'] = gap
            
            # Intraday return
            intraday_return = (prices_df['Close'] - prices_df['Open']) / prices_df['Open']
            features_df['intraday_return'] = intraday_return
            
            # High-Low spread
            hl_spread = (prices_df['High'] - prices_df['Low']) / prices_df['Open']
            features_df['hl_spread'] = hl_spread
        
        # Seasonality features
        features_df['day_of_week'] = pd.to_datetime(prices_df.index).dayofweek
        features_df['month'] = pd.to_datetime(prices_df.index).month
        
        # Encode cyclical features
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Drop original categorical features
        features_df = features_df.drop(['day_of_week', 'month'], axis=1)
        
        return features_df
    
    def create_fundamental_features(self, fundamental_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create fundamental value, quality, and growth features"""
        
        if not fundamental_data:
            return pd.DataFrame()
        
        # This would typically use actual fundamental data
        # For demo purposes, creating placeholder structure
        
        features_list = []
        
        for symbol, fund_df in fundamental_data.items():
            if fund_df.empty:
                continue
                
            symbol_features = pd.DataFrame(index=fund_df.index)
            symbol_features['symbol'] = symbol
            
            # Value metrics
            if self.feature_config.enable_value_metrics:
                if 'market_cap' in fund_df.columns and 'earnings' in fund_df.columns:
                    symbol_features['pe_ratio'] = fund_df['market_cap'] / (fund_df['earnings'] + 1e-8)
                    symbol_features['earnings_yield'] = fund_df['earnings'] / (fund_df['market_cap'] + 1e-8)
                
                if 'book_value' in fund_df.columns:
                    symbol_features['pb_ratio'] = fund_df['market_cap'] / (fund_df['book_value'] + 1e-8)
                    symbol_features['book_to_market'] = fund_df['book_value'] / (fund_df['market_cap'] + 1e-8)
                
                if 'free_cash_flow' in fund_df.columns:
                    symbol_features['fcf_yield'] = fund_df['free_cash_flow'] / (fund_df['market_cap'] + 1e-8)
            
            # Quality metrics
            if self.feature_config.enable_quality_metrics:
                if 'roe' in fund_df.columns:
                    symbol_features['roe'] = fund_df['roe']
                    
                if 'debt' in fund_df.columns and 'equity' in fund_df.columns:
                    symbol_features['debt_to_equity'] = fund_df['debt'] / (fund_df['equity'] + 1e-8)
                
                if 'gross_profit' in fund_df.columns and 'assets' in fund_df.columns:
                    symbol_features['gross_profitability'] = fund_df['gross_profit'] / (fund_df['assets'] + 1e-8)
            
            # Growth metrics
            if self.feature_config.enable_growth_metrics:
                if 'revenue' in fund_df.columns:
                    symbol_features['revenue_growth'] = fund_df['revenue'].pct_change(4)  # YoY
                    
                if 'earnings' in fund_df.columns:
                    symbol_features['earnings_growth'] = fund_df['earnings'].pct_change(4)  # YoY
                    
                if 'book_value' in fund_df.columns:
                    symbol_features['book_growth'] = fund_df['book_value'].pct_change(4)  # YoY
            
            features_list.append(symbol_features)
        
        if features_list:
            combined_features = pd.concat(features_list, axis=0)
            return combined_features
        else:
            return pd.DataFrame()
    
    def create_cross_sectional_features(self, all_features: pd.DataFrame, 
                                       date: pd.Timestamp) -> pd.DataFrame:
        """Create cross-sectional ranking features"""
        
        # Get features for specific date
        date_features = all_features.loc[all_features.index.get_level_values('date') == date].copy()
        
        if date_features.empty:
            return pd.DataFrame()
        
        # Remove date from index for processing
        if 'date' in date_features.index.names:
            date_features = date_features.droplevel('date')
        
        cross_sectional_features = date_features.copy()
        
        # Overall universe rankings
        numeric_columns = date_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'symbol':
                continue
                
            # Percentile ranks (0-1)
            cross_sectional_features[f'{col}_rank'] = date_features[col].rank(pct=True)
            
            # Z-scores (cross-sectional standardization)
            mean_val = date_features[col].mean()
            std_val = date_features[col].std()
            if std_val > 0:
                cross_sectional_features[f'{col}_zscore'] = (date_features[col] - mean_val) / std_val
            else:
                cross_sectional_features[f'{col}_zscore'] = 0
        
        # Sector-neutral rankings
        if self.feature_config.use_sector_neutralization and 'symbol' in date_features.columns:
            
            # Add sector information
            cross_sectional_features['sector'] = cross_sectional_features['symbol'].map(self.psx_sectors)
            cross_sectional_features['sector'] = cross_sectional_features['sector'].fillna('Other')
            
            for col in numeric_columns:
                if col == 'symbol':
                    continue
                
                sector_ranks = []
                sector_zscores = []
                
                for symbol in cross_sectional_features.index:
                    if symbol not in cross_sectional_features['symbol'].values:
                        continue
                        
                    sector = cross_sectional_features.loc[
                        cross_sectional_features['symbol'] == symbol, 'sector'
                    ].iloc[0]
                    
                    # Get sector data
                    sector_mask = cross_sectional_features['sector'] == sector
                    sector_data = cross_sectional_features.loc[sector_mask, col]
                    
                    if len(sector_data) > 1:
                        # Sector percentile rank
                        sector_rank = sector_data.rank(pct=True).loc[
                            cross_sectional_features['symbol'] == symbol
                        ].iloc[0] if symbol in cross_sectional_features['symbol'].values else 0.5
                        
                        # Sector z-score
                        sector_mean = sector_data.mean()
                        sector_std = sector_data.std()
                        if sector_std > 0:
                            sector_zscore = (sector_data.loc[
                                cross_sectional_features['symbol'] == symbol
                            ].iloc[0] - sector_mean) / sector_std
                        else:
                            sector_zscore = 0
                    else:
                        sector_rank = 0.5
                        sector_zscore = 0
                    
                    sector_ranks.append(sector_rank)
                    sector_zscores.append(sector_zscore)
                
                cross_sectional_features[f'{col}_sector_rank'] = sector_ranks
                cross_sectional_features[f'{col}_sector_zscore'] = sector_zscores
        
        return cross_sectional_features
    
    def apply_winsorization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to handle outliers"""
        
        if not self.feature_config.use_winsorization:
            return features_df
        
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        winsorized_df = features_df.copy()
        
        for col in numeric_columns:
            lower_bound = features_df[col].quantile(self.feature_config.winsorize_percentile)
            upper_bound = features_df[col].quantile(1 - self.feature_config.winsorize_percentile)
            
            winsorized_df[col] = features_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return winsorized_df
    
    def select_features(self, features_df: pd.DataFrame, 
                       target_series: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using mutual information"""
        
        # Align features and target
        common_index = features_df.index.intersection(target_series.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target_series.loc[common_index]
        
        # Remove non-numeric columns
        numeric_features = features_aligned.select_dtypes(include=[np.number])
        
        # Remove features with too many NaNs
        valid_features = numeric_features.dropna(axis=1, thresh=len(numeric_features) * 0.5)
        
        # Fill remaining NaNs
        valid_features = valid_features.fillna(valid_features.median())
        
        if valid_features.empty or len(valid_features.columns) <= self.feature_config.max_features:
            return valid_features, list(valid_features.columns)
        
        # Feature selection
        if self.feature_config.feature_selection_method == "mutual_info":
            selector = SelectKBest(
                score_func=mutual_info_regression, 
                k=min(self.feature_config.max_features, len(valid_features.columns))
            )
            
            X_selected = selector.fit_transform(valid_features, target_aligned)
            selected_features = valid_features.columns[selector.get_support()].tolist()
            
            selected_df = pd.DataFrame(
                X_selected, 
                index=valid_features.index, 
                columns=selected_features
            )
            
            # Store feature importance
            feature_scores = dict(zip(selected_features, selector.scores_[selector.get_support()]))
            self.feature_importance_history[dt.datetime.now()] = feature_scores
            
            return selected_df, selected_features
        
        else:
            # Return top features by variance for now
            feature_variances = valid_features.var().sort_values(ascending=False)
            top_features = feature_variances.head(self.feature_config.max_features).index.tolist()
            
            return valid_features[top_features], top_features
    
    def create_labels(self, prices_df: pd.DataFrame, 
                     forward_days: int = 5) -> pd.Series:
        """Create forward-looking labels"""
        
        # Calculate forward returns
        forward_returns = prices_df['Close'].pct_change(forward_days).shift(-forward_days)
        
        # Volatility adjustment
        if self.config.model.label_type == "vol_adjusted":
            volatility = prices_df['Close'].pct_change().rolling(21).std()
            forward_returns = forward_returns / (volatility + 1e-8)
        
        elif self.config.model.label_type == "rank":
            # Convert to cross-sectional ranks
            forward_returns = forward_returns.groupby(forward_returns.index.get_level_values('date')).rank(pct=True)
        
        return forward_returns
    
    def process_symbol_data(self, symbol: str, 
                           start_date: dt.date, 
                           end_date: dt.date) -> Tuple[pd.DataFrame, pd.Series]:
        """Process complete feature pipeline for a single symbol"""
        
        try:
            # Fetch price data
            price_data = self.data_fetcher.fetch(symbol, start_date, end_date)
            
            if price_data.empty:
                logger.warning(f"No price data for {symbol}")
                return pd.DataFrame(), pd.Series()
            
            # Create price features
            price_features = self.create_price_features(price_data)
            
            # Create labels
            labels = self.create_labels(price_data, self.config.model.label_forward_days)
            
            # Add symbol identifier
            price_features['symbol'] = symbol
            
            return price_features, labels
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return pd.DataFrame(), pd.Series()
    
    def process_universe(self, symbols: List[str], 
                        start_date: dt.date, 
                        end_date: dt.date) -> Tuple[pd.DataFrame, pd.Series]:
        """Process entire universe of symbols"""
        
        all_features = []
        all_labels = []
        
        logger.info(f"Processing {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                logger.info(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
            
            features, labels = self.process_symbol_data(symbol, start_date, end_date)
            
            if not features.empty and not labels.empty:
                # Add multi-index for symbol and date
                features.index = pd.MultiIndex.from_product(
                    [[symbol], features.index], 
                    names=['symbol', 'date']
                )
                
                labels.index = pd.MultiIndex.from_product(
                    [[symbol], labels.index], 
                    names=['symbol', 'date']
                )
                
                all_features.append(features)
                all_labels.append(labels)
        
        if not all_features:
            logger.error("No features created for any symbol")
            return pd.DataFrame(), pd.Series()
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=0)
        combined_labels = pd.concat(all_labels, axis=0)
        
        # Remove rows with NaN labels
        valid_mask = ~combined_labels.isna()
        combined_features = combined_features[valid_mask]
        combined_labels = combined_labels[valid_mask]
        
        # Apply winsorization
        combined_features = self.apply_winsorization(combined_features)
        
        logger.info(f"Created {len(combined_features.columns)} features for {len(combined_features)} observations")
        
        return combined_features, combined_labels
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of features"""
        
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        summary = {
            'total_features': len(features_df.columns),
            'numeric_features': len(numeric_features.columns),
            'total_observations': len(features_df),
            'missing_data_pct': (features_df.isnull().sum().sum() / features_df.size) * 100,
            'feature_stats': {}
        }
        
        for col in numeric_features.columns[:10]:  # Top 10 for summary
            summary['feature_stats'][col] = {
                'mean': numeric_features[col].mean(),
                'std': numeric_features[col].std(),
                'min': numeric_features[col].min(),
                'max': numeric_features[col].max(),
                'missing_pct': (numeric_features[col].isnull().sum() / len(numeric_features)) * 100
            }
        
        return summary

# Test function
def test_feature_engineering():
    """Test feature engineering pipeline"""
    print("🚀 Testing Feature Engineering Pipeline")
    print("=" * 50)
    
    # Create feature engineer
    config = SystemConfig()
    feature_engineer = FeatureEngineer(config)
    
    print(f"📊 Configuration:")
    print(f"   Momentum Periods: {config.features.momentum_periods}")
    print(f"   Reversal Periods: {config.features.reversal_periods}")
    print(f"   Max Features: {config.features.max_features}")
    print(f"   Use Sector Neutralization: {config.features.use_sector_neutralization}")
    
    # Test with sample symbols
    test_symbols = ['UBL', 'MCB', 'FFC']
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365)  # 1 year
    
    print(f"\n🔍 Processing {len(test_symbols)} symbols...")
    print(f"   Date Range: {start_date} to {end_date}")
    
    try:
        # Process universe
        features, labels = feature_engineer.process_universe(
            test_symbols, start_date, end_date
        )
        
        if not features.empty:
            print(f"\n✅ Feature Engineering Results:")
            print(f"   Features Shape: {features.shape}")
            print(f"   Labels Shape: {labels.shape}")
            
            # Get feature summary
            summary = feature_engineer.get_feature_summary(features)
            print(f"\n📈 Feature Summary:")
            print(f"   Total Features: {summary['total_features']}")
            print(f"   Numeric Features: {summary['numeric_features']}")
            print(f"   Observations: {summary['total_observations']}")
            print(f"   Missing Data: {summary['missing_data_pct']:.2f}%")
            
            # Show sample features
            print(f"\n🔍 Sample Features:")
            sample_features = features.head()
            for col in list(sample_features.columns)[:5]:
                if sample_features[col].dtype in ['float64', 'int64']:
                    print(f"   {col}: {sample_features[col].iloc[0]:.4f}")
            
            # Test feature selection
            print(f"\n🎯 Testing Feature Selection...")
            selected_features, selected_names = feature_engineer.select_features(features, labels)
            print(f"   Selected Features: {len(selected_names)}")
            print(f"   Top 5: {selected_names[:5]}")
            
        else:
            print("❌ No features created")
            
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
    
    print(f"\n🏁 Feature engineering test completed!")

if __name__ == "__main__":
    test_feature_engineering()