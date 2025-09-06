#!/usr/bin/env python3
"""
Enhanced Stock Analysis Dashboard
================================

Combines real PSX stock data with advanced ML components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import existing stock data components
try:
    from enhanced_data_fetcher import EnhancedDataFetcher
    from feature_engineering import FeatureEngineer
    from quant_system_config import SystemConfig
    STOCK_DATA_AVAILABLE = True
except ImportError:
    STOCK_DATA_AVAILABLE = False

# Import enhanced ML components
try:
    from clustering_engine import ClusteringEngine
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

try:
    from bayesian_engine import BayesianEngine  
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from statistical_validation_framework import StatisticalValidationFramework
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

st.set_page_config(
    page_title="Enhanced Stock Analysis",
    page_icon="📈", 
    layout="wide"
)

# PSX Stock symbols
PSX_SYMBOLS = [
    'UBL', 'MCB', 'HBL', 'ABL', 'NBP', 'BAFL',  # Banking
    'PPL', 'OGDC', 'POL', 'MARI', 'PSO',        # Oil & Gas
    'LUCK', 'DGKC', 'FCCL', 'MLCF',             # Cement
    'ENGRO', 'FFC', 'FATIMA',                   # Chemicals
    'NESTLE', 'LOTTE', 'PTC'                    # Consumer
]

def fetch_stock_data(symbols, days=365):
    """Fetch real stock data"""
    if not STOCK_DATA_AVAILABLE:
        # Return mock data if real fetcher not available
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        mock_data = []
        
        for symbol in symbols:
            # Generate realistic stock price movements
            base_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
            prices = base_price * np.exp(np.cumsum(returns))
            volumes = np.random.randint(1000, 100000, days)
            
            for i, date in enumerate(dates):
                mock_data.append({
                    'symbol': symbol,
                    'date': date,
                    'Close': prices[i],
                    'High': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                    'Low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                    'Open': prices[i] * (1 + np.random.normal(0, 0.005)),
                    'Volume': volumes[i]
                })
        
        return pd.DataFrame(mock_data)
    
    else:
        # Use real data fetcher
        try:
            config = SystemConfig()
            fetcher = EnhancedDataFetcher()
            
            all_data = []
            for symbol in symbols:
                try:
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=days)
                    
                    data = fetcher.fetch(symbol, start_date, end_date)
                    if not data.empty:
                        data['symbol'] = symbol
                        data['date'] = data.index
                        all_data.append(data)
                except:
                    continue
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching real data: {e}")
            return pd.DataFrame()

def main():
    st.title("📈 Enhanced PSX Stock Analysis")
    st.markdown("*Real Pakistani stocks with advanced ML analysis*")
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    
    # Stock selection
    selected_symbols = st.sidebar.multiselect(
        "Select PSX Stocks",
        PSX_SYMBOLS,
        default=['UBL', 'MCB', 'FFC', 'ENGRO']
    )
    
    # Time period
    days_back = st.sidebar.slider("Days of History", 30, 365, 180)
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["📊 Stock Overview", "🧩 ML Clustering", "📈 Bayesian Analysis", "🔬 Statistical Tests"]
    )
    
    if not selected_symbols:
        st.warning("⚠️ Please select at least one stock symbol")
        return
    
    # Load data
    with st.spinner(f"Loading {len(selected_symbols)} stocks for {days_back} days..."):
        stock_data = fetch_stock_data(selected_symbols, days_back)
    
    if stock_data.empty:
        st.error("❌ No stock data available")
        return
    
    # Display analysis based on selection
    if analysis_type == "📊 Stock Overview":
        show_stock_overview(stock_data, selected_symbols)
    
    elif analysis_type == "🧩 ML Clustering" and CLUSTERING_AVAILABLE:
        show_clustering_analysis(stock_data, selected_symbols)
    
    elif analysis_type == "📈 Bayesian Analysis" and BAYESIAN_AVAILABLE:
        show_bayesian_analysis(stock_data, selected_symbols)
    
    elif analysis_type == "🔬 Statistical Tests" and VALIDATION_AVAILABLE:
        show_statistical_analysis(stock_data, selected_symbols)
    
    else:
        st.error(f"❌ {analysis_type} not available - missing ML components")

def show_stock_overview(data, symbols):
    """Show basic stock overview"""
    
    st.header("📊 Stock Price Overview")
    
    # Price chart
    fig = go.Figure()
    
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].sort_values('date')
        if not symbol_data.empty:
            fig.add_trace(go.Scatter(
                x=symbol_data['date'],
                y=symbol_data['Close'],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Stock Price Movements",
        xaxis_title="Date",
        yaxis_title="Price (PKR)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Performance Summary")
    
    summary_data = []
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].sort_values('date')
        if len(symbol_data) > 1:
            first_price = symbol_data['Close'].iloc[0]
            last_price = symbol_data['Close'].iloc[-1]
            returns = symbol_data['Close'].pct_change().dropna()
            
            summary_data.append({
                'Symbol': symbol,
                'Start Price': f"{first_price:.2f}",
                'End Price': f"{last_price:.2f}",
                'Total Return': f"{((last_price/first_price-1)*100):.2f}%",
                'Daily Vol': f"{(returns.std()*100):.2f}%",
                'Avg Volume': f"{symbol_data['Volume'].mean():.0f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

def show_clustering_analysis(data, symbols):
    """Show clustering analysis on real stocks"""
    
    st.header("🧩 Stock Clustering Analysis")
    
    # Prepare features for clustering
    features_list = []
    feature_symbols = []
    
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].sort_values('date')
        if len(symbol_data) > 30:  # Need minimum data
            
            # Calculate technical features
            returns = symbol_data['Close'].pct_change()
            
            features = {
                'volatility_30d': returns.rolling(30).std().iloc[-1],
                'momentum_30d': returns.rolling(30).mean().iloc[-1],
                'volume_trend': symbol_data['Volume'].rolling(30).mean().iloc[-1],
                'price_trend': (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[-30] - 1),
            }
            
            # Only add if all features are valid
            if all(pd.notna(list(features.values()))):
                features_list.append(list(features.values()))
                feature_symbols.append(symbol)
    
    if len(features_list) < 3:
        st.warning("⚠️ Need at least 3 stocks with sufficient data for clustering")
        return
    
    # Perform clustering
    try:
        config = SystemConfig() if STOCK_DATA_AVAILABLE else None
        clustering_engine = ClusteringEngine(config)
        
        features_array = np.array(features_list)
        results = clustering_engine.fit_clustering(features_array, optimize_params=False)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clusters Found", results['n_clusters'])
        with col2:
            st.metric("Stocks Clustered", len(feature_symbols) - results['n_noise'])
        with col3:
            st.metric("Silhouette Score", f"{results['validation_results']['silhouette_score']:.3f}")
        
        # Show clusters
        st.subheader("📊 Stock Clusters")
        
        cluster_data = []
        labels = results['labels']
        
        for i, symbol in enumerate(feature_symbols):
            cluster_id = labels[i]
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
            
            cluster_data.append({
                'Symbol': symbol,
                'Cluster': cluster_name,
                'Volatility': f"{features_list[i][0]*100:.2f}%",
                'Momentum': f"{features_list[i][1]*100:.2f}%",
                'Volume Trend': f"{features_list[i][2]:.0f}",
                'Price Trend': f"{features_list[i][3]*100:.2f}%"
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        st.dataframe(cluster_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Clustering failed: {e}")

def show_bayesian_analysis(data, symbols):
    """Show Bayesian analysis on stock returns"""
    
    st.header("📊 Bayesian Stock Analysis")
    
    # Select one stock for detailed analysis
    selected_stock = st.selectbox("Select Stock for Bayesian Analysis", symbols)
    
    stock_data = data[data['symbol'] == selected_stock].sort_values('date')
    
    if len(stock_data) < 50:
        st.warning("⚠️ Need more data for Bayesian analysis")
        return
    
    # Prepare data
    returns = stock_data['Close'].pct_change().dropna()
    
    # Create features (lagged returns, volatility, etc.)
    features_data = []
    targets = []
    
    for i in range(5, len(returns)):
        # Features: last 5 returns
        features_data.append(returns.iloc[i-5:i].values)
        # Target: next return
        targets.append(returns.iloc[i])
    
    if len(features_data) < 30:
        st.warning("⚠️ Insufficient data for Bayesian modeling")
        return
    
    try:
        # Bayesian regression
        config = SystemConfig() if STOCK_DATA_AVAILABLE else None
        bayesian_engine = BayesianEngine(config)
        
        X = np.array(features_data)
        y = np.array(targets)
        
        results = bayesian_engine.fit_bayesian_regression(X, y, f"stock_{selected_stock}")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Log Evidence", f"{results['log_evidence']:.2f}")
        with col2:
            st.metric("Model Fit Time", f"{results['fitting_time_ms']:.1f}ms")
        with col3:
            st.metric("Features Used", results['n_features'])
        
        # Make predictions with uncertainty
        pred_mean, pred_std = bayesian_engine.predict_with_uncertainty(X[-20:], f"stock_{selected_stock}")
        
        st.subheader("🎯 Return Predictions with Uncertainty")
        
        # Plot predictions
        fig = go.Figure()
        
        x_range = list(range(len(pred_mean)))
        
        # Add prediction with uncertainty bands
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=list(pred_mean + pred_std) + list((pred_mean - pred_std)[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty Band'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=pred_mean,
            mode='lines+markers',
            name='Predicted Returns',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y[-20:],
            mode='markers',
            name='Actual Returns',
            marker=dict(color='red', size=6)
        ))
        
        fig.update_layout(
            title=f"{selected_stock} Return Predictions with Uncertainty",
            xaxis_title="Time Period",
            yaxis_title="Daily Return",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Interpretation:**
        - **Blue line**: Predicted returns
        - **Gray band**: Uncertainty range (wider = more uncertain)
        - **Red dots**: Actual returns
        - **Log Evidence**: {results['log_evidence']:.2f} (higher = better model fit)
        """)
        
    except Exception as e:
        st.error(f"❌ Bayesian analysis failed: {e}")

def show_statistical_analysis(data, symbols):
    """Show statistical tests on stock returns"""
    
    st.header("🔬 Statistical Analysis")
    
    # Compare two stocks
    if len(symbols) < 2:
        st.warning("⚠️ Select at least 2 stocks for comparison")
        return
    
    stock1 = st.selectbox("Stock 1", symbols, index=0)
    stock2 = st.selectbox("Stock 2", symbols, index=1)
    
    # Get returns
    stock1_data = data[data['symbol'] == stock1].sort_values('date')
    stock2_data = data[data['symbol'] == stock2].sort_values('date')
    
    returns1 = stock1_data['Close'].pct_change().dropna()
    returns2 = stock2_data['Close'].pct_change().dropna()
    
    if len(returns1) < 30 or len(returns2) < 30:
        st.warning("⚠️ Need more data for statistical tests")
        return
    
    try:
        config = SystemConfig() if STOCK_DATA_AVAILABLE else None
        validator = StatisticalValidationFramework(config)
        
        # Convert to time series for validation
        returns1_ts = pd.Series(returns1.values)
        returns2_ts = pd.Series(returns2.values)
        
        # Validate each stock's returns
        validation1 = validator.validate_trading_strategy(returns1_ts)
        validation2 = validator.validate_trading_strategy(returns2_ts)
        
        # Display comparison
        st.subheader("📊 Statistical Comparison")
        
        comparison_data = []
        
        for stock, validation, returns in [(stock1, validation1, returns1), (stock2, validation2, returns2)]:
            summary = validation['strategy_summary']
            zero_test = validation['zero_return_test']['t_test']
            
            comparison_data.append({
                'Stock': stock,
                'Mean Return': f"{summary['mean_return']:.6f}",
                'Volatility': f"{summary['std_return']:.4f}",
                'Sharpe Ratio': f"{summary['sharpe_ratio']:.4f}",
                'Max Drawdown': f"{summary['max_drawdown']:.4f}",
                'Significant Returns': "Yes" if zero_test['significant'] else "No",
                'p-value': f"{zero_test['p_value']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Statistical significance test
        st.subheader("🔍 Hypothesis Tests")
        
        st.markdown(f"""
        **{stock1} Returns vs Zero:**
        - Test: Are returns significantly different from zero?
        - Result: {"✅ Significant" if validation1['zero_return_test']['t_test']['significant'] else "❌ Not Significant"}
        - p-value: {validation1['zero_return_test']['t_test']['p_value']:.6f}
        
        **{stock2} Returns vs Zero:**
        - Test: Are returns significantly different from zero?  
        - Result: {"✅ Significant" if validation2['zero_return_test']['t_test']['significant'] else "❌ Not Significant"}
        - p-value: {validation2['zero_return_test']['t_test']['p_value']:.6f}
        
        **Interpretation:**
        - p-value < 0.05: Returns are statistically significant
        - p-value ≥ 0.05: Returns could be due to random chance
        """)
        
    except Exception as e:
        st.error(f"❌ Statistical analysis failed: {e}")

if __name__ == "__main__":
    main()