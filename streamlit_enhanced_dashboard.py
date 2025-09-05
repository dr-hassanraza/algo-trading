#!/usr/bin/env python3
"""
Enhanced Streamlit Dashboard for Algorithmic Trading System
==========================================================

Professional dashboard integrating all enhanced architecture components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Try importing enhanced components
try:
    from quant_system_config import SystemConfig
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

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
    from feature_correlation_manager import FeatureCorrelationManager
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False

try:
    from statistical_validation_framework import StatisticalValidationFramework
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Enhanced Algo Trading",
    page_icon="🚀",
    layout="wide"
)

def create_sample_data(n_samples=200, n_features=10):
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some correlation
    X_df['corr_feature'] = X_df['feature_0'] + np.random.randn(n_samples) * 0.1
    
    # Create target
    y = X_df['feature_0'] * 2 + X_df['feature_1'] * -1 + np.random.randn(n_samples) * 0.5
    
    return X_df, pd.Series(y, name='target')

def main():
    st.title("🚀 Enhanced Algorithmic Trading Dashboard")
    st.markdown("*Next-generation trading system with advanced ML capabilities*")
    
    # Component status
    st.header("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Available" if CORE_AVAILABLE else "❌ Not Available"
        st.metric("Core Components", status)
        
    with col2:
        status = "✅ Available" if CLUSTERING_AVAILABLE else "❌ Not Available"  
        st.metric("HDBSCAN Clustering", status)
        
    with col3:
        status = "✅ Available" if BAYESIAN_AVAILABLE else "❌ Not Available"
        st.metric("Bayesian Engine", status)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        status = "✅ Available" if FEATURE_AVAILABLE else "❌ Not Available"
        st.metric("Feature Manager", status)
        
    with col5:
        status = "✅ Available" if VALIDATION_AVAILABLE else "❌ Not Available"
        st.metric("Statistical Validation", status)
        
    with col6:
        enhancement_level = sum([CLUSTERING_AVAILABLE, BAYESIAN_AVAILABLE, 
                               FEATURE_AVAILABLE, VALIDATION_AVAILABLE])
        st.metric("Enhancement Level", f"{enhancement_level}/4")
    
    # Demo section
    st.header("🧪 Component Demo")
    
    if st.button("Run Demo Analysis"):
        with st.spinner("Running enhanced analysis..."):
            # Generate sample data
            X, y = create_sample_data(300, 15)
            
            # Basic analysis
            st.subheader("📊 Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Samples**: {len(X)}")
                st.write(f"**Features**: {len(X.columns)}")
                st.write(f"**Target Mean**: {y.mean():.4f}")
                st.write(f"**Target Std**: {y.std():.4f}")
            
            with col2:
                # Correlation heatmap
                corr_matrix = X.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced component tests
            if CLUSTERING_AVAILABLE:
                st.subheader("🧩 HDBSCAN Clustering Test")
                try:
                    config = SystemConfig() if CORE_AVAILABLE else None
                    clustering_engine = ClusteringEngine(config)
                    
                    # Simple clustering test
                    sample_features = X.select_dtypes(include=[np.number]).iloc[:100, :6].values
                    results = clustering_engine.fit_clustering(sample_features, optimize_params=False)
                    
                    st.success(f"✅ Clustering completed: {results['n_clusters']} clusters found")
                    st.write(f"Silhouette Score: {results['validation_results']['silhouette_score']:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Clustering test failed: {e}")
            
            if BAYESIAN_AVAILABLE:
                st.subheader("📊 Bayesian Analysis Test")
                try:
                    config = SystemConfig() if CORE_AVAILABLE else None
                    bayesian_engine = BayesianEngine(config)
                    
                    # Simple regression test
                    X_sample = X.select_dtypes(include=[np.number]).iloc[:150, :5].values
                    y_sample = y.iloc[:150].values
                    
                    results = bayesian_engine.fit_bayesian_regression(X_sample, y_sample)
                    
                    st.success(f"✅ Bayesian regression completed")
                    st.write(f"Log Evidence: {results['log_evidence']:.2f}")
                    st.write(f"Fitting Time: {results['fitting_time_ms']:.1f}ms")
                    
                    # Predictions with uncertainty
                    pred_mean, pred_std = bayesian_engine.predict_with_uncertainty(X_sample[:20])
                    st.write(f"Average Prediction Uncertainty: {np.mean(pred_std):.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Bayesian test failed: {e}")
            
            if FEATURE_AVAILABLE:
                st.subheader("🎯 Feature Selection Test")
                try:
                    config = SystemConfig() if CORE_AVAILABLE else None
                    feature_manager = FeatureCorrelationManager(config)
                    
                    # Feature analysis
                    analysis = feature_manager.analyze_feature_set(X, y)
                    
                    st.success(f"✅ Feature analysis completed")
                    st.write(f"Original Features: {analysis['n_features']}")
                    st.write(f"Selected Features: {analysis['quality_metrics']['selected_feature_count']}")
                    st.write(f"Reduction Ratio: {analysis['quality_metrics']['reduction_ratio']:.2f}")
                    st.write(f"Max Correlation: {analysis['correlation']['max_correlation']:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Feature analysis test failed: {e}")
            
            if VALIDATION_AVAILABLE:
                st.subheader("📈 Statistical Validation Test")
                try:
                    config = SystemConfig() if CORE_AVAILABLE else None
                    validator = StatisticalValidationFramework(config)
                    
                    # Generate sample returns
                    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
                    validation = validator.validate_trading_strategy(returns)
                    
                    st.success(f"✅ Statistical validation completed")
                    
                    summary = validation['strategy_summary']
                    st.write(f"Mean Return: {summary['mean_return']:.6f}")
                    st.write(f"Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
                    st.write(f"Max Drawdown: {summary['max_drawdown']:.4f}")
                    
                    if 'zero_return_test' in validation:
                        zero_test = validation['zero_return_test']['t_test']
                        significance = "Significant" if zero_test['significant'] else "Not Significant"
                        st.write(f"Returns vs Zero: {significance} (p={zero_test['p_value']:.4f})")
                    
                except Exception as e:
                    st.error(f"❌ Validation test failed: {e}")
            
            st.balloons()
    
    # Architecture info
    st.header("🏗️ Enhanced Architecture")
    
    st.markdown("""
    ### Key Components:
    
    1. **🧩 HDBSCAN Clustering Engine**
       - Market regime detection
       - Asset similarity clustering
       - Parameter optimization
       - Real-time cluster updates
    
    2. **📊 Bayesian Statistics Engine**
       - Bayesian linear regression
       - Uncertainty quantification
       - Hierarchical models
       - Online learning
    
    3. **🎯 Feature Correlation Manager**
       - Multi-method feature selection
       - Correlation analysis
       - Feature interaction detection
       - Real-time drift monitoring
    
    4. **📈 Statistical Validation Framework**
       - Hypothesis testing
       - Bootstrap confidence intervals
       - Cross-validation
       - Multiple testing correction
    
    5. **⚡ Real-time Processing Pipeline**
       - Sub-millisecond latency
       - Streaming feature computation
       - Performance monitoring
       - Circuit breaker patterns
    
    6. **🌐 API Integration Layer**
       - RESTful APIs
       - WebSocket support
       - Database integration
       - Caching layer
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **System Info:**
    - Python: {sys.version.split()[0]}
    - Streamlit: {st.__version__}
    - Enhanced Components: {sum([CLUSTERING_AVAILABLE, BAYESIAN_AVAILABLE, FEATURE_AVAILABLE, VALIDATION_AVAILABLE])}/4 Available
    
    🚀 *Institutional-grade algorithmic trading platform*
    """)

if __name__ == "__main__":
    main()