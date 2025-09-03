# PSX Quantitative Trading System 📈

A professional-grade quantitative trading system for the Pakistan Stock Exchange (PSX) featuring real-time data integration, machine learning models, and comprehensive backtesting capabilities.

## 🚀 Live Demo

**[Try the Live Dashboard →](https://your-streamlit-app-url.streamlit.app/)**

## ✨ Key Features

### 📊 Real-Time Data Integration
- **PSX DPS Official API**: Direct integration with PSX's official data source
- **Tick-by-tick Processing**: Handle high-frequency intraday data  
- **Multiple Data Sources**: EODHD Premium API as backup
- **Live Market Monitoring**: Real-time price and volume tracking

### 🤖 Advanced Machine Learning
- **LightGBM Ensemble Models**: Professional gradient boosting
- **Purged Time Series Cross-Validation**: Prevents data leakage
- **Walk-Forward Analysis**: Robust out-of-sample testing
- **Feature Engineering Pipeline**: 50+ technical and fundamental features

### 💼 Professional Portfolio Management
- **Kelly Criterion Position Sizing**: Optimal capital allocation
- **Risk Parity Optimization**: Balanced risk exposure
- **Dynamic Rebalancing**: Automated portfolio adjustments
- **Sector Exposure Limits**: Diversification controls

### 🛡️ Comprehensive Risk Management
- **Real-time Stop Loss/Take Profit**: Automated risk controls
- **Drawdown Circuit Breakers**: Portfolio protection
- **Position Limits**: Risk-based sizing
- **VaR & Stress Testing**: Advanced risk metrics

### 📈 Professional Backtesting
- **Walk-Forward Validation**: Institutional-grade testing
- **Intraday & Daily Strategies**: Multiple timeframes
- **Performance Attribution**: Detailed analysis
- **Transaction Cost Modeling**: Realistic execution simulation

## 🎯 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Annual Alpha** | 6%+ | Excess return vs KSE100 |
| **Sharpe Ratio** | 1.5+ | Risk-adjusted returns |
| **Max Drawdown** | <20% | Maximum portfolio decline |
| **Win Rate** | 55%+ | Profitable trades ratio |

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/psx-quantitative-trading.git
cd psx-quantitative-trading
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Dashboard
```bash
streamlit run streamlit_professional_dashboard.py
```

## 📱 Dashboard Features

- **🏠 System Overview**: Real-time system status and performance targets
- **📊 Live Market Data**: Real-time PSX prices with auto-refresh
- **🔬 Strategy Backtesting**: Daily and intraday strategy testing
- **🎯 Performance Analytics**: Comprehensive metrics and analysis
- **🔧 System Status**: Component health monitoring
- **📚 Documentation**: Complete system documentation

## 📊 Data Sources

### Primary: PSX DPS Official API
- **Endpoint**: `https://dps.psx.com.pk/timeseries/int/{SYMBOL}`
- **Format**: `[timestamp, price, volume]` arrays
- **Coverage**: All PSX listed securities
- **Update**: Real-time during market hours

### Backup: EODHD Premium API
- **Coverage**: Historical and fundamental data
- **Reliability**: 99.9% uptime guarantee

## 🔄 Usage Examples

### Live Data Fetching
```python
from psx_dps_fetcher import PSXDPSFetcher

fetcher = PSXDPSFetcher()
data = fetcher.fetch_intraday_ticks('HBL')
print(f"✅ Fetched {len(data)} ticks for HBL")
```

### Backtesting
```python
from intraday_backtesting_engine import IntradayWalkForwardBacktester
from quant_system_config import SystemConfig

config = SystemConfig()
backtester = IntradayWalkForwardBacktester(config)
results = backtester.run_intraday_backtest(['HBL', 'UBL'], start_date, end_date)
```

## 🛡️ Risk Management

- **Position Sizing**: Kelly criterion with risk overlay
- **Stop Loss**: Dynamic based on volatility (2-5%)
- **Take Profit**: 1:2 risk-reward minimum
- **Daily Loss Limit**: 3% of portfolio value
- **Max Drawdown**: 20% circuit breaker

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Local Development
```bash
streamlit run streamlit_professional_dashboard.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PSX**: Official DPS API
- **EODHD**: Market data provider
- **LightGBM**: ML framework
- **Streamlit**: Dashboard framework

---

**⭐ Star this repository if you find it useful!**