# 🏦 Advanced Institutional Trading System Setup

## Overview

This guide will help you install and configure the advanced institutional-grade trading system that includes:

- 🧠 **LSTM Primary Model** for 60-second directional prediction
- ⚖️ **LightGBM Meta-Model** for trade approval and sizing
- 📰 **NLP Sentiment Analysis** with FinBERT
- 📊 **Real-time Order Flow Analysis** 
- 🔄 **Live Data Feeds** (Crypto, Stocks, News)

## 🛠️ Installation

### 1. Basic Requirements

```bash
# Core ML libraries
pip install tensorflow>=2.13.0
pip install lightgbm>=4.0.0
pip install scikit-learn>=1.3.0

# Deep Learning
pip install keras>=2.13.1

# NLP for Sentiment Analysis
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install nltk>=3.8
pip install textblob>=0.17.1

# Market Data
pip install ccxt>=4.0.0
pip install yfinance>=0.2.0
pip install pandas-datareader>=0.10.0

# Additional utilities
pip install aiohttp>=3.8.0
pip install asyncio-mqtt>=0.13.0
pip install python-dotenv>=1.0.0
```

### 2. Optional Advanced Features

```bash
# For advanced NLP models
pip install sentence-transformers>=2.2.0
pip install spacy>=3.6.0

# For additional market data sources
pip install alpha-vantage>=2.3.1
pip install polygon-api-client>=1.12.0
pip install newsapi-python>=0.2.6

# For enhanced visualization
pip install plotly>=5.15.0
pip install seaborn>=0.12.0
```

## 🔑 API Configuration

### 1. Create Environment File

Create a `.env` file in your project root:

```bash
# Market Data APIs
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
POLYGON_API_KEY=your_polygon_key

# Crypto Exchange APIs (Optional)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET=your_coinbase_secret

# NLP Model Configuration
HUGGINGFACE_TOKEN=your_huggingface_token

# System Configuration
MAX_CONCURRENT_REQUESTS=10
DATA_REFRESH_INTERVAL=100  # milliseconds
```

### 2. Get API Keys

#### News Data (Required)
- **NewsAPI**: [https://newsapi.org/](https://newsapi.org/) - Free tier: 1000 requests/day
- **Alpha Vantage**: [https://www.alphavantage.co/](https://www.alphavantage.co/) - Free tier: 5 calls/minute

#### Crypto Data (Optional)
- **Binance**: [https://www.binance.com/en/binance-api](https://www.binance.com/en/binance-api)
- **Coinbase Pro**: [https://docs.pro.coinbase.com/](https://docs.pro.coinbase.com/)

#### Advanced Features (Optional)
- **Polygon**: [https://polygon.io/](https://polygon.io/) - Real-time market data
- **Hugging Face**: [https://huggingface.co/](https://huggingface.co/) - Advanced NLP models

## 🧠 Model Setup

### 1. Download NLP Models

```python
# Run this once to download required models
import nltk
from transformers import pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize FinBERT model (this will download automatically)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
print("✅ NLP models ready!")
```

### 2. Initialize System

```python
from src.advanced_trading_system import create_advanced_trading_system
import asyncio

# Create the advanced system
system = create_advanced_trading_system()

# Check system status
status = system.get_system_status()
print(f"System Status: {status}")

# Generate a test signal
signal = asyncio.run(system.generate_advanced_signal('BTC/USD'))
print(f"Test Signal: {signal.primary_signal} with {signal.final_probability:.1%} probability")
```

## 🚀 Usage Examples

### 1. Basic Signal Generation

```python
import asyncio
from src.advanced_trading_system import create_advanced_trading_system

async def main():
    # Initialize system
    system = create_advanced_trading_system()
    
    # Generate signal for Bitcoin
    signal = await system.generate_advanced_signal('BTC/USD')
    
    if signal.meta_approval and signal.final_probability > 0.65:
        print(f"🚀 TRADE SIGNAL: {signal.primary_signal}")
        print(f"   Confidence: {signal.final_probability:.1%}")
        print(f"   Position Size: {signal.position_size:.2%}")
        print(f"   Entry: ${signal.entry_price:.2f}")
        print(f"   Stop Loss: ${signal.stop_loss:.2f}")
        print(f"   Take Profit: ${signal.take_profit:.2f}")
    else:
        print("❌ No trade signal - criteria not met")

# Run the example
asyncio.run(main())
```

### 2. Real-Time Data Feed

```python
import asyncio
from src.advanced_trading_system import create_advanced_trading_system

async def real_time_trading():
    system = create_advanced_trading_system()
    
    # Start real-time data feed
    await system.start_real_time_data_feed('BTC/USD', 'crypto')
    
    # Monitor for 1 hour
    for _ in range(3600):  # 1 hour in seconds
        signal = await system.generate_advanced_signal('BTC/USD')
        
        if signal.meta_approval:
            print(f"⚡ Real-time Signal: {signal.primary_signal} - {signal.final_probability:.1%}")
        
        await asyncio.sleep(1)  # Check every second
    
    # Stop data feed
    system.stop_data_feed()

# Run real-time trading
asyncio.run(real_time_trading())
```

### 3. Model Training

```python
import pandas as pd
from src.advanced_trading_system import create_advanced_trading_system

# Prepare training data (example)
training_data = pd.read_csv('historical_market_data.csv')
labels = pd.read_csv('historical_labels.csv')

# Initialize system
system = create_advanced_trading_system()

# Train models
success = await system.train_models_with_data(training_data, labels['target'])

if success:
    print("✅ Models trained successfully!")
else:
    print("❌ Model training failed")
```

## 📊 Streamlit Interface

### Access the Advanced System

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Navigate to **🏦 Institutional System** in the sidebar

3. Features available:
   - Real-time signal generation
   - Model status monitoring
   - Configuration management
   - Performance analytics

## 🔧 Configuration Options

### System Parameters

```python
config = {
    'primary_model_confidence_threshold': 0.60,  # LSTM confidence threshold
    'meta_model_threshold': 0.65,                # Meta-model approval threshold
    'max_position_size': 0.025,                  # Maximum 2.5% position size
    'base_position_size': 0.01,                  # Base 1% position size
    'lookback_window': 60,                       # 60-second LSTM window
    'news_sentiment_weight': 0.3,                # News sentiment weighting
    'order_flow_weight': 0.7                     # Order flow weighting
}
```

### Risk Management

```python
risk_params = {
    'max_drawdown': 0.10,          # 10% max drawdown
    'position_correlation': 0.7,    # Max correlation between positions
    'sector_exposure': 0.25,        # Max 25% per sector
    'single_stock_limit': 0.05,     # Max 5% per single stock
    'volatility_adjustment': True,  # Enable volatility-based sizing
    'dynamic_stops': True           # Enable ATR-based stops
}
```

## 🚨 Important Notes

### Performance Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Multi-core processor recommended
- **Storage**: 5GB for models and data
- **Network**: Stable internet for real-time feeds

### Data Requirements
- Historical price data (1+ years recommended)
- Real-time market data feeds
- News data access
- Order book data (for advanced features)

### Legal Disclaimers
- This is an educational/research system
- Not financial advice
- Test thoroughly before live trading
- Comply with local regulations
- Use appropriate risk management

## 🐛 Troubleshooting

### Common Issues

#### TensorFlow Installation
```bash
# If TensorFlow fails to install:
pip install --upgrade pip
pip install tensorflow-cpu  # For CPU-only version
# or
pip install tensorflow-gpu  # For GPU version (requires CUDA)
```

#### Memory Issues
```python
# Reduce batch size and model complexity
config = {
    'lstm_batch_size': 16,     # Reduce from default 32
    'max_sequence_length': 30, # Reduce from default 60
    'lstm_units': 64           # Reduce from default 128
}
```

#### API Rate Limits
```python
# Add delays between API calls
import time
time.sleep(0.1)  # 100ms delay between requests
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review system logs for error messages
3. Ensure all dependencies are correctly installed
4. Verify API keys and network connectivity

---

🚀 **Ready to trade with institutional-grade algorithms!**