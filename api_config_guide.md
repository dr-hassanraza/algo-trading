# 📡 Real-Time API Pricing Configuration Guide

## Overview
This guide helps you configure real-time pricing APIs for your intraday trading system. The system supports multiple data sources with automatic fallback.

## 🔧 Current API Setup

### 1. PSX DPS API (Free - Primary)
**Endpoint:** `https://dps.psx.com.pk/timeseries/int/{symbol}`
- ✅ **Status:** Already configured
- ✅ **Rate Limit:** Reasonable for intraday use
- ✅ **Coverage:** All PSX symbols
- ⚠️ **Limitation:** May have delays during high volume

### 2. PSX Data Reader (Free - Secondary)
**Package:** `psx-data-reader`
- ✅ **Status:** Configured as fallback
- ✅ **Installation:** `pip install psx-data-reader`
- ✅ **Coverage:** Historical and current data
- ⚠️ **Limitation:** Limited real-time updates

## 🚀 Enhanced API Configuration

### Step 1: Environment Variables
Create a `.env` file in your project root:

```bash
# PSX API Configuration
PSX_API_ENABLED=true
PSX_API_TIMEOUT=10
PSX_API_RETRY_COUNT=3

# Alternative API Keys (if you get premium access)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
IEX_CLOUD_API_KEY=your_key_here

# Rate Limiting
API_RATE_LIMIT_PER_MINUTE=60
API_CACHE_DURATION_SECONDS=30
```

### Step 2: Premium API Options

#### A. Alpha Vantage (Global Markets)
- **Cost:** Free tier available (5 API calls/minute)
- **Premium:** $49.99/month (unlimited)
- **Coverage:** International markets
- **Best for:** Real-time quotes with technical indicators

#### B. Polygon.io
- **Cost:** $99/month for real-time data
- **Coverage:** US markets primarily
- **Best for:** High-frequency trading data

#### C. IEX Cloud
- **Cost:** $9/month for basic, $99/month for premium
- **Coverage:** US markets
- **Best for:** Reliable real-time data

### Step 3: PSX-Specific APIs

#### A. Business Recorder API
```python
# Example endpoint
"https://www.brecorder.com/api/stocks/{symbol}"
```

#### B. KTrade Securities API
```python
# Contact KTrade for API access
"https://api.ktrade.com.pk/v1/quotes/{symbol}"
```

#### C. Tez Financial API
```python
# Contact Tez Financial for access
"https://api.tezfinancial.com/market-data/{symbol}"
```

## 💻 Implementation Guide

### Current System Architecture
```
User Request → PSX DPS API → PSX Data Reader → Realistic Simulation
```

### Enhanced Architecture
```
User Request → Premium API → PSX DPS API → PSX Data Reader → Cache → Simulation
```

## 🔑 API Key Management

### Secure Storage
1. **Never commit API keys to GitHub**
2. **Use environment variables**
3. **Encrypt sensitive keys**
4. **Rotate keys regularly**

### Key Validation
```python
def validate_api_keys():
    required_keys = ['PSX_API_KEY', 'BACKUP_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Missing API keys: {missing_keys}")
    return len(missing_keys) == 0
```

## 📊 Data Quality Monitoring

### Real-Time Validation
- **Price sanity checks** (max 10% change per minute)
- **Volume validation** (reasonable ranges)
- **Timestamp verification** (recent data)
- **Missing data detection**

### Fallback Strategy
1. **Primary API fails** → Switch to secondary
2. **All APIs fail** → Use cached data
3. **Cache expired** → Use realistic simulation
4. **Log all failures** for monitoring

## 🚨 Rate Limiting & Best Practices

### Rate Limiting
```python
# Built-in rate limiting
API_CALLS_PER_MINUTE = 60
CACHE_DURATION = 30  # seconds
```

### Best Practices
1. **Cache frequently requested data**
2. **Batch API calls when possible**
3. **Use WebSocket connections for real-time data**
4. **Implement exponential backoff for retries**
5. **Monitor API usage and costs**

## 🔍 Testing & Validation

### API Health Checks
```python
def test_api_connectivity():
    apis = ['psx_dps', 'psx_data_reader', 'premium_api']
    results = {}
    for api in apis:
        results[api] = test_single_api(api)
    return results
```

### Data Validation
```python
def validate_price_data(data):
    checks = {
        'has_ohlcv': all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']),
        'price_range_valid': (data['High'] >= data['Low']).all(),
        'volume_positive': (data['Volume'] >= 0).all(),
        'recent_timestamp': data.index[-1] > datetime.now() - timedelta(hours=1)
    }
    return all(checks.values()), checks
```

## 🛠️ Configuration Files

### config/api_settings.yml
```yaml
apis:
  psx_dps:
    enabled: true
    base_url: "https://dps.psx.com.pk"
    timeout: 10
    retry_count: 3
  
  psx_data_reader:
    enabled: true
    cache_duration: 300
  
  premium_api:
    enabled: false
    provider: "alpha_vantage"
    key_env_var: "ALPHA_VANTAGE_API_KEY"

rate_limiting:
  calls_per_minute: 60
  cache_duration: 30

data_validation:
  max_price_change_percent: 10
  min_volume: 0
  max_data_age_minutes: 60
```

## 📈 Monitoring & Alerting

### Key Metrics to Monitor
- **API response times**
- **Success/failure rates**
- **Data freshness**
- **Cache hit rates**
- **Daily API usage**

### Alert Conditions
- **API downtime > 5 minutes**
- **Data age > 10 minutes**
- **Price anomalies detected**
- **Rate limit exceeded**

## 🎯 Next Steps

1. **Choose your preferred premium API** (if needed)
2. **Set up environment variables**
3. **Configure rate limiting**
4. **Test API connectivity**
5. **Monitor data quality**
6. **Set up alerting**

## 📞 Support Contacts

### PSX Official
- **Website:** https://www.psx.com.pk/
- **Data Services:** https://dps.psx.com.pk/

### Premium Providers
- **Alpha Vantage:** support@alphavantage.co
- **Polygon.io:** support@polygon.io
- **IEX Cloud:** support@iexcloud.io

---

*This guide ensures reliable real-time pricing for your intraday trading system with proper fallback mechanisms and monitoring.*