# 🎯 ALGORITHMIC TRADING TESTING HIT LIST

## 📊 **Complete Testing Protocol for Your Fixed Algorithm**

### **Phase 1: Core PSX Stocks Testing (High Priority)**

#### 🏦 **Banking Sector** (Stable, predictable trends)
- **UBL** (United Bank Limited) - Your original problem stock, now fixed!
- **MCB** (MCB Bank Limited) - Strong trends, good for testing
- **HBL** (Habib Bank Limited) - High liquidity, clear patterns
- **ABL** (Allied Bank Limited) - Medium volatility
- **NBP** (National Bank of Pakistan) - Government backing, stable

#### ⛽ **Oil & Gas Sector** (High volatility, trend-following)
- **PPL** (Pakistan Petroleum Limited) - Strong trends
- **OGDC** (Oil and Gas Development Company) - Government sector
- **POL** (Pakistan Oilfields Limited) - High-value stock
- **MARI** (Mari Petroleum Company) - Premium stock, clear signals
- **PSO** (Pakistan State Oil) - Commodity-linked, volatile

#### 🏗️ **Cement Sector** (Cyclical, good for RSI testing)
- **LUCK** (Lucky Cement Limited) - High liquidity
- **DGKC** (D.G. Khan Cement Company) - Construction-linked
- **FCCL** (Fauji Cement Company Limited) - Military-linked
- **MLCF** (Maple Leaf Cement Factory) - Smaller cap

#### 🧪 **Chemicals & Fertilizers** (Agricultural cycles)
- **ENGRO** (Engro Corporation Limited) - Diversified
- **FFC** (Fauji Fertilizer Company) - Seasonal patterns
- **FATIMA** (Fatima Fertilizer Company) - Agricultural demand

---

## 🧪 **Testing Matrix - What to Look For**

### **Expected Results Per Stock Category:**

#### **Banking Stocks (Stable Testing)**
```
Expected Results:
- Win Rate: 60-70% (stable trends)
- Profit Factor: 1.4-2.0 (consistent profits)  
- Max Drawdown: <10% (low volatility)
- Best for: RSI oversold/overbought signals

Test Focus:
✅ RSI 30/70 levels working properly
✅ Trend following in stable uptrends
✅ Stop-loss triggers at 2%
✅ Take-profit triggers at 4%
```

#### **Oil & Gas Stocks (Volatile Testing)**  
```
Expected Results:
- Win Rate: 50-65% (higher volatility)
- Profit Factor: 1.2-1.8 (larger swings)
- Max Drawdown: 10-15% (commodity volatility)
- Best for: MACD crossover signals

Test Focus:
✅ Algorithm handles high volatility
✅ Position sizing adjusts for risk
✅ Stop-losses prevent major losses
✅ Trend filtering works in both directions
```

#### **Cement Stocks (Cyclical Testing)**
```
Expected Results:  
- Win Rate: 55-65% (cyclical patterns)
- Profit Factor: 1.3-1.7 (seasonal cycles)
- Max Drawdown: <12% (moderate volatility)
- Best for: Mean reversion strategies

Test Focus:
✅ RSI signals work in ranging markets
✅ Trend detection in sideways action
✅ Signal confidence adjusts properly
✅ No conflicting BUY/SELL signals
```

---

## 📋 **Testing Checklist - Run This Exact Sequence**

### **Step 1: Individual Stock Testing**
For each stock in the hit list above:

1. **Navigate to**: Simple Algo Trading → Symbol Analysis
2. **Select Stock**: Choose from hit list
3. **Set Parameters**: 
   - Time Period: 90-180 days (optimal for testing)
   - Capital: 1,000,000 PKR (standard)
4. **Run Analysis**: Click "Generate Signals & Backtest"
5. **Record Results**: Use the template below

### **Step 2: Results Recording Template**

```
📊 STOCK: [SYMBOL] - [COMPANY NAME]
Sector: [Banking/Oil/Cement/Chemical]
Test Date: [DATE]

PERFORMANCE METRICS:
- Total Return: _____%
- Win Rate: _____%  
- Total Trades: _____
- Profit Factor: _____
- Max Drawdown: _____%
- Sharpe Ratio: _____
- Final Capital: _____ PKR

SIGNAL QUALITY:
- Average Confidence: _____%
- Strong Buy/Sell Signals: _____
- Hold Signals: _____

RISK MANAGEMENT:
- Stop-Loss Triggers: _____
- Take-Profit Triggers: _____
- Commission Paid: _____ PKR

ASSESSMENT: [EXCELLENT/GOOD/NEEDS IMPROVEMENT/POOR]
NOTES: ________________________________
```

### **Step 3: Success Criteria Benchmarks**

#### **🟢 EXCELLENT PERFORMANCE (Algorithm Working Perfectly)**
- Win Rate: >65%
- Profit Factor: >1.8
- Total Return: >12%
- Max Drawdown: <10%
- Sharpe Ratio: >1.0

#### **✅ GOOD PERFORMANCE (Algorithm Working Well)**
- Win Rate: 55-65%
- Profit Factor: 1.3-1.8
- Total Return: 5-12%
- Max Drawdown: 10-15%
- Sharpe Ratio: 0.5-1.0

#### **⚠️ NEEDS IMPROVEMENT (Marginal Performance)**
- Win Rate: 45-55%
- Profit Factor: 1.0-1.3
- Total Return: 0-5%
- Max Drawdown: 15-20%
- Sharpe Ratio: 0-0.5

#### **🔴 POOR PERFORMANCE (Still Issues)**
- Win Rate: <45%
- Profit Factor: <1.0
- Total Return: <0%
- Max Drawdown: >20%
- Sharpe Ratio: <0

---

## 🚨 **Critical Issues to Watch For**

### **RED FLAGS (Need Immediate Attention):**
1. **Win Rate still 0-20%** → Signal logic still broken
2. **All trades losing money** → Entry/exit logic reversed
3. **No stop-losses triggering** → Risk management not working
4. **Profit Factor < 0.5** → Losses far exceed profits
5. **Max Drawdown > 25%** → Position sizing too aggressive

### **YELLOW FLAGS (Monitor Closely):**
1. **Win Rate 30-45%** → Signals need refinement
2. **High commission costs** → Too many small trades
3. **Very short holding times** → Possible over-trading
4. **Inconsistent across sectors** → Indicator parameter issues

### **GREEN FLAGS (Algorithm Working):**
1. **Win Rate improving significantly** from 0%
2. **Stop-losses and take-profits triggering** properly
3. **Positive profit factors** across multiple stocks
4. **Reasonable drawdowns** under 15%
5. **Clear trend alignment** in signal reasoning

---

## 📈 **Advanced Testing Scenarios**

### **Scenario 1: Bull Market Testing**
**Test Stocks**: UBL, MCB, LUCK, ENGRO
**Expected**: Strong BUY signals, high win rates
**Focus**: RSI oversold entries work well

### **Scenario 2: Bear Market Testing**  
**Test Stocks**: Any declining stocks in the list
**Expected**: SELL signals, proper stop-losses
**Focus**: Risk management prevents major losses

### **Scenario 3: Sideways Market Testing**
**Test Stocks**: Stable banking stocks
**Expected**: More HOLD signals, moderate performance
**Focus**: Algorithm doesn't overtrade

### **Scenario 4: High Volatility Testing**
**Test Stocks**: Oil & Gas sector (MARI, POL, PPL)
**Expected**: Position sizing adjusts, controlled risk
**Focus**: Stop-losses prevent catastrophic losses

---

## 🎯 **Quick 30-Minute Test Protocol**

If you want a rapid assessment:

### **Priority Testing List (Top 5 Stocks)**
1. **UBL** (Your original problem - should now work!)
2. **LUCK** (High liquidity, clear trends)
3. **ENGRO** (Diversified, good patterns)
4. **PPL** (Volatile, stress test)
5. **MCB** (Banking stability test)

### **What to Look For in 30 Minutes:**
- **5 minutes per stock** testing
- **Focus on Win Rate** (should be >45% minimum)
- **Check Stop-Loss/Take-Profit** triggers
- **Verify no conflicting signals**
- **Ensure reasonable drawdowns**

---

## 📞 **Reporting Issues**

If you find problems during testing:

### **Document This Information:**
1. **Stock Symbol & Sector**
2. **Specific Performance Metrics**
3. **Screenshots of results**
4. **Error messages (if any)**
5. **Expected vs Actual behavior**

### **Common Issues & Solutions:**
- **Still 0% win rate** → Signal logic needs review
- **Very high drawdown** → Position sizing needs adjustment  
- **No stop-losses** → Risk management not implemented
- **Random results** → Indicator calculations may be off

---

## 🏆 **Success Validation**

### **Your Algorithm is FIXED if:**
✅ **Average win rate across 10 stocks >50%**
✅ **No stocks with 0% win rate**
✅ **Profit factors consistently >1.0**
✅ **Stop-losses and take-profits trigger properly**
✅ **Maximum drawdowns stay under 20%**
✅ **Clear improvement from original HBL results**

### **Next Steps After Successful Testing:**
1. **Paper trade** the algorithm for 1-2 weeks
2. **Start with small position sizes** (1-2% of capital)
3. **Monitor performance daily**
4. **Scale up** only after consistent profits
5. **Document lessons learned** for future improvements

**Remember**: The goal is to transform your 0% win rate to 50-65% win rate with controlled risk! 🚀📈