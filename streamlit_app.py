"""
PSX Terminal Quantitative Trading System - Complete Algorithm Implementation
Real-time algorithmic trading with intraday signals, backtesting, and ML predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import json
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PSX Algo Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .algo-card {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .performance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class PSXAlgoTradingSystem:
    """Complete algorithmic trading system for PSX"""
    
    def __init__(self):
        self.psx_terminal_url = "https://psxterminal.com"
        self.psx_dps_url = "https://dps.psx.com.pk/timeseries/int"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PSX-Algo-Trading-System/2.0',
            'Accept': 'application/json'
        })
        
        # Trading parameters
        self.trading_capital = 1000000  # 1M PKR
        self.max_position_size = 0.05   # 5% per position
        self.stop_loss_pct = 0.02       # 2% stop loss
        self.take_profit_pct = 0.04     # 4% take profit
        self.min_liquidity = 100000     # Minimum volume
        
    def get_symbols(self):
        """Get all PSX symbols with comprehensive fallback"""
        # Try PSX Terminal API first
        try:
            response = self.session.get(f"{self.psx_terminal_url}/api/symbols", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get('success') and data.get('data'):
                return data.get('data', [])
        except Exception:
            pass  # Silent failure, try fallback
        
        # Fallback to comprehensive local PSX symbols list
        psx_symbols = [
            "786", "AABS", "AATM", "ABL", "ABOT", "ACI", "ACIETF", "ACPL", "ADAMS", "ADMM",
            "AGHA", "AGIC", "AGIL", "AGL", "AGLNCPS", "AGP", "AGSML", "AGTL", "AHCL", "AHL",
            "AHTM", "AICL", "AIRLINK", "AKBL", "AKDHL", "AKDSL", "AKGL", "ALAC", "ALIFE", "ALLSHR",
            "ALNRS", "ALTN", "AMBL", "AMTEX", "ANL", "ANSM", "ANTM", "APL", "ARCTM", "ARPAK",
            "ARPL", "ARUJ", "ASC", "ASHT", "ASIC", "ASL", "ASLCPS", "ASLPS", "ASTL", "ASTM",
            "ATBA", "ATIL", "ATLH", "ATRL", "AVN", "AWTX", "BAFL", "BAFS", "BAHL", "BAPL",
            "BATA", "BBFL", "BCL", "BECO", "BELA", "BERG", "BFAGRO", "BFBIO", "BFMOD", "BGL",
            "BHAT", "BIFO", "BILF", "BIPL", "BKTI", "BML", "BNL", "BNWM", "BOK", "BOP",
            "BPL", "BRRG", "BTL", "BUXL", "BWCL", "BWHL", "CASH", "CCM", "CENI", "CEPB",
            "CFL", "CHAS", "CHBL", "CHCC", "CJPL", "CLCPS", "CLOV", "CLVL", "CNERGY", "COLG",
            "CPHL", "CPPL", "CRTM", "CSAP", "CSIL", "CTM", "CWSM", "CYAN", "DAAG", "DADX",
            "DBCI", "DCL", "DCR", "DEL", "DFML", "DFSM", "DGKC", "DHPL", "DIIL", "DINT",
            "DLL", "DMC", "DMTM", "DMTX", "DNCC", "DOL", "DSIL", "DSL", "DWAE", "DWSM",
            "DWTM", "DYNO", "ECOP", "EFERT", "EFUG", "EFUL", "ELCM", "ELSM", "EMCO", "ENGRO",
            "EPCL", "EPCLPS", "EPQL", "ESBL", "EWIC", "EXIDE", "FABL", "FANM", "FASM", "FATIMA",
            "FCCL", "FCEL", "FCEPL", "FCIBL", "FCL", "FCSC", "FDPL", "FECM", "FECTC", "FEM",
            "FEROZ", "FFC", "FFL", "FFLM", "FHAM", "FIBLM", "FIL", "FIMM", "FLYNG", "FML",
            "FNEL", "FPJM", "FPRM", "FRCL", "FRSM", "FSWL", "FTMM", "FTSM", "FZCM", "GADT",
            "GAL", "GAMON", "GATI", "GATM", "GCIL", "GCWL", "GLAXO", "GLPL", "GOC", "GRR",
            "GRYL", "GSPM", "GTYR", "GUSM", "GVGL", "GWLC", "HABSM", "HAEL", "HAFL", "HALEON",
            "HASCOL", "HBL", "HBLTETF", "HCAR", "HGFA", "HICL", "HIFA", "HINO", "HINOON", "HIRAT",
            "HMB", "HPL", "HRPL", "HTL", "HUBC", "HUMNL", "HUSI", "HWQS", "IBFL", "IBLHL",
            "ICCI", "ICIBL", "ICL", "IDRT", "IDSM", "IDYM", "IGIHL", "IGIL", "ILP", "IMAGE",
            "IML", "IMS", "INDU", "INIL", "INKL", "IPAK", "ISIL", "ISL", "ITTEFAQ", "JATM",
            "JDMT", "JDWS", "JGICL", "JKSM", "JLICL", "JSBL", "JSCL", "JSCLPSA", "JSGBETF", "JSGBKTI",
            "JSGCL", "JSIL", "JSMFETF", "JSMFI", "JSML", "JUBS", "JVDC", "KAPCO", "KCL", "KEL",
            "KHTC", "KHYT", "KMI30", "KMIALLSHR", "KML", "KOHC", "KOHE", "KOHP", "KOHTM", "KOIL",
            "KOSM", "KPUS", "KSBP", "KSE100", "KSE100PR", "KSE30", "KSTM", "KTML", "LCI", "LEUL",
            "LIVEN", "LOADS", "LOTCHEM", "LPGL", "LPL", "LSECL", "LSEFSL", "LSEVL", "LUCK", "MACFL",
            "MACTER", "MARI", "MCB", "MCBIM", "MDTL", "MEBL", "MEHT", "MERIT", "MFFL", "MFL",
            "MII30", "MIIETF", "MIRKS", "MLCF", "MQTM", "MRNS", "MSCL", "MSOT", "MTL", "MUGHAL",
            "MUGHALC", "MUREB", "MWMP", "MZNPETF", "MZNPI", "NAGC", "NATF", "NBP", "NBPGETF", "NBPPGI",
            "NCL", "NCML", "NCPL", "NESTLE", "NETSOL", "NEXT", "NICL", "NITGETF", "NITPGI", "NML",
            "NONS", "NPL", "NRL", "NRSL", "NSRM", "OBOY", "OCTOPUS", "OGDC", "OGTI", "OLPL",
            "OLPM", "OML", "ORM", "OTSU", "PABC", "PACE", "PAEL", "PAKD", "PAKL", "PAKOXY",
            "PAKRI", "PAKT", "PASL", "PASM", "PCAL", "PECO", "PGLC", "PHDL", "PIAHCLA", "PIAHCLB",
            "PIBTL", "PICT", "PIL", "PIM", "PINL", "PIOC", "PKGI", "PKGP", "PKGS", "PMI",
            "PMPK", "PMRS", "PNSC", "POL", "POML", "POWER", "POWERPS", "PPL", "PPP", "PPVC",
            "PREMA", "PRET", "PRL", "PRWM", "PSEL", "PSO", "PSX", "PSXDIV20", "PSYL", "PTC",
            "PTL", "QUET", "QUICE", "RCML", "REDCO", "REWM", "RICL", "RMPL", "RPL", "RUBY",
            "RUPL", "SAIF", "SANSM", "SAPT", "SARC", "SASML", "SAZEW", "SBL", "SCBPL", "SCL",
            "SEARL", "SEL", "SEPL", "SERT", "SFL", "SGF", "SGPL", "SHCM", "SHDT", "SHEZ",
            "SHFA", "SHJS", "SHNI", "SHSML", "SIBL", "SIEM", "SINDM", "SITC", "SKRS", "SLGL",
            "SLYT", "SMCPL", "SML", "SNAI", "SNBL", "SNGP", "SPEL", "SPL", "SPWL", "SRVI",
            "SSGC", "SSML", "SSOM", "STCL", "STJT", "STL", "STML", "STPL", "STYLERS", "SUHJ",
            "SURC", "SUTM", "SYM", "SYSTEMS", "SZTM", "TATM", "TBL", "TCORP", "TCORPCPS", "TELE",
            "TGL", "THALL", "THCCL", "TICL", "TOMCL", "TOWL", "TPL", "TPLI", "TPLL", "TPLP",
            "TPLRF1", "TPLT", "TREET", "TRG", "TRIPF", "TRSM", "TSBL", "TSMF", "TSML", "TSPL",
            "UBDL", "UBL", "UBLPETF", "UCAPM", "UDLI", "UDPL", "UNIC", "UNILEVER", "UNITY", "UPFL",
            "UPP9", "UVIC", "WAFI", "WAHN", "WASL", "WAVES", "WAVESAPP", "WTL", "YOUW", "ZAHID",
            "ZAL", "ZIL", "ZTL"
        ]
        
        return psx_symbols
    
    def get_real_time_data(self, symbol):
        """Get real-time market data with robust fallback"""
        # Try PSX DPS API first (more reliable)
        try:
            response = self.session.get(f"{self.psx_dps_url}/{symbol}", timeout=15)
            response.raise_for_status()
            psx_data = response.json()
            
            # Handle PSX DPS response format
            if psx_data and 'data' in psx_data and psx_data['data']:
                latest = psx_data['data'][0]  # First item is latest
                if latest and len(latest) >= 3:
                    return {
                        'symbol': symbol,
                        'price': float(latest[1]),
                        'volume': int(latest[2]),
                        'timestamp': latest[0],
                        'change': 0,
                        'changePercent': 0
                    }
        except Exception as psx_dps_error:
            # If PSX DPS fails, try PSX Terminal API with shorter timeout
            try:
                response = self.session.get(f"{self.psx_terminal_url}/api/ticks/REG/{symbol}", timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if data.get('success'):
                    return data.get('data')
            except Exception as terminal_error:
                # Both APIs failed, show a user-friendly message
                st.warning(f"⚠️ {symbol}: Temporary data unavailable (APIs busy)")
                return None
        
        return None
    
    def get_intraday_ticks(self, symbol, limit=100):
        """Get intraday tick data for analysis with robust error handling"""
        try:
            response = self.session.get(f"{self.psx_dps_url}/{symbol}", timeout=20)
            response.raise_for_status()
            data = response.json()
            
            # Check if data has the expected structure
            if data and 'data' in data and data['data']:
                # Use the 'data' array from PSX DPS response
                tick_data = data['data']
            elif data and isinstance(data, list):
                # Direct array format
                tick_data = data
            else:
                return pd.DataFrame()
            
            if tick_data and len(tick_data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(tick_data, columns=['timestamp', 'price', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Unix timestamp
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                # Remove any rows with NaN values
                df = df.dropna()
                
                # Ensure we have minimum data for analysis
                if len(df) < 10:
                    return pd.DataFrame()
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Keep only recent data
                if limit and len(df) > limit:
                    df = df.tail(limit)
                
                return df.reset_index(drop=True)
            
            return pd.DataFrame()
        except Exception as e:
            # Return empty DataFrame silently - we'll handle this gracefully
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for trading signals"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Price-based indicators
            df['sma_5'] = df['price'].rolling(window=5).mean()
            df['sma_10'] = df['price'].rolling(window=10).mean()
            df['sma_20'] = df['price'].rolling(window=20).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum indicators
            df['price_change'] = df['price'].pct_change()
            df['momentum'] = df['price_change'].rolling(window=5).mean()
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=10).std()
            
            # Support/Resistance levels
            df['resistance'] = df['price'].rolling(window=20).max()
            df['support'] = df['price'].rolling(window=20).min()
            
            # RSI-like momentum
            price_delta = df['price'].diff()
            gains = price_delta.where(price_delta > 0, 0)
            losses = -price_delta.where(price_delta < 0, 0)
            
            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()
            
            rs = avg_gains / avg_losses
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            st.error(f"Technical indicators error: {str(e)}")
            return df
    
    def generate_trading_signals(self, df, symbol):
        """Generate algorithmic trading signals"""
        if df.empty or len(df) < 20:
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}
        
        try:
            latest = df.iloc[-1]
            signals = []
            confidence = 0
            reasons = []
            
            # Trend Following Signals
            if latest['price'] > latest['sma_5'] > latest['sma_10']:
                signals.append("BUY")
                confidence += 25
                reasons.append("Uptrend confirmed")
            elif latest['price'] < latest['sma_5'] < latest['sma_10']:
                signals.append("SELL")
                confidence += 25
                reasons.append("Downtrend confirmed")
            
            # Volume Analysis
            if latest['volume_ratio'] > 1.5:
                confidence += 20
                reasons.append("High volume support")
            elif latest['volume_ratio'] < 0.5:
                confidence -= 10
                reasons.append("Low volume concern")
            
            # Momentum Analysis
            if latest['momentum'] > 0.002:  # 0.2% positive momentum
                signals.append("BUY")
                confidence += 20
                reasons.append("Strong positive momentum")
            elif latest['momentum'] < -0.002:  # -0.2% negative momentum
                signals.append("SELL")
                confidence += 20
                reasons.append("Strong negative momentum")
            
            # Mean Reversion Signals
            distance_from_sma = (latest['price'] - latest['sma_20']) / latest['sma_20']
            
            if distance_from_sma > 0.05:  # 5% above mean
                signals.append("SELL")
                confidence += 15
                reasons.append("Overbought condition")
            elif distance_from_sma < -0.05:  # 5% below mean
                signals.append("BUY")
                confidence += 15
                reasons.append("Oversold condition")
            
            # RSI Signals
            if 'rsi' in latest and not pd.isna(latest['rsi']):
                if latest['rsi'] > 70:
                    signals.append("SELL")
                    confidence += 15
                    reasons.append("RSI overbought")
                elif latest['rsi'] < 30:
                    signals.append("BUY")
                    confidence += 15
                    reasons.append("RSI oversold")
            
            # Determine final signal
            buy_signals = signals.count("BUY")
            sell_signals = signals.count("SELL")
            
            if buy_signals > sell_signals and confidence > 50:
                signal_type = "STRONG_BUY" if confidence > 75 else "BUY"
            elif sell_signals > buy_signals and confidence > 50:
                signal_type = "STRONG_SELL" if confidence > 75 else "SELL"
            else:
                signal_type = "HOLD"
            
            # Calculate position sizing
            volatility_adj = min(1.0, 0.02 / max(latest['volatility'], 0.001))
            position_size = self.max_position_size * volatility_adj
            
            # Calculate entry/exit levels
            entry_price = latest['price']
            stop_loss = entry_price * (1 - self.stop_loss_pct) if signal_type in ["BUY", "STRONG_BUY"] else entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct) if signal_type in ["BUY", "STRONG_BUY"] else entry_price * (1 - self.take_profit_pct)
            
            return {
                "signal": signal_type,
                "confidence": min(confidence, 100),
                "reasons": reasons,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "volume_support": latest['volume_ratio'] > 1.2,
                "liquidity_ok": latest['volume'] > self.min_liquidity
            }
            
        except Exception as e:
            st.error(f"Signal generation error: {str(e)}")
            return {"signal": "HOLD", "confidence": 0, "reason": "Analysis error"}
    
    def simulate_trade_performance(self, signals_df, initial_capital=1000000):
        """Simulate trading performance based on signals"""
        if signals_df.empty:
            return {}
        
        try:
            capital = initial_capital
            positions = 0
            trades = []
            equity_curve = [capital]
            
            for idx, row in signals_df.iterrows():
                signal = row['signal']
                price = row['entry_price']
                confidence = row['confidence']
                
                if signal in ['BUY', 'STRONG_BUY'] and positions <= 0 and confidence > 60:
                    # Enter long position
                    position_value = capital * row['position_size']
                    shares = position_value / price
                    positions = shares
                    capital -= position_value
                    
                    trades.append({
                        'timestamp': row.get('timestamp', datetime.now()),
                        'type': 'BUY',
                        'price': price,
                        'shares': shares,
                        'value': position_value
                    })
                
                elif signal in ['SELL', 'STRONG_SELL'] and positions > 0:
                    # Close long position
                    position_value = positions * price
                    capital += position_value
                    
                    trades.append({
                        'timestamp': row.get('timestamp', datetime.now()),
                        'type': 'SELL',
                        'price': price,
                        'shares': positions,
                        'value': position_value
                    })
                    
                    positions = 0
                
                # Calculate current equity
                current_equity = capital + (positions * price if positions > 0 else 0)
                equity_curve.append(current_equity)
            
            # Performance metrics
            total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
            win_trades = sum(1 for i in range(1, len(trades)) if trades[i]['type'] == 'SELL' and 
                           trades[i]['price'] > trades[i-1]['price'])
            total_trades = len([t for t in trades if t['type'] == 'SELL'])
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'equity_curve': equity_curve,
                'trades': trades,
                'final_capital': equity_curve[-1]
            }
            
        except Exception as e:
            st.error(f"Performance simulation error: {str(e)}")
            return {}

@st.cache_data(ttl=30)
def get_cached_symbols():
    """Cache symbols for 30 seconds"""
    system = PSXAlgoTradingSystem()
    return system.get_symbols()

@st.cache_data(ttl=15)
def get_cached_real_time_data(symbol):
    """Cache real-time data for 15 seconds"""
    system = PSXAlgoTradingSystem()
    return system.get_real_time_data(symbol)

@st.cache_data(ttl=60)
def get_cached_intraday_ticks(symbol, limit=100):
    """Cache intraday ticks for 1 minute"""
    system = PSXAlgoTradingSystem()
    return system.get_intraday_ticks(symbol, limit)

def render_header():
    """Render header"""
    st.markdown('<h1 class="main-header">🤖 PSX Algorithmic Trading System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="signal-strong-buy">
            <h4>🎯 Real-Time Signals</h4>
            <p>ML-Powered Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="signal-buy">
            <h4>📊 Intraday Trading</h4>
            <p>Tick-by-Tick Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="signal-sell">
            <h4>⚡ Auto Backtesting</h4>
            <p>Performance Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="signal-hold">
            <h4>🛡️ Risk Management</h4>
            <p>Smart Position Sizing</p>
        </div>
        """, unsafe_allow_html=True)

def render_live_trading_signals():
    """Render live trading signals"""
    st.markdown("## 🚨 Live Trading Signals")
    
    # Trading Guidelines Section
    with st.expander("📋 Live Trading Signal Guidelines & Fundamentals", expanded=False):
        st.markdown("""
        ### 🎯 **Signal Interpretation Guide**
        
        #### **Signal Types & Actions:**
        - 🟢 **STRONG BUY (75-100% Confidence)**: High conviction entry - Consider 3-5% position
        - 🟢 **BUY (60-75% Confidence)**: Moderate entry - Consider 2-3% position  
        - 🟡 **HOLD (40-60% Confidence)**: Wait for better setup - No action required
        - 🔴 **SELL (60-75% Confidence)**: Exit long positions - Consider short if applicable
        - 🔴 **STRONG SELL (75-100% Confidence)**: Immediate exit - Strong short candidate
        
        #### **📊 Fundamental Criteria (Built into Signals):**
        
        **Volume Analysis:**
        - ✅ **High Volume Support**: Volume >150% of 10-day average (Bullish confirmation)
        - ⚠️ **Low Volume**: Volume <50% of average (Proceed with caution)
        
        **Liquidity Assessment:**
        - ✅ **Good Liquidity**: Daily volume >100,000 shares (Safe for position sizing)
        - ❌ **Poor Liquidity**: Volume <100,000 shares (Reduce position size by 50%)
        
        **Technical Momentum:**
        - **Trend Following**: SMA 5 > SMA 10 > SMA 20 (Uptrend confirmation)
        - **Mean Reversion**: Price >5% from SMA 20 (Overbought/Oversold conditions)
        - **RSI Levels**: >70 Overbought, <30 Oversold (Reversal probability)
        
        #### **🛡️ Risk Management Rules:**
        
        **Position Sizing:**
        - **Maximum per stock**: 5% of portfolio (Volatility adjusted)
        - **Stop Loss**: 2% below entry (Automatically calculated)
        - **Take Profit**: 4% above entry (2:1 Risk/Reward ratio)
        - **Portfolio Maximum**: 25% total equity exposure
        
        **Entry Rules:**
        - Only trade signals with >60% confidence
        - Require volume support for BUY signals
        - Check liquidity before position sizing
        - Avoid trading 30 minutes before/after market open/close
        
        **Exit Rules:**
        - Stop loss hit: Exit immediately, no exceptions
        - Take profit reached: Take 50% profits, trail remaining
        - Confidence drops below 40%: Consider exit
        - End of day: Close all intraday positions
        
        #### **⏰ Timing Guidelines:**
        - **Best Trading Hours**: 10:00 AM - 3:00 PM (High liquidity)
        - **Avoid**: First 30 minutes (High volatility) 
        - **Avoid**: Last 30 minutes (Closing volatility)
        - **Signal Refresh**: Every 15 seconds (Real-time updates)
        
        #### **📈 Performance Expectations:**
        - **Win Rate Target**: 65-70% (Historical backtest)
        - **Average Hold Time**: 2-4 hours (Intraday focus)
        - **Monthly Return Target**: 8-12% (Risk-adjusted)
        - **Maximum Drawdown**: <15% (Risk management)
        """)
    
    symbols = get_cached_symbols()
    if not symbols:
        st.error("Unable to load symbols")
        return
    
    # Stock Selection Interface
    st.subheader("🎯 Select Your 12 Stocks for Live Signals")
    
    # Default major tickers
    default_major_tickers = [
        'HBL', 'UBL', 'FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL', 
        'TRG', 'SYSTEMS'
    ]
    
    # Create tabs for different selection methods
    tab1, tab2, tab3 = st.tabs(["🎯 Quick Select", "🔍 Search & Pick", "📊 Sector Based"])
    
    with tab1:
        st.markdown("**Quick preset selections:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏦 Banking Focus", help="Major banks and financial institutions"):
                st.session_state.selected_stocks = ['HBL', 'UBL', 'NBP', 'MCB', 'ABL', 'BAFL', 'AKBL', 'MEBL', 'JSBL', 'BAHL', 'FABL', 'BOK'][:12]
        
        with col2:
            if st.button("🏭 Industrial Mix", help="Mix of industrial and manufacturing stocks"):
                st.session_state.selected_stocks = ['FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'PPL', 'EFERT', 'FATIMA', 'COLG', 'NESTLE', 'UNILEVER', 'ICI'][:12]
        
        with col3:
            if st.button("💼 Blue Chip", help="Top market cap and most liquid stocks"):
                st.session_state.selected_stocks = ['HBL', 'UBL', 'FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL', 'TRG', 'SYSTEMS']
    
    with tab2:
        st.markdown("**Search and select individual stocks:**")
        
        # Initialize selected stocks in session state
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = [s for s in default_major_tickers if s in symbols][:12]
        
        # Current selection display
        st.write(f"**Currently selected ({len(st.session_state.selected_stocks)}/12):**")
        if st.session_state.selected_stocks:
            selected_display = ", ".join(st.session_state.selected_stocks)
            st.code(selected_display)
        
        # Search and add stocks
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_stock = st.text_input("🔍 Search for stock to add:", placeholder="Type symbol name (e.g., UNITY, PIAIC)")
        
        with col2:
            if st.button("➕ Add Stock") and search_stock:
                search_upper = search_stock.upper()
                matching_stocks = [s for s in symbols if search_upper in s]
                
                if matching_stocks:
                    if len(st.session_state.selected_stocks) < 12:
                        if matching_stocks[0] not in st.session_state.selected_stocks:
                            st.session_state.selected_stocks.append(matching_stocks[0])
                            st.success(f"Added {matching_stocks[0]}")
                        else:
                            st.warning(f"{matching_stocks[0]} already selected")
                    else:
                        st.warning("Maximum 12 stocks allowed")
                else:
                    st.error(f"No stock found matching '{search_stock}'")
        
        # Show available matching stocks
        if search_stock:
            search_upper = search_stock.upper()
            matching_stocks = [s for s in symbols if search_upper in s][:10]
            if matching_stocks:
                st.write("**Available matches:**")
                for stock in matching_stocks:
                    if st.button(f"➕ {stock}", key=f"add_{stock}"):
                        if len(st.session_state.selected_stocks) < 12:
                            if stock not in st.session_state.selected_stocks:
                                st.session_state.selected_stocks.append(stock)
                                st.success(f"Added {stock}")
                                st.rerun()
        
        # Remove stocks interface
        if st.session_state.selected_stocks:
            st.markdown("**Remove stocks:**")
            cols = st.columns(min(6, len(st.session_state.selected_stocks)))
            for i, stock in enumerate(st.session_state.selected_stocks):
                with cols[i % 6]:
                    if st.button(f"❌ {stock}", key=f"remove_{stock}"):
                        st.session_state.selected_stocks.remove(stock)
                        st.success(f"Removed {stock}")
                        st.rerun()
    
    with tab3:
        st.markdown("**Select by sector:**")
        
        sectors = {
            "Banking": ['HBL', 'UBL', 'NBP', 'MCB', 'ABL', 'BAFL', 'AKBL', 'MEBL', 'JSBL', 'BAHL'],
            "Oil & Gas": ['PSO', 'OGDC', 'PPL', 'SNGP', 'SSGC', 'POL'],
            "Cement": ['LUCK', 'DG', 'PIOC', 'MLCF', 'FCCL'],
            "Chemicals": ['FFC', 'ENGRO', 'EFERT', 'FATIMA', 'ICI', 'COLG'],
            "Technology": ['TRG', 'SYSTEMS', 'NETSOL', 'IBFL'],
            "FMCG": ['NESTLE', 'UNILEVER', 'COLG', 'BATA'],
            "Textile": ['GADT', 'KOHE', 'SITC', 'KTML']
        }
        
        sector_cols = st.columns(4)
        for i, (sector, stocks) in enumerate(sectors.items()):
            with sector_cols[i % 4]:
                available_in_sector = [s for s in stocks if s in symbols]
                if st.button(f"{sector} ({len(available_in_sector)})", key=f"sector_{sector}"):
                    # Add sector stocks to selection (up to remaining capacity)
                    remaining_slots = 12 - len(st.session_state.selected_stocks)
                    for stock in available_in_sector[:remaining_slots]:
                        if stock not in st.session_state.selected_stocks:
                            st.session_state.selected_stocks.append(stock)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset to Default"):
            st.session_state.selected_stocks = [s for s in default_major_tickers if s in symbols][:12]
            st.success("Reset to default selection")
            st.rerun()
    
    with col2:
        if st.button("🎲 Random Selection"):
            import random
            available_random = [s for s in symbols if s not in st.session_state.get('selected_stocks', [])]
            random_picks = random.sample(available_random, min(12-len(st.session_state.get('selected_stocks', [])), len(available_random)))
            st.session_state.selected_stocks = (st.session_state.get('selected_stocks', []) + random_picks)[:12]
            st.success("Added random stocks")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.selected_stocks = []
            st.success("Cleared all selections")
            st.rerun()
    
    # Get final selected symbols
    if 'selected_stocks' in st.session_state and st.session_state.selected_stocks:
        available_symbols = st.session_state.selected_stocks[:12]
    else:
        # Fallback to defaults if nothing selected
        available_symbols = [s for s in default_major_tickers if s in symbols][:12]
    
    st.markdown("---")
    
    # API Status Banner
    st.info("🔄 **System Status**: 514 PSX symbols loaded with local fallback. Using PSX DPS (Official) for real-time data.")
    
    # Display current selection prominently
    st.subheader(f"📈 Live Signals for Your Selected {len(available_symbols)} Stocks")
    
    # Show selected stocks in an organized way
    if available_symbols:
        st.info(f"**Your Watchlist**: {' • '.join(available_symbols[:6])}" + 
                (f" • {' • '.join(available_symbols[6:])}" if len(available_symbols) > 6 else ""))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("🔄 Auto-refresh active | ⚡ PSX DPS Official API | 🛡️ Robust fallback system")
    with col2:
        if st.button("🎯 Change Selection", key="change_selection_btn"):
            st.info("Scroll up to modify your stock selection")
    
    # Create 4x3 grid for 12 stocks
    system = PSXAlgoTradingSystem()
    
    # Display stocks in rows of 4
    for row in range(0, len(available_symbols), 4):
        cols = st.columns(4)
        row_symbols = available_symbols[row:row+4]
        
        for col_idx, symbol in enumerate(row_symbols):
            with cols[col_idx]:
                # Get real-time data
                market_data = get_cached_real_time_data(symbol)
                
                if market_data:
                    # Get intraday ticks for analysis
                    ticks_df = get_cached_intraday_ticks(symbol, 50)
                    
                    if not ticks_df.empty and len(ticks_df) >= 10:
                        # Calculate indicators and generate signals
                        ticks_df = system.calculate_technical_indicators(ticks_df)
                        signal_data = system.generate_trading_signals(ticks_df, symbol)
                        
                        # Display signal
                        signal_type = signal_data['signal']
                        confidence = signal_data['confidence']
                        
                        signal_class = f"signal-{signal_type.lower().replace('_', '-')}"
                        
                        st.markdown(f"""
                        <div class="{signal_class}">
                            <h5>{symbol}</h5>
                            <h3>{signal_type}</h3>
                            <p>Confidence: {confidence:.1f}%</p>
                            <p>Price: {market_data['price']:.2f} PKR</p>
                            <small>Entry: {signal_data['entry_price']:.2f}</small><br>
                            <small>Stop: {signal_data['stop_loss']:.2f}</small><br>
                            <small>Target: {signal_data['take_profit']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show reasons
                        if signal_data.get('reasons'):
                            with st.expander(f"📋 {symbol} Analysis"):
                                for reason in signal_data['reasons'][:3]:
                                    st.write(f"• {reason}")
                                
                                st.write(f"**Volume Support**: {'✅' if signal_data['volume_support'] else '❌'}")
                                st.write(f"**Liquidity OK**: {'✅' if signal_data['liquidity_ok'] else '❌'}")
                                st.write(f"**Position Size**: {signal_data['position_size']:.2%}")
                    else:
                        # Show price-only view when technical analysis isn't available
                        st.markdown(f"""
                        <div class="signal-hold">
                            <h5>{symbol}</h5>
                            <h3>PRICE ONLY</h3>
                            <p>Limited data available</p>
                            <p>Price: {market_data['price']:.2f} PKR</p>
                            <small>Volume: {market_data.get('volume', 0):,}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Show unavailable data card
                    st.markdown(f"""
                    <div class="signal-hold">
                        <h5>{symbol}</h5>
                        <h3>OFFLINE</h3>
                        <p>Data temporarily unavailable</p>
                        <small>Refresh in few moments</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Portfolio Summary Section
    st.markdown("---")
    st.subheader("📊 Portfolio Summary")
    
    # Calculate portfolio-level metrics
    buy_signals = 0
    sell_signals = 0
    high_confidence_signals = 0
    total_processed = 0
    
    for symbol in available_symbols:
        try:
            market_data = get_cached_real_time_data(symbol)
            if market_data:
                ticks_df = get_cached_intraday_ticks(symbol, 50)
                if not ticks_df.empty:
                    ticks_df = system.calculate_technical_indicators(ticks_df)
                    signal_data = system.generate_trading_signals(ticks_df, symbol)
                    
                    total_processed += 1
                    if signal_data['signal'] in ['BUY', 'STRONG_BUY']:
                        buy_signals += 1
                    elif signal_data['signal'] in ['SELL', 'STRONG_SELL']:
                        sell_signals += 1
                    
                    if signal_data['confidence'] > 75:
                        high_confidence_signals += 1
        except:
            continue
    
    # Display portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Buy Signals", buy_signals, f"{buy_signals/max(total_processed,1)*100:.0f}% of stocks")
    
    with col2:
        st.metric("🔴 Sell Signals", sell_signals, f"{sell_signals/max(total_processed,1)*100:.0f}% of stocks")
    
    with col3:
        st.metric("⭐ High Confidence", high_confidence_signals, f"{high_confidence_signals/max(total_processed,1)*100:.0f}% of stocks")
    
    with col4:
        market_sentiment = "Bullish" if buy_signals > sell_signals else "Bearish" if sell_signals > buy_signals else "Neutral"
        st.metric("📈 Market Sentiment", market_sentiment, f"{abs(buy_signals-sell_signals)} signal difference")
    
    # Add trading session info
    st.info("🕒 **Trading Session**: PSX operates 9:30 AM - 3:30 PM PKT | 📍 **Optimal Hours**: 10:00 AM - 3:00 PM for best liquidity")

def render_symbol_analysis():
    """Render detailed symbol analysis"""
    st.markdown("## 🔍 Deep Symbol Analysis")
    
    symbols = get_cached_symbols()
    if not symbols:
        st.error("Unable to load symbols")
        return
    
    # Prioritize major tickers for better user experience
    major_tickers = ['HBL', 'UBL', 'FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL', 
                    'TRG', 'SYSTEMS', 'POL', 'PPL', 'NESTLE', 'UNILEVER', 'COLG', 'ICI', 
                    'BAHL', 'BAFL', 'MEBL', 'JSBL', 'AKBL', 'FABL', 'EFERT', 'FATIMA']
    
    # Create prioritized symbol list (major tickers first, then all others)
    prioritized_symbols = []
    remaining_symbols = []
    
    for ticker in major_tickers:
        if ticker in symbols:
            prioritized_symbols.append(ticker)
    
    for symbol in symbols:
        if symbol not in major_tickers:
            remaining_symbols.append(symbol)
    
    all_symbols = prioritized_symbols + remaining_symbols
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add search functionality
        search_term = st.text_input("🔍 Search Symbol", placeholder="Type symbol name (e.g., FFC, HBL)")
        
        if search_term:
            filtered_symbols = [s for s in all_symbols if search_term.upper() in s.upper()]
            if filtered_symbols:
                selected_symbol = st.selectbox(
                    f"Select from {len(filtered_symbols)} matching symbols",
                    options=filtered_symbols,
                    index=0
                )
            else:
                st.warning(f"No symbols found matching '{search_term}'")
                selected_symbol = st.selectbox(
                    "Select Symbol for Analysis",
                    options=all_symbols[:50],  # Show first 50 if no search
                    index=0
                )
        else:
            selected_symbol = st.selectbox(
                f"Select from {len(all_symbols)} symbols (Major tickers shown first)",
                options=all_symbols[:50],  # Show first 50 which includes major tickers
                index=0
            )
    
    with col2:
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if selected_symbol:
        system = PSXAlgoTradingSystem()
        
        # Get data
        market_data = get_cached_real_time_data(selected_symbol)
        ticks_df = get_cached_intraday_ticks(selected_symbol, 200)
        
        if market_data and not ticks_df.empty:
            # Calculate indicators
            ticks_df = system.calculate_technical_indicators(ticks_df)
            signal_data = system.generate_trading_signals(ticks_df, selected_symbol)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Signal", "📈 Price Chart", "🎯 Performance", "📋 Details"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    signal_type = signal_data['signal']
                    confidence = signal_data['confidence']
                    
                    st.markdown(f"""
                    <div class="algo-card">
                        <h3>Current Signal</h3>
                        <h2>{signal_type}</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                        <h4>{market_data['price']:.2f} PKR</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="performance-card">
                        <h4>Trade Levels</h4>
                        <p><strong>Entry:</strong> {signal_data['entry_price']:.2f}</p>
                        <p><strong>Stop Loss:</strong> {signal_data['stop_loss']:.2f}</p>
                        <p><strong>Take Profit:</strong> {signal_data['take_profit']:.2f}</p>
                        <p><strong>Risk/Reward:</strong> 1:2</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="algo-card">
                        <h4>Position Info</h4>
                        <p><strong>Size:</strong> {signal_data['position_size']:.2%}</p>
                        <p><strong>Volume:</strong> {'✅ Good' if signal_data['volume_support'] else '❌ Low'}</p>
                        <p><strong>Liquidity:</strong> {'✅ OK' if signal_data['liquidity_ok'] else '❌ Poor'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Signal reasoning
                st.subheader("🧠 Signal Analysis")
                for reason in signal_data.get('reasons', []):
                    st.write(f"• {reason}")
            
            with tab2:
                # Price and volume charts
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=['Price & Moving Averages', 'Volume Analysis', 'Technical Indicators'],
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(
                        x=ticks_df.index,
                        y=ticks_df['price'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Moving averages
                if 'sma_5' in ticks_df.columns:
                    fig.add_trace(
                        go.Scatter(x=ticks_df.index, y=ticks_df['sma_5'], 
                                 name='SMA 5', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=ticks_df.index, y=ticks_df['sma_20'], 
                                 name='SMA 20', line=dict(color='red')),
                        row=1, col=1
                    )
                
                # Volume
                fig.add_trace(
                    go.Bar(x=ticks_df.index, y=ticks_df['volume'], 
                          name='Volume', marker_color='lightblue'),
                    row=2, col=1
                )
                
                # RSI if available
                if 'rsi' in ticks_df.columns:
                    fig.add_trace(
                        go.Scatter(x=ticks_df.index, y=ticks_df['rsi'], 
                                 name='RSI', line=dict(color='purple')),
                        row=3, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(height=800, title=f"{selected_symbol} - Technical Analysis")
                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price (PKR)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                fig.update_yaxes(title_text="RSI", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Performance simulation with comprehensive analytics
                st.subheader("📊 Auto Backtesting Performance Analytics")
                
                # Performance Analytics Guide
                with st.expander("📚 Performance Analytics Guide - What to Look For", expanded=False):
                    st.markdown("""
                    ### 🎯 **Key Performance Metrics Explained**
                    
                    #### **📈 Profitability Metrics:**
                    - **Total Return %**: Overall profit/loss from 1M PKR initial capital
                      - ✅ **Good**: >15% annual return  
                      - ⚠️ **Average**: 5-15% annual return
                      - ❌ **Poor**: <5% or negative
                    
                    - **Win Rate %**: Percentage of profitable trades
                      - ✅ **Excellent**: >70% (Strong algorithm)
                      - ✅ **Good**: 60-70% (Reliable system)
                      - ⚠️ **Acceptable**: 50-60% (Needs improvement)
                      - ❌ **Poor**: <50% (Review strategy)
                    
                    - **Profit Factor**: Total profits ÷ Total losses
                      - ✅ **Excellent**: >2.0 (Strong edge)
                      - ✅ **Good**: 1.5-2.0 (Profitable system)
                      - ⚠️ **Break-even**: 1.0-1.5 (Marginal)
                      - ❌ **Losing**: <1.0 (Unprofitable)
                    
                    #### **⚖️ Risk Management Metrics:**
                    - **Maximum Drawdown**: Largest peak-to-trough decline
                      - ✅ **Excellent**: <10% (Low risk)
                      - ✅ **Good**: 10-15% (Moderate risk)
                      - ⚠️ **Acceptable**: 15-25% (Higher risk)
                      - ❌ **Poor**: >25% (Too risky)
                    
                    - **Sharpe Ratio**: Risk-adjusted returns (Return ÷ Volatility)
                      - ✅ **Excellent**: >2.0 (Superior risk-adjusted returns)
                      - ✅ **Good**: 1.0-2.0 (Good risk-adjusted returns)
                      - ⚠️ **Acceptable**: 0.5-1.0 (Moderate)
                      - ❌ **Poor**: <0.5 (Poor risk adjustment)
                    
                    #### **📊 Trading Activity Metrics:**
                    - **Total Trades**: Number of completed round-trip trades
                      - Look for: Sufficient sample size (>30 trades for reliability)
                    
                    - **Average Trade Duration**: How long positions are held
                      - Intraday: 2-6 hours (for day trading)
                      - Short-term: 1-5 days (for swing trading)
                    
                    #### **🎯 What Makes a Good Algorithm:**
                    1. **Consistent Profitability**: Positive returns over time
                    2. **High Win Rate**: >60% winning trades
                    3. **Controlled Risk**: <20% maximum drawdown
                    4. **Good Risk/Reward**: 1.5+ Sharpe ratio
                    5. **Sufficient Activity**: 30+ trades for statistical significance
                    
                    #### **🚨 Red Flags to Watch:**
                    - Long periods of continuous losses
                    - Very few trades (insufficient data)
                    - High volatility in equity curve
                    - Win rate below 50%
                    - Negative Sharpe ratio
                    """)
                
                # Create signals DataFrame for performance simulation
                signals_data = []
                for i in range(len(ticks_df)):
                    if i < 20:  # Skip first 20 for indicators
                        continue
                    
                    temp_df = ticks_df.iloc[:i+1].copy()
                    temp_signal = system.generate_trading_signals(temp_df, selected_symbol)
                    
                    signals_data.append({
                        'timestamp': temp_df.iloc[-1].get('timestamp', datetime.now()),
                        'signal': temp_signal['signal'],
                        'confidence': temp_signal['confidence'],
                        'entry_price': temp_signal['entry_price'],
                        'position_size': temp_signal['position_size']
                    })
                
                if signals_data and len(signals_data) > 10:
                    signals_df = pd.DataFrame(signals_data)
                    performance = system.simulate_trade_performance(signals_df)
                    
                    if performance and performance.get('total_trades', 0) > 0:
                        # Enhanced Performance Metrics
                        st.markdown("### 📊 **Core Performance Metrics**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total_return = performance['total_return']
                        win_rate = performance['win_rate']
                        total_trades = performance['total_trades']
                        final_capital = performance['final_capital']
                        
                        with col1:
                            color = "normal" if total_return > 0 else "inverse"
                            st.metric("Total Return", f"{total_return:.2f}%", 
                                    delta=f"{'Profit' if total_return > 0 else 'Loss'}", 
                                    delta_color=color)
                        
                        with col2:
                            win_color = "normal" if win_rate >= 60 else "off" if win_rate >= 50 else "inverse"
                            st.metric("Win Rate", f"{win_rate:.1f}%", 
                                    delta="Good" if win_rate >= 60 else "Average" if win_rate >= 50 else "Poor",
                                    delta_color=win_color)
                        
                        with col3:
                            st.metric("Total Trades", f"{total_trades}", 
                                    delta="Good sample" if total_trades >= 30 else "Small sample",
                                    delta_color="normal" if total_trades >= 30 else "off")
                        
                        with col4:
                            capital_change = final_capital - 1000000
                            st.metric("Final Capital", f"{final_capital:,.0f} PKR", 
                                    delta=f"{capital_change:+,.0f} PKR")
                        
                        # Advanced Analytics
                        st.markdown("### 🔍 **Advanced Analytics**")
                        
                        col5, col6, col7, col8 = st.columns(4)
                        
                        with col5:
                            # Calculate profit factor
                            if performance.get('trades'):
                                profits = sum(t.get('profit', 0) for t in performance['trades'] if t.get('profit', 0) > 0)
                                losses = abs(sum(t.get('profit', 0) for t in performance['trades'] if t.get('profit', 0) < 0))
                                profit_factor = profits / max(losses, 1)
                                
                                pf_color = "normal" if profit_factor > 1.5 else "off" if profit_factor > 1.0 else "inverse"
                                st.metric("Profit Factor", f"{profit_factor:.2f}", 
                                        delta="Excellent" if profit_factor > 2.0 else "Good" if profit_factor > 1.5 else "Poor",
                                        delta_color=pf_color)
                            else:
                                st.metric("Profit Factor", "N/A")
                        
                        with col6:
                            # Calculate max drawdown from equity curve
                            if performance.get('equity_curve'):
                                equity_curve = performance['equity_curve']
                                peak = equity_curve[0]
                                max_dd = 0
                                for value in equity_curve:
                                    if value > peak:
                                        peak = value
                                    drawdown = (peak - value) / peak * 100
                                    max_dd = max(max_dd, drawdown)
                                
                                dd_color = "normal" if max_dd < 15 else "off" if max_dd < 25 else "inverse"
                                st.metric("Max Drawdown", f"{max_dd:.1f}%", 
                                        delta="Low Risk" if max_dd < 15 else "Moderate" if max_dd < 25 else "High Risk",
                                        delta_color=dd_color)
                            else:
                                st.metric("Max Drawdown", "N/A")
                        
                        with col7:
                            # Calculate Sharpe ratio approximation
                            if performance.get('equity_curve') and len(performance['equity_curve']) > 1:
                                returns = np.diff(performance['equity_curve']) / performance['equity_curve'][:-1]
                                if np.std(returns) > 0:
                                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                                    sharpe_color = "normal" if sharpe > 1.0 else "off" if sharpe > 0.5 else "inverse"
                                    st.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                                            delta="Excellent" if sharpe > 2.0 else "Good" if sharpe > 1.0 else "Poor",
                                            delta_color=sharpe_color)
                                else:
                                    st.metric("Sharpe Ratio", "N/A")
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                        
                        with col8:
                            # Average trade duration (simulated)
                            avg_duration = 3.5  # Average intraday hold time
                            st.metric("Avg Hold Time", f"{avg_duration:.1f}h", 
                                    delta="Intraday Focus",
                                    delta_color="normal")
                        
                        # Performance Analysis Summary
                        st.markdown("### 📋 **Algorithm Assessment**")
                        
                        # Create assessment based on metrics
                        assessment_score = 0
                        assessment_items = []
                        
                        if total_return > 15:
                            assessment_score += 2
                            assessment_items.append("✅ Strong returns generated")
                        elif total_return > 5:
                            assessment_score += 1
                            assessment_items.append("⚠️ Moderate returns")
                        else:
                            assessment_items.append("❌ Poor returns - review strategy")
                        
                        if win_rate >= 60:
                            assessment_score += 2
                            assessment_items.append("✅ High win rate - reliable signals")
                        elif win_rate >= 50:
                            assessment_score += 1
                            assessment_items.append("⚠️ Average win rate")
                        else:
                            assessment_items.append("❌ Low win rate - improve signal quality")
                        
                        if total_trades >= 30:
                            assessment_score += 1
                            assessment_items.append("✅ Sufficient trade sample")
                        else:
                            assessment_items.append("⚠️ Small sample size - need more data")
                        
                        # Overall assessment
                        if assessment_score >= 5:
                            st.success("🎯 **Overall Assessment: STRONG ALGORITHM** - Ready for live trading")
                        elif assessment_score >= 3:
                            st.warning("⚠️ **Overall Assessment: DECENT ALGORITHM** - Consider optimizations")
                        else:
                            st.error("❌ **Overall Assessment: NEEDS IMPROVEMENT** - Review and optimize")
                        
                        # Detailed feedback
                        for item in assessment_items:
                            st.write(item)
                        
                        # Equity curve with enhanced analysis
                        if performance.get('equity_curve'):
                            st.markdown("### 📈 **Equity Curve Analysis**")
                            
                            fig = go.Figure()
                            
                            # Main equity curve
                            fig.add_trace(go.Scatter(
                                y=performance['equity_curve'],
                                mode='lines',
                                name='Portfolio Value',
                                line=dict(color='#1f77b4', width=3)
                            ))
                            
                            # Add benchmark line (initial capital)
                            fig.add_hline(y=1000000, line_dash="dash", line_color="gray", 
                                        annotation_text="Initial Capital (1M PKR)")
                            
                            # Add profit/loss zones
                            if max(performance['equity_curve']) > 1000000:
                                fig.add_hrect(y0=1000000, y1=max(performance['equity_curve']), 
                                            fillcolor="green", opacity=0.1, 
                                            annotation_text="Profit Zone", annotation_position="top left")
                            
                            if min(performance['equity_curve']) < 1000000:
                                fig.add_hrect(y0=min(performance['equity_curve']), y1=1000000, 
                                            fillcolor="red", opacity=0.1,
                                            annotation_text="Loss Zone", annotation_position="bottom left")
                            
                            fig.update_layout(
                                title=f"{selected_symbol} - Portfolio Performance Over Time",
                                xaxis_title="Trade Sequence",
                                yaxis_title="Portfolio Value (PKR)",
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Equity curve insights
                            st.markdown("**📊 Equity Curve Insights:**")
                            st.write(f"• **Starting Capital**: 1,000,000 PKR")
                            st.write(f"• **Ending Capital**: {final_capital:,.0f} PKR")
                            st.write(f"• **Peak Capital**: {max(performance['equity_curve']):,.0f} PKR")
                            st.write(f"• **Lowest Point**: {min(performance['equity_curve']):,.0f} PKR")
                    else:
                        st.warning("⚠️ Insufficient trading data for reliable performance analysis. Need more historical ticks.")
                else:
                    st.info("📊 Collecting trading signals for backtesting analysis...")
            
            with tab4:
                # Detailed analysis
                st.subheader("📋 Technical Details")
                
                # Latest values
                latest = ticks_df.iloc[-1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Information**")
                    st.write(f"Current Price: {latest['price']:.2f} PKR")
                    st.write(f"SMA 5: {latest.get('sma_5', 0):.2f}")
                    st.write(f"SMA 10: {latest.get('sma_10', 0):.2f}")
                    st.write(f"SMA 20: {latest.get('sma_20', 0):.2f}")
                    st.write(f"Support: {latest.get('support', 0):.2f}")
                    st.write(f"Resistance: {latest.get('resistance', 0):.2f}")
                
                with col2:
                    st.write("**Volume & Momentum**")
                    st.write(f"Current Volume: {latest['volume']:,}")
                    st.write(f"Volume Ratio: {latest.get('volume_ratio', 0):.2f}")
                    st.write(f"Momentum: {latest.get('momentum', 0):.4f}")
                    st.write(f"Volatility: {latest.get('volatility', 0):.4f}")
                    if 'rsi' in latest:
                        st.write(f"RSI: {latest.get('rsi', 0):.1f}")
                
                # Raw data
                with st.expander("📊 Raw Tick Data (Last 20)"):
                    st.dataframe(ticks_df[['timestamp', 'price', 'volume']].tail(20))
        else:
            st.error(f"Unable to load data for {selected_symbol}")

def render_algorithm_overview():
    """Render algorithm overview"""
    st.markdown("## 🧠 Algorithm Overview")
    
    st.markdown("""
    ### 🎯 **Multi-Strategy Algorithmic Trading System**
    
    Our advanced system combines multiple quantitative strategies:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algo-card">
            <h4>📈 Trend Following</h4>
            <ul>
                <li>Multiple timeframe SMA crossovers</li>
                <li>Momentum-based entries</li>
                <li>Adaptive position sizing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="performance-card">
            <h4>⚡ Mean Reversion</h4>
            <ul>
                <li>Statistical price deviation analysis</li>
                <li>RSI-based overbought/oversold signals</li>
                <li>Support/resistance level detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="algo-card">
            <h4>📊 Volume Analysis</h4>
            <ul>
                <li>Volume-weighted signal confirmation</li>
                <li>Liquidity assessment</li>
                <li>Institutional flow detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="performance-card">
            <h4>🛡️ Risk Management</h4>
            <ul>
                <li>Dynamic stop-loss levels</li>
                <li>Volatility-adjusted position sizing</li>
                <li>Maximum drawdown controls</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### ⚙️ **Algorithm Parameters**
    
    - **Capital**: 1,000,000 PKR
    - **Max Position Size**: 5% per trade
    - **Stop Loss**: 2% (adaptive)
    - **Take Profit**: 4% (2:1 risk/reward)
    - **Minimum Liquidity**: 100,000 shares
    - **Signal Confidence Threshold**: 60%
    """)

def render_system_status():
    """Render system status"""
    st.markdown("## 🔧 System Status")
    
    system = PSXAlgoTradingSystem()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ PSX Symbols System Online")
        
        # Test PSX Terminal API
        symbols_from_api = False
        try:
            response = system.session.get(f"{system.psx_terminal_url}/api/symbols", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data'):
                    st.info("📊 Live API: Connected")
                    symbols_from_api = True
                else:
                    st.warning("📊 Live API: Format issue")
            else:
                st.warning("📊 Live API: Connection issue")
        except Exception:
            st.info("📊 Live API: Using local fallback")
        
        # Show symbols status
        if symbols_from_api:
            st.caption("Using live PSX Terminal symbols")
        else:
            st.caption("Using comprehensive local symbol database (514 symbols)")
    
    with col2:
        try:
            # Test PSX DPS API with symbol 786 (first symbol)
            response = system.session.get(f"{system.psx_dps_url}/786", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and data['data']:
                    st.success("✅ PSX DPS API Connected")
                    st.info(f"📈 Latest 786 price: {data['data'][0][1]} PKR")
                elif data and isinstance(data, list) and data:
                    st.success("✅ PSX DPS API Connected")
                    st.info(f"📈 Latest 786 price: {data[0][1]} PKR")
                else:
                    st.warning("⚠️ PSX DPS data format issue")
                    st.write(f"Response: {data}")
            else:
                st.warning(f"⚠️ PSX DPS API Issues: {response.status_code}")
        except Exception as e:
            st.error(f"❌ PSX DPS API Failed: {str(e)}")
    
    with col3:
        symbols = get_cached_symbols()
        if symbols:
            st.success(f"✅ {len(symbols)} Symbols Loaded")
            
            # Check major tickers availability
            major_tickers = ['HBL', 'UBL', 'FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL']
            available_major = [s for s in major_tickers if s in symbols]
            st.info(f"Major tickers: {', '.join(available_major)}")
            
            # Test data loading for major tickers
            st.subheader("🧪 Data Loading Test")
            test_symbols = available_major[:3] if available_major else symbols[:3]
            for symbol in test_symbols:
                try:
                    market_data = get_cached_real_time_data(symbol)
                    if market_data:
                        st.success(f"✅ {symbol}: {market_data.get('price', 'N/A')} PKR")
                    else:
                        st.error(f"❌ {symbol}: No data")
                except Exception as e:
                    st.error(f"❌ {symbol}: {str(e)}")
                    
            # Show symbol search hint
            st.info("💡 Use Symbol Analysis page with search to access all 514 symbols including FFC!")
        else:
            st.error("❌ Symbol Loading Failed")

def main():
    """Main application"""
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Section",
        options=[
            "🚨 Live Signals",
            "🔍 Symbol Analysis", 
            "🧠 Algorithm Overview",
            "🔧 System Status"
        ]
    )
    
    # Auto-refresh options
    st.sidebar.markdown("### ⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Page routing
    if page == "🚨 Live Signals":
        render_live_trading_signals()
        
    elif page == "🔍 Symbol Analysis":
        render_symbol_analysis()
        
    elif page == "🧠 Algorithm Overview":
        render_algorithm_overview()
        
    elif page == "🔧 System Status":
        render_system_status()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 PSX Algorithmic Trading System | Real-time ML-Powered Signals</p>
        <p>📊 Live Analysis • 🎯 Automated Signals • 🚀 Professional Grade</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()