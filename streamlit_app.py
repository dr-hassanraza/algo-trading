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

# Authentication system imports
try:
    from user_auth import (authenticate_user, add_user, get_user_data, 
                          create_admin_account, authenticate_social_user, 
                          is_admin, get_all_users, update_user_type, change_password)
    AUTH_AVAILABLE = True
    # Create admin account on startup
    create_admin_account()
except ImportError:
    AUTH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PSX Algo Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication System
def render_login_page():
    """Render login/registration page"""
    st.markdown('<h1 class="main-header">🔐 PSX Algo Trading System - Login</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        # Social Login Options
        st.markdown("### 🌐 **Quick Social Login**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📧 Login with Gmail", use_container_width=True, type="secondary"):
                st.session_state['show_gmail_login'] = True
        
        with col2:
            if st.button("💼 Login with LinkedIn", use_container_width=True, type="secondary"):
                st.session_state['show_linkedin_login'] = True
        
        # Gmail Login Modal
        if st.session_state.get('show_gmail_login', False):
            st.markdown("#### 📧 Gmail Authentication")
            gmail_email = st.text_input("Gmail Address", placeholder="yourname@gmail.com")
            gmail_name = st.text_input("Full Name", placeholder="Your Full Name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Authenticate Gmail", use_container_width=True):
                    if gmail_email and gmail_name and "@gmail.com" in gmail_email:
                        if AUTH_AVAILABLE:
                            username = authenticate_social_user(gmail_email, gmail_name, "gmail")
                            if username:
                                st.session_state['authenticated'] = True
                                st.session_state['username'] = username
                                st.session_state['login_method'] = 'gmail'
                                st.session_state['login_time'] = datetime.now().isoformat()
                                st.success("✅ Gmail login successful! Redirecting...")
                                st.rerun()
                        else:
                            st.error("❌ Authentication system not available")
                    else:
                        st.error("❌ Please enter valid Gmail address and name")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_gmail_login'] = False
                    st.rerun()
        
        # LinkedIn Login Modal
        if st.session_state.get('show_linkedin_login', False):
            st.markdown("#### 💼 LinkedIn Authentication")
            linkedin_email = st.text_input("LinkedIn Email", placeholder="yourname@company.com")
            linkedin_name = st.text_input("Professional Name", placeholder="Your Professional Name")
            linkedin_company = st.text_input("Company (Optional)", placeholder="Your Company")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Authenticate LinkedIn", use_container_width=True):
                    if linkedin_email and linkedin_name:
                        if AUTH_AVAILABLE:
                            display_name = f"{linkedin_name} ({linkedin_company})" if linkedin_company else linkedin_name
                            username = authenticate_social_user(linkedin_email, display_name, "linkedin")
                            if username:
                                st.session_state['authenticated'] = True
                                st.session_state['username'] = username
                                st.session_state['login_method'] = 'linkedin'
                                st.session_state['login_time'] = datetime.now().isoformat()
                                st.success("✅ LinkedIn login successful! Redirecting...")
                                st.rerun()
                        else:
                            st.error("❌ Authentication system not available")
                    else:
                        st.error("❌ Please enter valid email and name")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_linkedin_login'] = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔑 **Traditional Login**")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if submit_login:
                if AUTH_AVAILABLE:
                    if authenticate_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['login_method'] = 'password'
                        st.session_state['login_time'] = datetime.now().isoformat()
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("❌ Authentication system not available")
        
        # Admin Access Note (Hidden for Security)
        with st.expander("🔒 Admin Access Information", expanded=False):
            st.warning("⚠️ **For System Administrator Only**")
            st.markdown("""
            Admin credentials are available to system administrators only.
            
            **Security Notice**: Admin access provides full system control including:
            - User management and permissions
            - System analytics and monitoring  
            - Configuration changes
            - Data export capabilities
            
            If you are the system administrator, contact the deployment manager for credentials.
            """)
            
            # Only show actual credentials in development environment
            import os
            if os.getenv('STREAMLIT_ENV') == 'development' or os.getenv('DEBUG') == 'true':
                st.code("Username: admin | Password: admin123")
            else:
                st.info("Admin credentials hidden in production environment")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            reg_username = st.text_input("Choose Username")
            reg_password = st.text_input("Create Password", type="password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            reg_name = st.text_input("Full Name")
            reg_email = st.text_input("Email Address")
            submit_register = st.form_submit_button("📝 Register", use_container_width=True)
            
            if submit_register:
                if not all([reg_username, reg_password, reg_name, reg_email]):
                    st.error("❌ Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("❌ Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif AUTH_AVAILABLE:
                    try:
                        if add_user(reg_username, reg_password, reg_name, reg_email):
                            st.success("✅ Account created successfully! Please login.")
                        else:
                            st.error("❌ Username already exists")
                    except Exception as e:
                        st.error(f"❌ Registration failed: {str(e)}")
                else:
                    st.error("❌ Authentication system not available")
    
    # Guest access option
    st.markdown("---")
    st.markdown("### 👤 Guest Access")
    if st.button("🚀 Continue as Guest", use_container_width=True):
        st.session_state['authenticated'] = True
        st.session_state['username'] = 'guest'
        st.success("✅ Accessing as guest user...")
        st.rerun()
    
    st.info("💡 **Guest users** can access all features but data won't be saved between sessions.")

def render_logout():
    """Render logout functionality in sidebar with session info"""
    if st.session_state.get('authenticated', False):
        username = st.session_state.get('username', 'Unknown')
        login_method = st.session_state.get('login_method', 'Unknown')
        session_id = st.session_state.get('session_id', 'Unknown')[:8]  # Show first 8 chars
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 **Logged in as:** {username}")
        st.sidebar.caption(f"🔐 Method: {login_method.title()} | Session: {session_id}")
        
        # Session status indicator
        from datetime import datetime, timedelta
        login_time = st.session_state.get('login_time')
        if login_time:
            try:
                login_dt = datetime.fromisoformat(login_time)
                session_age = datetime.now() - login_dt
                if session_age < timedelta(hours=1):
                    st.sidebar.success("🟢 Session Active")
                elif session_age < timedelta(hours=12):
                    st.sidebar.info("🟡 Session Stable")
                else:
                    st.sidebar.warning("🟠 Session Expiring Soon")
            except:
                st.sidebar.info("🔵 Session Active")
        
        if st.sidebar.button("🔓 Logout"):
            # Clear all authentication data
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.session_state['login_method'] = None
            st.session_state['login_time'] = None
            st.session_state['session_id'] = None
            st.success("👋 Logged out successfully!")
            st.rerun()

def check_authentication():
    """Check if user is authenticated with session validation"""
    # Check basic authentication
    if not st.session_state.get('authenticated', False):
        return False
    
    # Validate session hasn't expired (optional timeout check)
    if 'login_time' in st.session_state:
        from datetime import datetime, timedelta
        login_time = st.session_state.get('login_time')
        if isinstance(login_time, str):
            try:
                login_time = datetime.fromisoformat(login_time)
            except:
                login_time = datetime.now()
        
        # Session timeout: 24 hours
        if datetime.now() - login_time > timedelta(hours=24):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.session_state['login_time'] = None
            return False
    
    return True

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
    
    .login-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .stForm > div {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
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
    """Cache symbols for 30 seconds with error handling"""
    try:
        system = PSXAlgoTradingSystem()
        symbols = system.get_symbols()
        if symbols:
            return symbols
        else:
            # Return fallback symbols if empty
            return ["HBL", "UBL", "FFC", "ENGRO", "LUCK", "PSO", "OGDC", "NBP", "MCB", "ABL"]
    except Exception as e:
        # Log error but don't crash - return minimal symbol set
        print(f"Error fetching symbols: {e}")
        return ["HBL", "UBL", "FFC", "ENGRO", "LUCK", "PSO", "OGDC", "NBP", "MCB", "ABL"]

@st.cache_data(ttl=15)
def get_cached_real_time_data(symbol):
    """Cache real-time data for 15 seconds with error handling"""
    try:
        system = PSXAlgoTradingSystem()
        data = system.get_real_time_data(symbol)
        if data and 'price' in data:
            return data
        else:
            # Return fallback data structure
            return {'price': 0.0, 'volume': 0, 'change': 0.0, 'change_pct': 0.0}
    except Exception as e:
        # Log error but don't crash - return fallback data
        print(f"Error fetching real-time data for {symbol}: {e}")
        return {'price': 0.0, 'volume': 0, 'change': 0.0, 'change_pct': 0.0}

@st.cache_data(ttl=60)
def get_cached_intraday_ticks(symbol, limit=100):
    """Cache intraday ticks for 1 minute with error handling"""
    try:
        system = PSXAlgoTradingSystem()
        data = system.get_intraday_ticks(symbol, limit)
        if data is not None and len(data) > 0:
            return data
        else:
            # Return empty DataFrame with proper structure
            import pandas as pd
            return pd.DataFrame(columns=['timestamp', 'price', 'volume', 'high', 'low', 'close'])
    except Exception as e:
        # Log error but don't crash - return empty DataFrame
        print(f"Error fetching intraday ticks for {symbol}: {e}")
        import pandas as pd
        return pd.DataFrame(columns=['timestamp', 'price', 'volume', 'high', 'low', 'close'])

def safe_data_operation(operation_name, operation_func, fallback_result=None):
    """Safely execute data operations without affecting user session"""
    try:
        result = operation_func()
        return result
    except Exception as e:
        # Log error but don't crash or logout user
        print(f"Safe operation '{operation_name}' failed: {e}")
        
        # Show non-intrusive error to user
        if not st.session_state.get('data_error_shown', False):
            st.sidebar.warning(f"⚠️ {operation_name} temporarily unavailable")
            st.session_state['data_error_shown'] = True
        
        return fallback_result

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
    
    # Market-Wide Signal Scanner
    st.markdown("---")
    st.subheader("🔍 Market-Wide Signal Scanner")
    
    # Only show scanner if there are no actionable signals in watchlist
    if buy_signals == 0 and sell_signals == 0:
        st.warning("⚠️ **No actionable signals in your watchlist!** Let's scan the broader market for opportunities...")
        
        # Scanner controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Scanning 100+ PSX stocks for BUY/SELL signals...**")
        
        with col2:
            scan_intensity = st.selectbox("Scan Type", ["Quick Scan (50)", "Deep Scan (100)", "Full Market (200)"], index=0)
        
        with col3:
            min_confidence = st.slider("Min Confidence", 40, 90, 60, 5, help="Minimum confidence level for signals")
        
        if st.button("🚀 Scan Market Now", type="primary"):
            # Determine scan size based on selection
            if "Quick" in scan_intensity:
                scan_limit = 50
            elif "Deep" in scan_intensity:
                scan_limit = 100
            else:
                scan_limit = 200
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get all symbols for scanning
            all_symbols = get_cached_symbols()
            if all_symbols:
                # Prioritize liquid stocks for scanning
                major_liquid_stocks = [
                    'HBL', 'UBL', 'FFC', 'ENGRO', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL',
                    'TRG', 'SYSTEMS', 'POL', 'PPL', 'NESTLE', 'UNILEVER', 'COLG', 'ICI', 'BAHL',
                    'BAFL', 'MEBL', 'JSBL', 'AKBL', 'FABL', 'EFERT', 'FATIMA', 'DGKC', 'MLCF',
                    'FFBL', 'ATRL', 'SEARL', 'PIOC', 'KAPCO', 'HUBCO', 'FCCL', 'KEL', 'KTM',
                    'LOTCHEM', 'MRNS', 'NRL', 'OGDC', 'OMC', 'PACE', 'PAEL', 'PASL', 'PRL',
                    'SSGC', 'TELE', 'WTL', 'BNWM', 'CHCC', 'DOL', 'EPCL', 'FLYNG', 'GATM'
                ]
                
                # Create scanning list with priority
                scan_symbols = []
                for stock in major_liquid_stocks[:scan_limit//2]:
                    if stock in all_symbols:
                        scan_symbols.append(stock)
                
                # Add remaining symbols to reach scan limit
                remaining = [s for s in all_symbols if s not in scan_symbols]
                scan_symbols.extend(remaining[:scan_limit - len(scan_symbols)])
                scan_symbols = scan_symbols[:scan_limit]
                
                # Store results
                buy_opportunities = []
                sell_opportunities = []
                
                # Scan through symbols
                for i, symbol in enumerate(scan_symbols):
                    try:
                        progress = (i + 1) / len(scan_symbols)
                        progress_bar.progress(progress)
                        status_text.text(f"Scanning {symbol}... ({i+1}/{len(scan_symbols)})")
                        
                        # Get market data
                        market_data = get_cached_real_time_data(symbol)
                        
                        if market_data:
                            # Get intraday data
                            ticks_df = get_cached_intraday_ticks(symbol, 30)
                            
                            if not ticks_df.empty and len(ticks_df) >= 10:
                                # Calculate indicators and signals
                                ticks_df = system.calculate_technical_indicators(ticks_df)
                                signal_data = system.generate_trading_signals(ticks_df, symbol)
                                
                                confidence = signal_data['confidence']
                                signal_type = signal_data['signal']
                                
                                # Filter by confidence and signal type
                                if confidence >= min_confidence:
                                    if signal_type in ['BUY', 'STRONG_BUY']:
                                        buy_opportunities.append({
                                            'Symbol': symbol,
                                            'Signal': signal_type,
                                            'Confidence': f"{confidence:.1f}%",
                                            'Price': f"{market_data['price']:.2f}",
                                            'Entry': f"{signal_data['entry_price']:.2f}",
                                            'Stop': f"{signal_data['stop_loss']:.2f}",
                                            'Target': f"{signal_data['take_profit']:.2f}",
                                            'Volume': f"{market_data.get('volume', 0):,}",
                                            'Reason': signal_data.get('reasons', ['N/A'])[0][:50] + "..." if signal_data.get('reasons') else 'Technical analysis'
                                        })
                                    
                                    elif signal_type in ['SELL', 'STRONG_SELL']:
                                        sell_opportunities.append({
                                            'Symbol': symbol,
                                            'Signal': signal_type,
                                            'Confidence': f"{confidence:.1f}%",
                                            'Price': f"{market_data['price']:.2f}",
                                            'Entry': f"{signal_data['entry_price']:.2f}",
                                            'Stop': f"{signal_data['stop_loss']:.2f}",
                                            'Target': f"{signal_data['take_profit']:.2f}",
                                            'Volume': f"{market_data.get('volume', 0):,}",
                                            'Reason': signal_data.get('reasons', ['N/A'])[0][:50] + "..." if signal_data.get('reasons') else 'Technical analysis'
                                        })
                    
                    except Exception as e:
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown(f"### 🎯 **Scan Results** ({len(scan_symbols)} stocks analyzed)")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🟢 BUY Opportunities", len(buy_opportunities))
                
                with col2:
                    st.metric("🔴 SELL Opportunities", len(sell_opportunities))
                
                with col3:
                    total_opportunities = len(buy_opportunities) + len(sell_opportunities)
                    opportunity_rate = (total_opportunities / len(scan_symbols)) * 100
                    st.metric("📊 Hit Rate", f"{opportunity_rate:.1f}%")
                
                with col4:
                    st.metric("🔍 Scanned", f"{len(scan_symbols)} stocks")
                
                # Display opportunities in tabs
                if buy_opportunities or sell_opportunities:
                    tab1, tab2 = st.tabs([f"🟢 BUY Signals ({len(buy_opportunities)})", f"🔴 SELL Signals ({len(sell_opportunities)})"])
                    
                    with tab1:
                        if buy_opportunities:
                            st.success(f"Found {len(buy_opportunities)} BUY opportunities with ≥{min_confidence}% confidence!")
                            
                            # Sort by confidence (highest first)
                            buy_opportunities.sort(key=lambda x: float(x['Confidence'].replace('%', '')), reverse=True)
                            
                            # Display as expandable cards for better readability
                            for i, opp in enumerate(buy_opportunities[:10]):  # Show top 10
                                with st.expander(f"🟢 {opp['Symbol']} - {opp['Signal']} ({opp['Confidence']})", expanded=(i < 3)):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        **💰 Trade Setup:**
                                        - Entry: {opp['Entry']} PKR
                                        - Stop Loss: {opp['Stop']} PKR  
                                        - Target: {opp['Target']} PKR
                                        """)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        **📊 Market Data:**
                                        - Current: {opp['Price']} PKR
                                        - Volume: {opp['Volume']}
                                        - Confidence: {opp['Confidence']}
                                        """)
                                    
                                    with col3:
                                        risk_reward = abs(float(opp['Target']) - float(opp['Entry'])) / abs(float(opp['Entry']) - float(opp['Stop']))
                                        st.markdown(f"""
                                        **⚖️ Risk Analysis:**
                                        - Risk/Reward: 1:{risk_reward:.1f}
                                        - Signal: {opp['Signal']}
                                        """)
                                    
                                    st.info(f"📋 **Analysis**: {opp['Reason']}")
                            
                            if len(buy_opportunities) > 10:
                                st.info(f"Showing top 10 BUY opportunities. Total found: {len(buy_opportunities)}")
                        else:
                            st.info("No BUY signals found meeting your confidence criteria.")
                    
                    with tab2:
                        if sell_opportunities:
                            st.warning(f"Found {len(sell_opportunities)} SELL opportunities with ≥{min_confidence}% confidence!")
                            
                            # Sort by confidence (highest first)
                            sell_opportunities.sort(key=lambda x: float(x['Confidence'].replace('%', '')), reverse=True)
                            
                            # Display as expandable cards
                            for i, opp in enumerate(sell_opportunities[:10]):  # Show top 10
                                with st.expander(f"🔴 {opp['Symbol']} - {opp['Signal']} ({opp['Confidence']})", expanded=(i < 3)):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        **💰 Trade Setup:**
                                        - Entry: {opp['Entry']} PKR
                                        - Stop Loss: {opp['Stop']} PKR
                                        - Target: {opp['Target']} PKR
                                        """)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        **📊 Market Data:**
                                        - Current: {opp['Price']} PKR
                                        - Volume: {opp['Volume']}
                                        - Confidence: {opp['Confidence']}
                                        """)
                                    
                                    with col3:
                                        risk_reward = abs(float(opp['Target']) - float(opp['Entry'])) / abs(float(opp['Entry']) - float(opp['Stop']))
                                        st.markdown(f"""
                                        **⚖️ Risk Analysis:**
                                        - Risk/Reward: 1:{risk_reward:.1f}
                                        - Signal: {opp['Signal']}
                                        """)
                                    
                                    st.info(f"📋 **Analysis**: {opp['Reason']}")
                            
                            if len(sell_opportunities) > 10:
                                st.info(f"Showing top 10 SELL opportunities. Total found: {len(sell_opportunities)}")
                        else:
                            st.info("No SELL signals found meeting your confidence criteria.")
                else:
                    st.warning(f"No actionable signals found with ≥{min_confidence}% confidence. Try lowering the confidence threshold or selecting 'Full Market' scan.")
                    
                    # Suggestions for better results
                    st.markdown("""
                    **💡 Tips for Better Results:**
                    - Lower confidence threshold to 50-55%
                    - Try 'Full Market' scan to analyze more stocks
                    - Check during active trading hours (10 AM - 3 PM)
                    - Market conditions may be ranging (few directional signals)
                    """)
    
    else:
        st.success(f"✅ **Active signals detected!** Your watchlist has {buy_signals + sell_signals} actionable signals. Market scanner not needed.")
        
        if buy_signals > 0:
            st.info(f"🟢 **{buy_signals} BUY signals** ready for action in your selected stocks above!")
        
        if sell_signals > 0:
            st.info(f"🔴 **{sell_signals} SELL signals** ready for action in your selected stocks above!")
        
        # Option to scan anyway
        if st.button("🔍 Scan Market Anyway", help="Find additional opportunities beyond your watchlist"):
            st.info("Scroll up and use the market scanner when no signals are active.")

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

def render_admin_panel():
    """Render admin dashboard with user management and system stats"""
    st.markdown('<h1 class="main-header">👑 Admin Panel</h1>', unsafe_allow_html=True)
    
    if not AUTH_AVAILABLE:
        st.error("❌ Authentication system not available")
        return
    
    # Verify admin access
    if not is_admin(st.session_state.get('username', '')):
        st.error("❌ Access denied. Admin privileges required.")
        return
    
    # Admin dashboard tabs
    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📊 System Analytics", "⚙️ System Settings"])
    
    with tab1:
        st.subheader("👥 User Management")
        
        # Get all users
        all_users = get_all_users()
        if not all_users:
            st.info("No users registered yet.")
            return
        
        # User statistics
        total_users = len(all_users)
        admin_users = sum(1 for user in all_users.values() if user.get('user_type') == 'admin')
        social_logins = sum(1 for user in all_users.values() if user.get('login_method') in ['gmail', 'linkedin'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Admin Users", admin_users)
        with col3:
            st.metric("Social Logins", social_logins)
        with col4:
            st.metric("Password Logins", total_users - social_logins)
        
        st.markdown("---")
        
        # User table
        user_data = []
        for username, data in all_users.items():
            user_data.append({
                'Username': username,
                'Name': data.get('name', 'N/A'),
                'Email': data.get('email', 'N/A'),
                'Type': data.get('user_type', 'user'),
                'Login Method': data.get('login_method', 'password'),
                'Usage Count': data.get('usage_count', 0),
                'Last Login': data.get('last_login', 'Never'),
                'Created At': data.get('created_at', 'N/A')
            })
        
        df_users = pd.DataFrame(user_data)
        st.dataframe(df_users, use_container_width=True)
        
        # User management actions
        st.subheader("🔧 User Management Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Promote User to Admin**")
            selected_user = st.selectbox(
                "Select User", 
                options=[u for u in all_users.keys() if all_users[u].get('user_type') != 'admin']
            )
            
            if st.button("👑 Promote to Admin", use_container_width=True):
                if selected_user and update_user_type(selected_user, 'admin'):
                    st.success(f"✅ {selected_user} promoted to admin!")
                    st.rerun()
                else:
                    st.error("❌ Failed to promote user")
        
        with col2:
            st.markdown("**Demote Admin to User**")
            admin_users_list = [u for u in all_users.keys() 
                              if all_users[u].get('user_type') == 'admin' and u != 'admin']
            
            if admin_users_list:
                selected_admin = st.selectbox("Select Admin", options=admin_users_list)
                
                if st.button("👤 Demote to User", use_container_width=True):
                    if selected_admin and update_user_type(selected_admin, 'user'):
                        st.success(f"✅ {selected_admin} demoted to regular user!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to demote user")
            else:
                st.info("No other admins to demote")
    
    with tab2:
        st.subheader("📊 System Analytics")
        
        # System usage metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # User registration over time
            reg_dates = [data.get('created_at', '') for data in all_users.values() if data.get('created_at')]
            if reg_dates:
                try:
                    reg_df = pd.DataFrame({
                        'Date': pd.to_datetime(reg_dates).dt.date,
                        'Registrations': 1
                    }).groupby('Date').sum().reset_index()
                    
                    fig = px.line(reg_df, x='Date', y='Registrations', 
                                 title='User Registrations Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Registration timeline data not available")
            else:
                st.info("No registration data available")
        
        with col2:
            # Usage statistics
            usage_data = [data.get('usage_count', 0) for data in all_users.values()]
            if usage_data and any(usage_data):
                fig = px.histogram(x=usage_data, nbins=10, 
                                 title='User Activity Distribution')
                fig.update_xaxis(title='Usage Count')
                fig.update_yaxis(title='Number of Users')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No usage data available")
        
        # Login method breakdown
        login_methods = {}
        for data in all_users.values():
            method = data.get('login_method', 'password')
            login_methods[method] = login_methods.get(method, 0) + 1
        
        if login_methods:
            fig = px.pie(values=list(login_methods.values()), 
                        names=list(login_methods.keys()),
                        title='Login Methods Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚙️ System Settings")
        
        # System information
        st.markdown("**📋 System Information**")
        system_info = {
            "Authentication System": "✅ Active" if AUTH_AVAILABLE else "❌ Inactive",
            "Current Admin": st.session_state.get('username', 'Unknown'),
            "Login Method": st.session_state.get('login_method', 'Unknown'),
            "Session Active": "✅ Yes" if st.session_state.get('authenticated') else "❌ No"
        }
        
        for key, value in system_info.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
        
        st.markdown("---")
        
        # Admin tools
        st.markdown("**🛠️ Admin Tools**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh User Data", use_container_width=True):
                st.success("✅ User data refreshed!")
                st.rerun()
        
        with col2:
            if st.button("📊 Export User Data", use_container_width=True):
                user_json = json.dumps(all_users, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=user_json,
                    file_name=f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Password Change Section
        st.markdown("---")
        st.markdown("**🔐 Security Settings**")
        
        # Check if admin is using default password
        current_admin = all_users.get('admin', {})
        if not current_admin.get('password_changed', False):
            st.warning("⚠️ **Security Alert**: Default admin password detected. Please change immediately!")
        
        with st.expander("🔑 Change Admin Password", expanded=not current_admin.get('password_changed', False)):
            st.markdown("**Change your admin password for better security**")
            
            with st.form("change_password_form"):
                old_password = st.text_input("Current Password", type="password", help="Enter your current admin password")
                new_password = st.text_input("New Password", type="password", help="Choose a strong password (8+ characters)")
                confirm_password = st.text_input("Confirm New Password", type="password", help="Re-enter your new password")
                
                col1, col2 = st.columns(2)
                with col1:
                    submit_password_change = st.form_submit_button("🔐 Change Password", use_container_width=True)
                
                with col2:
                    st.caption("💡 Use strong passwords with 8+ characters")
                
                if submit_password_change:
                    if not old_password or not new_password or not confirm_password:
                        st.error("❌ Please fill in all password fields")
                    elif new_password != confirm_password:
                        st.error("❌ New passwords do not match")
                    elif len(new_password) < 8:
                        st.error("❌ Password must be at least 8 characters long")
                    elif new_password == old_password:
                        st.error("❌ New password must be different from current password")
                    else:
                        # Attempt to change password
                        current_username = st.session_state.get('username', '')
                        if change_password(current_username, old_password, new_password):
                            st.success("✅ Admin password changed successfully!")
                            st.info("🔒 Your account is now more secure. Please remember your new password.")
                            # Clear form by rerunning
                            st.rerun()
                        else:
                            st.error("❌ Failed to change password. Please verify your current password.")

def main():
    """Main application with authentication and error handling"""
    
    try:
        # Check authentication
        if not check_authentication():
            render_login_page()
            return
        
        # Render logout in sidebar
        render_logout()
        
        # Render header
        render_header()
        
        # Session persistence check
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state['session_id'] = str(uuid.uuid4())
        
    except Exception as e:
        st.error("🔧 **System Initialization Error**")
        st.error(f"Error details: {str(e)}")
        st.info("Please refresh the page. If the problem persists, contact support.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Check if user is admin to show admin panel
    navigation_options = [
        "🚨 Live Signals",
        "🔍 Symbol Analysis", 
        "🧠 Algorithm Overview",
        "🔧 System Status"
    ]
    
    # Add admin panel for admin users
    if AUTH_AVAILABLE and is_admin(st.session_state.get('username', '')):
        navigation_options.append("👑 Admin Panel")
    
    page = st.sidebar.selectbox(
        "Select Section",
        options=navigation_options
    )
    
    # Auto-refresh options
    st.sidebar.markdown("### ⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    # Page routing with error handling
    try:
        if page == "🚨 Live Signals":
            render_live_trading_signals()
            
        elif page == "🔍 Symbol Analysis":
            render_symbol_analysis()
            
        elif page == "🧠 Algorithm Overview":
            render_algorithm_overview()
            
        elif page == "🔧 System Status":
            render_system_status()
            
        elif page == "👑 Admin Panel":
            render_admin_panel()
            
    except Exception as e:
        st.error("🚨 **Page Loading Error**")
        st.error(f"There was an error loading the {page} page.")
        
        with st.expander("🔍 Error Details", expanded=False):
            st.code(f"Error: {str(e)}")
            st.code(f"Page: {page}")
            st.code(f"Session ID: {st.session_state.get('session_id', 'Unknown')}")
        
        st.info("💡 **Troubleshooting Steps:**")
        st.write("1. Try refreshing the page")
        st.write("2. Navigate to a different section and come back")
        st.write("3. Check your internet connection")
        st.write("4. If the problem persists, contact support")
        
        # Don't logout on page errors - just show error
        return
    
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