import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

ML_AVAILABLE = True

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
            
            # MACD Indicator
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['price'].rolling(window=bb_period).mean()
            bb_rolling_std = df['price'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ADX (Average Directional Index) - Simplified Version
            high_low = df['price'].rolling(2).max() - df['price'].rolling(2).min()
            high_close = abs(df['price'] - df['price'].shift(1))
            low_close = abs(df['price'].shift(1) - df['price'])
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Directional Movement
            plus_dm = df['price'].diff().where(df['price'].diff() > 0, 0)
            minus_dm = (-df['price'].diff()).where(df['price'].diff() < 0, 0)
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / df['atr'])
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / df['atr'])
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            return df
        except Exception as e:
            st.error(f"Technical indicators error: {str(e)}")
            return df
    
    def generate_trading_signals(self, df, symbol):
        """🚀 ENHANCED PROFESSIONAL SIGNAL GENERATION - Now calls ML-enhanced version"""
        return self.generate_ml_enhanced_trading_signals(df, symbol)

    # =================== NEW ML & FEATURE ENHANCEMENT METHODS ===================

    def _prepare_ml_features(self, df):
        """Enhanced feature engineering for ML model."""
        if df.empty or len(df) < 50:
            return None, None, None
            
        try:
            # Core technical features
            features = [
                'sma_5', 'sma_10', 'sma_20', 'volume_ratio', 'momentum',
                'volatility', 'rsi', 'macd_histogram', 'bb_position', 'adx',
                'obv', 'vwap', 'plus_di', 'minus_di', 'atr'
            ]
            
            # Add advanced features
            df = self._add_advanced_features(df)
            
            # Extended feature set
            extended_features = features + [
                'rsi_divergence', 'price_momentum_3', 'price_momentum_10',
                'volume_price_trend', 'stochastic_k', 'stochastic_d',
                'williams_r', 'commodity_channel_index', 'price_rate_change',
                'volume_oscillator', 'regime_score'
            ]
            
            # Check feature availability
            available_features = [f for f in extended_features if f in df.columns]
            if len(available_features) < 10:
                return None, None, None

            df_ml = df[available_features].copy()
            
            # Enhanced target variable with multiple horizons
            horizon_short = 3
            horizon_medium = 5
            horizon_long = 10
            threshold = 0.008  # 0.8% threshold for better signal quality
            
            # Multi-horizon target
            df_ml['future_price_short'] = df['price'].shift(-horizon_short)
            df_ml['future_price_medium'] = df['price'].shift(-horizon_medium)
            df_ml['future_price_long'] = df['price'].shift(-horizon_long)
            
            # Weighted target considering multiple horizons
            short_signal = (df_ml['future_price_short'] > df['price'] * (1 + threshold)).astype(int)
            medium_signal = (df_ml['future_price_medium'] > df['price'] * (1 + threshold)).astype(int)
            long_signal = (df_ml['future_price_long'] > df['price'] * (1 + threshold)).astype(int)
            
            # Composite target with higher weight on medium-term
            df_ml['target'] = ((short_signal * 0.3 + medium_signal * 0.5 + long_signal * 0.2) >= 0.5).astype(int)
            
            df_ml = df_ml.dropna()
            
            if df_ml.empty or len(df_ml) < 30:
                return None, None, None
                
            X = df_ml[available_features]
            y = df_ml['target']
            
            # Ensure balanced classes
            if len(y.unique()) < 2 or min(y.value_counts()) < 5:
                return None, None, None
            
            return X, y, available_features
            
        except Exception as e:
            st.error(f"Feature preparation error: {str(e)}")
            return None, None, None
    
    def _add_advanced_features(self, df):
        """Add advanced technical indicators for enhanced ML performance."""
        try:
            # RSI Divergence
            df['rsi_change'] = df['rsi'].diff()
            df['price_change_pct'] = df['price'].pct_change()
            df['rsi_divergence'] = np.where(
                (df['rsi_change'] > 0) & (df['price_change_pct'] < 0), 1,
                np.where((df['rsi_change'] < 0) & (df['price_change_pct'] > 0), -1, 0)
            )
            
            # Multi-period momentum
            df['price_momentum_3'] = df['price'].pct_change(3)
            df['price_momentum_10'] = df['price'].pct_change(10)
            
            # Volume-Price Trend
            df['volume_price_trend'] = df['volume'] * df['price'].pct_change()
            
            # Stochastic Oscillator
            low_min = df['price'].rolling(window=14).min()
            high_max = df['price'].rolling(window=14).max()
            df['stochastic_k'] = 100 * (df['price'] - low_min) / (high_max - low_min)
            df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_max - df['price']) / (high_max - low_min)
            
            # Commodity Channel Index (CCI)
            typical_price = df['price']  # Using price as proxy for typical price
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['commodity_channel_index'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Price Rate of Change
            df['price_rate_change'] = df['price'].pct_change(10) * 100
            
            # Volume Oscillator
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_oscillator'] = ((df['volume_sma_5'] - df['volume_sma_10']) / df['volume_sma_10']) * 100
            
            # Market Regime Score
            trend_score = np.where(df['sma_5'] > df['sma_20'], 1, -1)
            momentum_score = np.where(df['rsi'] > 50, 1, -1)
            volume_score = np.where(df['volume_ratio'] > 1, 1, -1)
            df['regime_score'] = (trend_score + momentum_score + volume_score) / 3
            
            return df
            
        except Exception as e:
            return df

    def get_fundamental_data(self, symbol):
        """Placeholder for fetching fundamental data."""
        # In a real implementation, you would use an API like FMP or EODHD here
        # This is a placeholder returning static data.
        return {
            'pe_ratio': 15.0,
            'eps_growth': 0.05,
            'is_profitable': True
        }

    def get_market_regime(self, df):
        """Enhanced market regime detection with multiple indicators."""
        try:
            if df.empty or len(df) < 20:
                return {"regime": "Unknown", "confidence": 0, "details": {}}
                
            latest = df.iloc[-1]
            
            # ADX-based trend strength
            adx = latest.get('adx', 20)
            adx_score = 0
            if adx > 30:
                adx_score = 2  # Strong trend
            elif adx > 25:
                adx_score = 1  # Moderate trend
            elif adx < 15:
                adx_score = -2  # Strong range
            elif adx < 20:
                adx_score = -1  # Moderate range
            
            # Moving average alignment
            sma_5 = latest.get('sma_5', 0)
            sma_10 = latest.get('sma_10', 0) 
            sma_20 = latest.get('sma_20', 0)
            
            ma_alignment_score = 0
            if sma_5 > sma_10 > sma_20:  # Perfect bullish alignment
                ma_alignment_score = 2
            elif sma_5 < sma_10 < sma_20:  # Perfect bearish alignment
                ma_alignment_score = 2
            elif abs(sma_5 - sma_20) / sma_20 < 0.01:  # Converged MAs (ranging)
                ma_alignment_score = -2
            
            # Volatility analysis
            volatility = latest.get('volatility', 0.02)
            vol_score = 1 if volatility > 0.025 else -1 if volatility < 0.015 else 0
            
            # Bollinger Band width (trending vs ranging)
            bb_width = latest.get('bb_width', 0.1)
            bb_score = 1 if bb_width > 0.08 else -1 if bb_width < 0.04 else 0
            
            # Price momentum
            momentum = latest.get('momentum', 0)
            momentum_score = 1 if abs(momentum) > 0.005 else -1
            
            # Combine all scores
            total_score = adx_score + ma_alignment_score + vol_score + bb_score + momentum_score
            confidence = min(abs(total_score) * 10, 100)
            
            # Determine regime
            if total_score >= 4:
                regime = "Strong_Trending"
            elif total_score >= 2:
                regime = "Trending"
            elif total_score <= -4:
                regime = "Strong_Ranging"
            elif total_score <= -2:
                regime = "Ranging"
            else:
                regime = "Transitional"
            
            # Additional regime characteristics
            details = {
                "adx_value": adx,
                "volatility": volatility,
                "bb_width": bb_width,
                "ma_alignment": "Bullish" if sma_5 > sma_20 else "Bearish" if sma_5 < sma_20 else "Neutral",
                "momentum": momentum,
                "total_score": total_score
            }
            
            return {
                "regime": regime,
                "confidence": confidence,
                "details": details
            }
            
        except Exception as e:
            return {"regime": "Unknown", "confidence": 0, "details": {}}

    # =================== ENHANCED PROFESSIONAL FEATURES ===================

    # VOLUME_ANALYSIS METHODS

    def calculate_volume_indicators(self, df):
        """Calculate comprehensive volume indicators"""
        try:
            # Volume moving averages
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratio (current vs average)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On-Balance Volume (OBV)
            df['obv'] = 0
            if not df.empty:
                df['obv'] = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()

            # Volume Weighted Average Price (VWAP)
            df['typical_price'] = (df['price'] + df.get('high', df['price']) + df.get('low', df['price'])) / 3
            df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            return df
            
        except Exception as e:
            # st.error(f"Volume indicators error: {str(e)}") # Silenced for cleaner UI
            return df
    
    def analyze_volume_confirmation(self, df):
        """Analyze volume confirmation for signals"""
        if df.empty or len(df) < 20:
            return {"confirmed": False, "strength": 0, "reasons": []}
        
        latest = df.iloc[-1]
        volume_signals = []
        strength_score = 0
        
        # Volume surge analysis
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio >= 2.0:
            volume_signals.append("Strong volume surge (2x+ average)")
            strength_score += 3
        elif volume_ratio >= 1.5:
            volume_signals.append(f"Above average volume ({volume_ratio:.1f}x)")
            strength_score += 2
        
        # OBV trend analysis
        if 'obv' in df.columns and len(df) >= 5:
            obv_trend = df['obv'].diff(5).iloc[-1]
            if obv_trend > 0:
                volume_signals.append("OBV trending up (buying pressure)")
                strength_score += 2
            elif obv_trend < 0:
                volume_signals.append("OBV trending down (selling pressure)")
                strength_score -= 2
        
        # VWAP analysis
        price = latest.get('price', 0)
        vwap = latest.get('vwap', price)
        
        if price > vwap * 1.005:
            volume_signals.append("Price above VWAP")
            strength_score += 1
        elif price < vwap * 0.995:
            volume_signals.append("Price below VWAP")
            strength_score -= 1
        
        return {
            "confirmed": strength_score >= 2,
            "strength": strength_score,
            "reasons": volume_signals,
            "volume_ratio": volume_ratio
        }
    

    # MULTI_TIMEFRAME METHODS

    def analyze_multi_timeframe_signals(self, symbol, current_df):
        """Analyze signals across multiple timeframes"""
        try:
            timeframe_signals = {}
            timeframes = {'5m': 5, '15m': 15, '1h': 60}
            consensus_score = 0
            
            for tf_name, minutes in timeframes.items():
                tf_signal = self.generate_basic_timeframe_signal(current_df, tf_name)
                timeframe_signals[tf_name] = tf_signal
                if tf_signal['signal'] in ['BUY', 'STRONG_BUY']:
                    consensus_score += 1
                elif tf_signal['signal'] in ['SELL', 'STRONG_SELL']:
                    consensus_score -= 1
            
            alignment_pct = abs(consensus_score) / len(timeframes)
            overall_direction = "BULLISH" if consensus_score > 0 else "BEARISH" if consensus_score < 0 else "NEUTRAL"
            
            return {
                "alignment_score": alignment_pct,
                "overall_direction": overall_direction,
                "consensus": alignment_pct >= 0.6,
                "timeframe_signals": timeframe_signals
            }
            
        except Exception as e:
            return {"alignment_score": 0, "overall_direction": "NEUTRAL", "consensus": False}
    
    def generate_basic_timeframe_signal(self, df, timeframe):
        """Generate basic signal for timeframe"""
        if df.empty or len(df) < 10:
            return {"signal": "HOLD", "confidence": 0}
        
        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)
        sma_5 = latest.get('sma_5', latest.get('price', 0))
        sma_20 = latest.get('sma_20', latest.get('price', 0))
        
        score = 0
        if rsi <= 30: score += 2
        elif rsi >= 70: score -= 2
        if sma_5 > sma_20: score += 1
        elif sma_5 < sma_20: score -= 1
        
        if score >= 2:
            return {"signal": "BUY", "confidence": 60 + score * 10}
        elif score <= -2:
            return {"signal": "SELL", "confidence": 60 + abs(score) * 10}
        else:
            return {"signal": "HOLD", "confidence": 20}
    

    # ENHANCED_SIGNALS METHODS

    def generate_ml_enhanced_trading_signals(self, df, symbol):
        """🚀 ADVANCED ML-ENHANCED TRADING SIGNALS - v4.0 with Regime Adaptation"""
        
        if df.empty or len(df) < 50: # Increased requirement for advanced ML
            return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data for advanced ML analysis"}
        
        try:
            # STEP 0: Get Enhanced ML Model & Predictions
            ml_model = get_enhanced_ml_model(symbol, df)
            
            # Calculate all indicators
            df = self.calculate_technical_indicators(df)
            df = self.calculate_volume_indicators(df)
            
            latest = df.iloc[-1]
            
            # ML PREDICTION COMPONENT (40% weight - increased)
            ml_score = 0
            ml_confidence = 0
            ml_prediction_text = "ML: Not Available"
            feature_importance = {}

            if ml_model and ML_AVAILABLE:
                _, _, features = self._prepare_ml_features(df)
                if features:
                    try:
                        # Get feature values for latest data point
                        feature_values = latest[features].values.reshape(1, -1)
                        
                        # ML Prediction with probability
                        prediction_proba = ml_model.predict_proba(feature_values)[0]
                        ml_pred = ml_model.classes_[np.argmax(prediction_proba)]
                        
                        # Enhanced ML scoring with confidence consideration
                        max_proba = max(prediction_proba)
                        if ml_pred == 1 and max_proba > 0.55: # Bullish prediction
                            ml_score = 8 * prediction_proba[1] * max_proba
                            ml_confidence = prediction_proba[1] * 100
                            ml_prediction_text = f"ML: BULLISH ({ml_confidence:.0f}%)"
                        elif ml_pred == 0 and max_proba > 0.55: # Bearish prediction
                            ml_score = -8 * prediction_proba[0] * max_proba
                            ml_confidence = prediction_proba[0] * 100
                            ml_prediction_text = f"ML: BEARISH ({ml_confidence:.0f}%)"
                        else: # Low confidence prediction
                            ml_score = 0
                            ml_confidence = 50
                            ml_prediction_text = f"ML: NEUTRAL ({max_proba*100:.0f}%)"
                        
                        # Get feature importance for interpretability
                        if hasattr(ml_model, 'feature_importances_'):
                            importance_dict = dict(zip(features, ml_model.feature_importances_))
                            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                            feature_importance = {f: round(imp, 3) for f, imp in top_features}
                        
                    except Exception as e:
                        ml_score, ml_confidence = 0, 50
                        ml_prediction_text = "ML: Error in prediction"

            # STEP 1: Market Regime Analysis (20% weight - enhanced)
            regime_analysis = self.get_market_regime(df)
            regime_type = regime_analysis['regime']
            regime_confidence = regime_analysis['confidence']
            regime_details = regime_analysis['details']
            
            # STEP 2: Regime-Adaptive Traditional Analysis (20% weight)
            traditional_score, traditional_confidence, traditional_reasons = self.analyze_regime_adaptive_signals(df, regime_type)
            
            # STEP 3: Volume Analysis (15% weight)  
            volume_analysis = self.analyze_volume_confirmation(df)
            volume_score = volume_analysis['strength']
            
            # STEP 4: Multi-Timeframe Analysis (5% weight - reduced)
            mtf_analysis = self.analyze_multi_timeframe_signals(symbol, df)
            mtf_score = mtf_analysis['alignment_score']
            mtf_direction = mtf_analysis['overall_direction']
            
            # COMBINE ALL SCORES WITH REGIME ADAPTATION
            total_score = 0
            all_reasons = []
            confidence_components = []

            # ML Model (40%)
            ml_weight = 0.40
            total_score += ml_score * ml_weight
            all_reasons.append(ml_prediction_text)
            confidence_components.append(ml_confidence * ml_weight)

            # Traditional signals (20%) - regime adapted
            trad_weight = 0.20
            total_score += traditional_score * trad_weight
            all_reasons.extend([f"Tech: {r}" for r in traditional_reasons[:2]])
            confidence_components.append(traditional_confidence * trad_weight)
            
            # Volume (15%)
            vol_weight = 0.15
            total_score += volume_score * vol_weight
            if volume_analysis['confirmed']:
                all_reasons.extend([f"Vol: {r}" for r in volume_analysis.get('reasons', [])[:1]])
            confidence_components.append(abs(volume_score) * 10 * vol_weight)
            
            # Market Regime (20%)
            regime_weight = 0.20
            regime_score = self.calculate_regime_score(regime_type, regime_details)
            total_score += regime_score * regime_weight
            all_reasons.append(f"Regime: {regime_type} ({regime_confidence:.0f}%)")
            confidence_components.append(regime_confidence * regime_weight / 100 * 100)
            
            # Multi-timeframe (5%)
            mtf_weight = 0.05
            if mtf_direction == 'BULLISH':
                total_score += mtf_score * 3 * mtf_weight
            elif mtf_direction == 'BEARISH':
                total_score -= mtf_score * 3 * mtf_weight
            confidence_components.append(mtf_score * 100 * mtf_weight)

            # REGIME-SPECIFIC ADJUSTMENTS
            total_score, regime_adjustment = self.apply_regime_adjustments(total_score, regime_type, regime_details, latest)
            all_reasons.append(f"Regime Adj: {regime_adjustment}")

            # FINAL CONFIDENCE CALCULATION
            final_confidence = sum(confidence_components)
            final_confidence = max(20, min(final_confidence, 95))

            # ENHANCED SIGNAL DETERMINATION
            if total_score >= 5.0 and final_confidence >= 75:
                final_signal = "STRONG_BUY"
            elif total_score >= 2.5 and final_confidence >= 60:
                final_signal = "BUY"
            elif total_score <= -5.0 and final_confidence >= 75:
                final_signal = "STRONG_SELL"
            elif total_score <= -2.5 and final_confidence >= 60:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"
            
            # ADVANCED RISK MANAGEMENT
            risk_metrics = self.calculate_advanced_risk_metrics(df, regime_type, final_confidence)
            
            return {
                "signal": final_signal,
                "confidence": round(final_confidence, 1),
                "reasons": all_reasons[:6],
                "entry_price": latest['price'],
                "stop_loss": risk_metrics['stop_loss'],
                "take_profit": risk_metrics['take_profit'],
                "position_size": risk_metrics['position_size'],
                "ml_prediction": ml_prediction_text,
                "market_regime": regime_type,
                "regime_confidence": regime_confidence,
                "total_score": round(total_score, 2),
                "risk_reward_ratio": risk_metrics['risk_reward'],
                "feature_importance": feature_importance,
                "volatility": latest.get('volatility', 0.02),
                "regime_details": regime_details
            }
            
        except Exception as e:
            return {"signal": "HOLD", "confidence": 0, "reason": f"Advanced analysis error: {str(e)}"}
    
    def analyze_regime_adaptive_signals(self, df, regime_type):
        """Regime-adaptive traditional technical analysis"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        signal_score = 0
        confidence = 0
        reasons = []
        
        # RSI Analysis - Regime Adaptive
        rsi = latest.get('rsi', 50)
        if regime_type in ["Ranging", "Strong_Ranging"]:
            # In ranging markets, prioritize mean reversion
            if rsi <= 25:
                signal_score += 4; confidence += 45; reasons.append("RSI deeply oversold (range)")
            elif rsi <= 35:
                signal_score += 2; confidence += 25; reasons.append("RSI oversold (range)")
            elif rsi >= 75:
                signal_score -= 4; confidence += 45; reasons.append("RSI deeply overbought (range)")
            elif rsi >= 65:
                signal_score -= 2; confidence += 25; reasons.append("RSI overbought (range)")
        else:
            # In trending markets, use different RSI thresholds
            if rsi <= 35 and rsi >= 30:
                signal_score += 2; confidence += 20; reasons.append("RSI pullback in uptrend")
            elif rsi >= 65 and rsi <= 70:
                signal_score -= 2; confidence += 20; reasons.append("RSI pullback in downtrend")
        
        # Trend Analysis - Regime Adaptive
        sma_5 = latest.get('sma_5', 0)
        sma_10 = latest.get('sma_10', 0) 
        sma_20 = latest.get('sma_20', 0)
        
        if regime_type in ["Trending", "Strong_Trending"]:
            # In trending markets, follow the trend
            if sma_5 > sma_10 > sma_20:
                signal_score += 3; confidence += 25; reasons.append("Strong bullish trend confirmed")
            elif sma_5 < sma_10 < sma_20:
                signal_score -= 3; confidence += 25; reasons.append("Strong bearish trend confirmed")
            elif sma_5 > sma_20:
                signal_score += 1; confidence += 10; reasons.append("Bullish trend")
            elif sma_5 < sma_20:
                signal_score -= 1; confidence += 10; reasons.append("Bearish trend")
        else:
            # In ranging markets, fade extremes
            ma_convergence = abs(sma_5 - sma_20) / sma_20
            if ma_convergence < 0.01:
                confidence += 15; reasons.append("MAs converged (range market)")
        
        # MACD Analysis - Regime Adaptive
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_crossover_bull = latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']
            macd_crossover_bear = latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']
            
            if regime_type in ["Trending", "Strong_Trending"]:
                # In trending markets, MACD crossovers are more reliable
                if macd_crossover_bull:
                    signal_score += 3; confidence += 30; reasons.append("MACD bullish crossover (trend)")
                elif macd_crossover_bear:
                    signal_score -= 3; confidence += 30; reasons.append("MACD bearish crossover (trend)")
            else:
                # In ranging markets, MACD crossovers are less reliable
                if macd_crossover_bull:
                    signal_score += 1; confidence += 15; reasons.append("MACD bullish (range)")
                elif macd_crossover_bear:
                    signal_score -= 1; confidence += 15; reasons.append("MACD bearish (range)")
        
        # Bollinger Bands - Regime Adaptive
        if 'bb_position' in latest:
            bb_pos = latest['bb_position']
            if regime_type in ["Ranging", "Strong_Ranging"]:
                # In ranging markets, use BB as mean reversion indicator
                if bb_pos <= 0.1:
                    signal_score += 2; confidence += 20; reasons.append("Near lower BB (range)")
                elif bb_pos >= 0.9:
                    signal_score -= 2; confidence += 20; reasons.append("Near upper BB (range)")
            else:
                # In trending markets, BB breakouts are significant
                if bb_pos >= 0.95:
                    signal_score += 1; confidence += 15; reasons.append("BB breakout up (trend)")
                elif bb_pos <= 0.05:
                    signal_score -= 1; confidence += 15; reasons.append("BB breakout down (trend)")
        
        return signal_score, confidence, reasons
    
    def calculate_regime_score(self, regime_type, regime_details):
        """Calculate score based on market regime"""
        if regime_type == "Strong_Trending":
            return 2 if regime_details.get('ma_alignment') == 'Bullish' else -2 if regime_details.get('ma_alignment') == 'Bearish' else 0
        elif regime_type == "Trending":
            return 1 if regime_details.get('ma_alignment') == 'Bullish' else -1 if regime_details.get('ma_alignment') == 'Bearish' else 0
        elif regime_type in ["Ranging", "Strong_Ranging"]:
            # In ranging markets, neutral is preferred
            return 0
        else:
            return 0
    
    def apply_regime_adjustments(self, total_score, regime_type, regime_details, latest):
        """Apply regime-specific adjustments to total score"""
        adjustment_text = "None"
        
        if regime_type in ["Strong_Trending", "Trending"]:
            # In trending markets, amplify trend-following signals
            if total_score > 2:
                total_score *= 1.2
                adjustment_text = "Trend amplification (+20%)"
            elif total_score < -2:
                total_score *= 1.2
                adjustment_text = "Trend amplification (+20%)"
            else:
                total_score *= 0.8
                adjustment_text = "Weak trend signal (-20%)"
                
        elif regime_type in ["Strong_Ranging", "Ranging"]:
            # In ranging markets, amplify mean-reversion signals
            rsi = latest.get('rsi', 50)
            if (rsi <= 30 and total_score > 0) or (rsi >= 70 and total_score < 0):
                total_score *= 1.3
                adjustment_text = "Mean reversion boost (+30%)"
            elif (rsi > 45 and rsi < 55):
                total_score *= 0.7
                adjustment_text = "Neutral range (-30%)"
            else:
                total_score *= 0.9
                adjustment_text = "Range damping (-10%)"
                
        elif regime_type == "Transitional":
            # In transitional markets, reduce all signals
            total_score *= 0.6
            adjustment_text = "Transitional damping (-40%)"
            
        return total_score, adjustment_text
    
    def calculate_advanced_risk_metrics(self, df, regime_type, confidence):
        """Calculate advanced risk management metrics"""
        latest = df.iloc[-1]
        entry_price = latest['price']
        volatility = latest.get('volatility', 0.02)
        atr = latest.get('atr', entry_price * 0.02)
        
        # Base risk parameters
        base_stop_pct = max(0.015, volatility * 1.5)
        
        # Regime-adaptive risk management
        if regime_type in ["Strong_Trending", "Trending"]:
            # In trending markets, use wider stops
            stop_loss_pct = base_stop_pct * 1.5
            take_profit_pct = stop_loss_pct * 2.5
            risk_reward = 2.5
        elif regime_type in ["Strong_Ranging", "Ranging"]:
            # In ranging markets, use tighter stops
            stop_loss_pct = base_stop_pct * 0.8
            take_profit_pct = stop_loss_pct * 2.0
            risk_reward = 2.0
        else:
            # Default parameters
            stop_loss_pct = base_stop_pct
            take_profit_pct = stop_loss_pct * 2.0
            risk_reward = 2.0
        
        # Position sizing based on confidence and volatility
        base_position = 0.02  # 2% base position
        confidence_multiplier = confidence / 100
        volatility_multiplier = 0.02 / max(volatility, 0.01)
        
        position_size = min(0.08, base_position * confidence_multiplier * volatility_multiplier)
        
        return {
            'stop_loss': entry_price * (1 - stop_loss_pct),
            'take_profit': entry_price * (1 + take_profit_pct),
            'position_size': position_size,
            'risk_reward': risk_reward,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
    
    def get_market_sentiment_advanced(self, symbol, df):
        """Advanced market sentiment analysis using multiple factors."""
        try:
            if df.empty or len(df) < 10:
                return 0.0
                
            # Price momentum sentiment
            price_changes = df['price'].pct_change().tail(10)
            momentum_sentiment = np.tanh(price_changes.mean() * 100)  # Normalize to [-1, 1]
            
            # Volume sentiment
            volume_trend = df['volume'].tail(5).mean() / df['volume'].tail(10).mean()
            volume_sentiment = np.tanh((volume_trend - 1) * 2)
            
            # Technical sentiment
            latest = df.iloc[-1]
            rsi = latest.get('rsi', 50)
            tech_sentiment = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
            
            # MACD sentiment
            macd_hist = latest.get('macd_histogram', 0)
            macd_sentiment = np.tanh(macd_hist * 1000)  # Scale and normalize
            
            # Combine all sentiments
            overall_sentiment = (
                momentum_sentiment * 0.4 +
                volume_sentiment * 0.3 +
                tech_sentiment * 0.2 +
                macd_sentiment * 0.1
            )
            
            return np.clip(overall_sentiment, -1, 1)
            
        except Exception:
            return 0.0
    

    def simulate_trade_performance_advanced(self, signals_df, initial_capital=1000000, 
                                           min_confidence=50, use_trailing_stop=False, 
                                           trailing_stop_pct=0.02, max_position_size=0.1,
                                           commission_rate=0.001, slippage_rate=0.002,
                                           risk_per_trade=0.02, use_dynamic_sizing=True):
        """V4: Advanced trading simulation with comprehensive risk management and analytics."""
        if signals_df.empty:
            return self._create_empty_performance_advanced()
        
        try:
            # Initialize tracking variables
            capital = initial_capital
            positions = 0
            entry_price = 0
            entry_timestamp = None
            trailing_stop_price = 0
            equity_curve = [initial_capital]
            completed_trades = []
            drawdown_curve = [0]
            peak_capital = initial_capital
            
            # Performance metrics
            total_commission_paid = 0
            total_slippage_cost = 0
            max_drawdown = 0
            winning_trades = 0
            losing_trades = 0
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for idx, row in signals_df.iterrows():
                signal = row['signal']
                price = row['entry_price']
                confidence = row.get('confidence', 0)
                timestamp = row.get('timestamp', idx)
                regime = row.get('market_regime', 'Unknown')
                volatility = row.get('volatility', 0.02)
                
                # Calculate actual price with slippage
                if signal in ['BUY', 'STRONG_BUY']:
                    actual_price = price * (1 + slippage_rate)
                else:
                    actual_price = price * (1 - slippage_rate)
                
                # Entry Logic
                if signal in ['BUY', 'STRONG_BUY'] and positions == 0 and confidence >= min_confidence:
                    if use_dynamic_sizing:
                        # Dynamic position sizing based on confidence and volatility
                        base_size = min(max_position_size, risk_per_trade / max(volatility, 0.01))
                        confidence_adj = (confidence / 100) ** 0.5  # Square root scaling
                        position_size_pct = base_size * confidence_adj
                    else:
                        position_size_pct = min(max_position_size, confidence / 100.0 * 0.1)
                    
                    position_value = capital * position_size_pct
                    
                    if position_value > 1000 and capital > position_value:
                        commission_cost = position_value * commission_rate
                        slippage_cost = position_value * slippage_rate
                        total_cost = position_value + commission_cost
                        
                        if capital >= total_cost:
                            shares = position_value / actual_price
                            capital -= total_cost
                            positions = shares
                            entry_price = actual_price
                            entry_timestamp = timestamp
                            
                            total_commission_paid += commission_cost
                            total_slippage_cost += slippage_cost
                            
                            if use_trailing_stop:
                                trailing_stop_price = actual_price * (1 - trailing_stop_pct)
                
                # Exit Logic
                elif positions > 0:
                    should_exit = False
                    exit_reason = ""
                    stop_loss_price = row.get('stop_loss', entry_price * 0.98)
                    take_profit_price = row.get('take_profit', entry_price * 1.04)

                    # Trailing stop logic
                    if use_trailing_stop:
                        trailing_stop_price = max(trailing_stop_price, actual_price * (1 - trailing_stop_pct))
                        if actual_price <= trailing_stop_price:
                            should_exit = True
                            exit_reason = f"Trailing Stop ({trailing_stop_pct:.1%})"
                    
                    # Other exit conditions
                    if not should_exit:
                        if signal in ['SELL', 'STRONG_SELL'] and confidence >= min_confidence:
                            should_exit, exit_reason = True, f"SELL Signal (C:{confidence:.0f}%)"
                        elif actual_price <= stop_loss_price:
                            should_exit, exit_reason = True, "Stop-Loss"
                        elif actual_price >= take_profit_price:
                            should_exit, exit_reason = True, "Take-Profit"
                        elif regime in ['Strong_Ranging'] and abs(actual_price - entry_price)/entry_price > 0.03:
                            should_exit, exit_reason = True, "Regime Change"

                    if should_exit:
                        position_value = positions * actual_price
                        commission_cost = position_value * commission_rate
                        net_proceeds = position_value - commission_cost
                        capital += net_proceeds
                        
                        # Calculate trade performance
                        gross_pnl = (actual_price - entry_price) * positions
                        net_pnl = gross_pnl - (position_value * commission_rate) - (positions * entry_price * commission_rate)
                        pnl_pct = (actual_price / entry_price - 1) * 100
                        
                        # Track trade duration
                        try:
                            if entry_timestamp is not None and pd.notna(entry_timestamp):
                                trade_duration = pd.Timestamp(timestamp) - pd.Timestamp(entry_timestamp)
                            else:
                                trade_duration = pd.Timedelta(0)
                        except Exception:
                            trade_duration = pd.Timedelta(0)
                        
                        completed_trades.append({
                            'Entry Time': entry_timestamp,
                            'Exit Time': timestamp,
                            'Duration': trade_duration,
                            'Entry Price': round(entry_price, 2),
                            'Exit Price': round(actual_price, 2),
                            'Shares': round(positions, 0),
                            'Gross P&L': round(gross_pnl, 2),
                            'Net P&L': round(net_pnl, 2),
                            'P&L (%)': round(pnl_pct, 2),
                            'Exit Reason': exit_reason,
                            'Regime': regime,
                            'Confidence': confidence
                        })
                        
                        # Update win/loss tracking
                        if pnl_pct > 0:
                            winning_trades += 1
                            consecutive_wins += 1
                            consecutive_losses = 0
                            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        else:
                            losing_trades += 1
                            consecutive_losses += 1
                            consecutive_wins = 0
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        total_commission_paid += commission_cost
                        positions = 0
                        entry_price = 0
                        entry_timestamp = None
                
                # Update equity curve and drawdown
                current_equity = capital + (positions * actual_price if positions > 0 else 0)
                equity_curve.append(current_equity)
                
                if current_equity > peak_capital:
                    peak_capital = current_equity
                
                current_drawdown = (peak_capital - current_equity) / peak_capital * 100
                drawdown_curve.append(current_drawdown)
                max_drawdown = max(max_drawdown, current_drawdown)

            # Calculate comprehensive performance metrics
            final_capital = equity_curve[-1]
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            if completed_trades:
                trades_df = pd.DataFrame(completed_trades)
                win_rate = (winning_trades / len(completed_trades)) * 100
                
                avg_win = trades_df[trades_df['P&L (%)'] > 0]['P&L (%)'].mean() if winning_trades > 0 else 0
                avg_loss = trades_df[trades_df['P&L (%)'] < 0]['P&L (%)'].mean() if losing_trades > 0 else 0
                
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
                
                # Calculate Sharpe ratio (simplified)
                returns = pd.Series(equity_curve).pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Maximum adverse excursion
                max_adverse_excursion = trades_df['P&L (%)'].min() if not trades_df.empty else 0
                
            else:
                win_rate = 0
                avg_win = avg_loss = profit_factor = sharpe_ratio = max_adverse_excursion = 0
                trades_df = pd.DataFrame()
            
            return {
                'total_return': round(total_return, 2),
                'final_capital': round(final_capital, 2),
                'win_rate': round(win_rate, 1),
                'total_trades': len(completed_trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf',
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'total_commission': round(total_commission_paid, 2),
                'total_slippage': round(total_slippage_cost, 2),
                'max_adverse_excursion': round(max_adverse_excursion, 2),
                'equity_curve': equity_curve,
                'drawdown_curve': drawdown_curve,
                'trades': trades_df,
            }
            
        except Exception as e:
            st.error(f"Backtesting error: {str(e)}")
            return self._create_empty_performance_advanced()
    
    def _create_empty_performance_advanced(self):
        """Create empty performance results for advanced backtesting"""
        return {
            'total_return': 0.0, 'final_capital': 1000000, 'win_rate': 0.0, 'total_trades': 0,
            'winning_trades': 0, 'losing_trades': 0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'profit_factor': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0,
            'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
            'total_commission': 0.0, 'total_slippage': 0.0, 'max_adverse_excursion': 0.0,
            'equity_curve': [1000000], 'drawdown_curve': [0], 'trades': pd.DataFrame()
        }
    
    def optimize_strategy_parameters(self, signals_df, optimization_params):
        """Optimize strategy parameters using grid search"""
        if signals_df.empty:
            return {}
            
        best_params = {}
        best_score = -float('inf')
        optimization_results = []
        
        # Define parameter ranges
        confidence_range = optimization_params.get('confidence_range', [40, 50, 60, 70])
        trailing_stop_range = optimization_params.get('trailing_stop_range', [0.01, 0.015, 0.02, 0.025])
        max_position_range = optimization_params.get('max_position_range', [0.05, 0.08, 0.1, 0.12])
        
        total_combinations = len(confidence_range) * len(trailing_stop_range) * len(max_position_range)
        progress_bar = st.progress(0)
        current_iteration = 0
        
        for min_conf in confidence_range:
            for trailing_stop in trailing_stop_range:
                for max_pos in max_position_range:
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_combinations)
                    
                    # Run backtest with these parameters
                    results = self.simulate_trade_performance_advanced(
                        signals_df,
                        min_confidence=min_conf,
                        use_trailing_stop=True,
                        trailing_stop_pct=trailing_stop,
                        max_position_size=max_pos
                    )
                    
                    # Calculate optimization score (combination of return, win rate, and drawdown)
                    score = (
                        results['total_return'] * 0.4 +
                        results['win_rate'] * 0.3 -
                        results['max_drawdown'] * 0.2 +
                        (results['sharpe_ratio'] * 10) * 0.1
                    )
                    
                    optimization_results.append({
                        'min_confidence': min_conf,
                        'trailing_stop': trailing_stop,
                        'max_position': max_pos,
                        'score': score,
                        'total_return': results['total_return'],
                        'win_rate': results['win_rate'],
                        'max_drawdown': results['max_drawdown'],
                        'sharpe_ratio': results['sharpe_ratio'],
                        'total_trades': results['total_trades']
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'min_confidence': min_conf,
                            'trailing_stop_pct': trailing_stop,
                            'max_position_size': max_pos,
                            'score': score
                        }
        
        progress_bar.empty()
        
        return {
            'best_params': best_params,
            'all_results': pd.DataFrame(optimization_results),
            'best_score': best_score
        }
    
    def _create_empty_performance(self):
        """Create empty performance results - backward compatibility"""
        return {
            'total_return': 0.0, 'win_rate': 0.0, 'total_trades': 0,
            'equity_curve': [1000000], 'trades': pd.DataFrame(), 'final_capital': 1000000
        }
    
    def simulate_trade_performance(self, signals_df, initial_capital=1000000, min_confidence=50, use_trailing_stop=False, trailing_stop_pct=0.02):
        """Backward compatibility wrapper for the original method"""
        return self.simulate_trade_performance_advanced(
            signals_df, initial_capital, min_confidence, 
            use_trailing_stop, trailing_stop_pct
        )

@st.cache_resource(ttl=1800)  # Cache model for 30 minutes for more frequent updates
def get_enhanced_ml_model(symbol, df):
    """Train and cache an enhanced ML model with advanced features."""
    if not ML_AVAILABLE:
        return None
        
    try:
        system = PSXAlgoTradingSystem()
        df_full = system.calculate_technical_indicators(df)
        df_full = system.calculate_volume_indicators(df_full)

        X, y, features = system._prepare_ml_features(df_full)
        
        if X is None or y is None or X.empty or y.empty or len(y.unique()) < 2:
            return None
        
        if len(X) < 50:  # Need sufficient data for training
            return None

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Enhanced Random Forest with optimized parameters
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validate model performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Only return model if it performs reasonably well
        if test_score > 0.52:  # Better than random + some edge
            return model
        else:
            return None
            
    except Exception as e:
        st.warning(f"ML model training failed for {symbol}: {str(e)}")
        return None

# Backward compatibility
def get_ml_model(symbol, df):
    """Backward compatibility function"""
    return get_enhanced_ml_model(symbol, df)
