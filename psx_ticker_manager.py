#!/usr/bin/env python3
"""
PSX Ticker Manager
==================

Complete ticker management system for Pakistan Stock Exchange
- All 893 PSX symbols
- 722 stock symbols (excluding bonds/ETFs) 
- Organized by sectors
- Search and filtering capabilities
"""

from typing import List, Dict, Optional
import pandas as pd

# Get all PSX tickers dynamically
def get_all_psx_tickers() -> List[str]:
    """Get all 893 PSX tickers including bonds, ETFs, etc."""
    try:
        from psx import tickers
        ticker_df = tickers()
        return ticker_df['symbol'].tolist()
    except Exception as e:
        print(f"Warning: Could not fetch live tickers, using fallback list: {e}")
        return get_fallback_tickers()

def get_stock_symbols_only() -> List[str]:
    """Get 722 stock symbols only (excluding bonds, ETFs)"""
    try:
        from psx import tickers
        ticker_df = tickers()
        stocks_only = ticker_df[
            (ticker_df['isDebt'] != True) & 
            (ticker_df['isETF'] != True)
        ]
        return stocks_only['symbol'].tolist()
    except Exception as e:
        print(f"Warning: Could not fetch live tickers, using fallback list: {e}")
        return get_fallback_stock_symbols()

def get_tickers_by_sector() -> Dict[str, List[str]]:
    """Get tickers organized by sectors"""
    try:
        from psx import tickers
        ticker_df = tickers()
        
        # Filter for stocks only
        stocks_df = ticker_df[
            (ticker_df['isDebt'] != True) & 
            (ticker_df['isETF'] != True)
        ]
        
        # Group by sector
        sectors = {}
        for sector in stocks_df['sectorName'].unique():
            if pd.notna(sector) and sector.strip():  # Skip empty sectors
                sector_tickers = stocks_df[stocks_df['sectorName'] == sector]['symbol'].tolist()
                sectors[sector.strip()] = sorted(sector_tickers)
        
        return sectors
    except Exception as e:
        print(f"Warning: Could not fetch live sectors, using fallback: {e}")
        return get_fallback_sectors()

def get_top_symbols() -> List[str]:
    """Get top/popular PSX symbols for quick access"""
    return [
        # Banking
        'UBL', 'MCB', 'HBL', 'ABL', 'NBP', 'BAFL', 'AKBL', 'BAHL', 'SILK', 'KASB',
        # Oil & Gas
        'OGDC', 'PPL', 'POL', 'MARI', 'MPCL', 'SNGP', 'SSGC',
        # Cement
        'LUCK', 'DG&CEPR', 'MLCF', 'PIOC', 'ACPL', 'CHCC', 'FCCL',
        # Fertilizer
        'ENGRO', 'FFC', 'FATIMA',
        # Power
        'KAPCO', 'HUBC', 'KEL', 'KESC',
        # Telecom
        'PTC', 'TRG', 'NETSOL',
        # Food & Beverages
        'NESTLE', 'UFL', 'UNITY', 'PSMC',
        # Pharma
        'SEARL', 'GSK', 'IBL', 'HINOON',
        # Steel
        'ISL', 'ASL', 'ASTL',
        # Textile
        'GATM', 'KTML', 'SITC', 'KOTML'
    ]

def search_symbols(query: str, limit: int = 20) -> List[str]:
    """Search for symbols matching query"""
    try:
        all_symbols = get_stock_symbols_only()
        query_upper = query.upper()
        
        # Exact matches first
        exact_matches = [s for s in all_symbols if s == query_upper]
        
        # Then starts with
        starts_with = [s for s in all_symbols if s.startswith(query_upper) and s not in exact_matches]
        
        # Then contains
        contains = [s for s in all_symbols if query_upper in s and s not in exact_matches and s not in starts_with]
        
        # Combine and limit
        results = exact_matches + starts_with + contains
        return results[:limit]
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

# Fallback data when PSX API is not available
def get_fallback_tickers() -> List[str]:
    """Fallback ticker list for when API is unavailable"""
    return [
        '786', 'AABS', 'AAL', 'AASM', 'AATM', 'ABL', 'ABOT', 'ABSON', 'ACPL', 'ADAMS',
        'ADMM', 'ADOS', 'ADTM', 'AEL', 'AGHA', 'AGIC', 'AGIL', 'AGL', 'AGLNCPS', 'AGP',
        'AGSML', 'AGTML', 'AHL', 'AICL', 'AIRLINK', 'AKD', 'AKBL', 'AKS', 'AL-ABBAS',
        'AL-NOOR', 'ALICO', 'ALYSP', 'AMBL', 'AMRC', 'AMSL', 'ANL', 'AOT', 'APL', 'ARPL',
        'ASC', 'ASGHAR', 'ASL', 'ASTL', 'ATBA', 'ATIL', 'ATLH', 'ATRL', 'BAFL', 'BAHL',
        'BAHRIA', 'BATIC', 'BCL', 'BNWM', 'BOP', 'BUXL', 'BYCO', 'CHCC', 'CHCL', 'CIL',
        'CLCPS', 'CLL', 'COLG', 'CRTM', 'DCL', 'DCTL', 'DGKC', 'DOL', 'DPL', 'DSL',
        'DYNO', 'EFOODS', 'EFU', 'EFUG', 'EFUGI', 'EFUGS', 'ENGRO', 'ESBL', 'EXIDE',
        'FASM', 'FATIMA', 'FCCL', 'FCL', 'FEL', 'FFC', 'FFBL', 'FFL', 'FHAM', 'FHBL',
        'FIBL', 'FIL', 'FLYNG', 'FML', 'FRSH', 'FZCM', 'GATM', 'GGL', 'GHGL', 'GHNI',
        'GIL', 'GLAXO', 'GSKCH', 'GTL', 'GWLC', 'HABSM', 'HADC', 'HASCOL', 'HBL',
        'HCAR', 'HCL', 'HGFA', 'HIBL', 'HINOON', 'HMB', 'HUMNL', 'HUBC', 'HWA', 'IBL',
        'ICI', 'ICL', 'IDRT', 'IFL', 'IGIHL', 'IGIL', 'ILP', 'IMAGE', 'INDU', 'ISL',
        'JDWS', 'JGICL', 'JSBL', 'JSCL', 'JSML', 'KAPCO', 'KASB', 'KASM', 'KCL', 'KESC',
        'KEL', 'KGTL', 'KOHINOOR', 'KOTML', 'KPHL', 'KSBP', 'KTML', 'LADDF', 'LOADS',
        'LPGL', 'LRBS', 'LSVM', 'LUCK', 'MACFL', 'MARI', 'MCB', 'MEBL', 'MERIT', 'MFL',
        'MGHL', 'MLCF', 'MTL', 'MUBL', 'MUREB', 'NATF', 'NATM', 'NBP', 'NCL', 'NCPL',
        'NESTLE', 'NETSOL', 'NML', 'NRSL', 'NTML', 'NTM', 'ODGC', 'OGDC', 'OLPL', 'OML',
        'PACE', 'PAKD', 'PAKOXY', 'PAKRI', 'PAKT', 'PASL', 'PCAL', 'PGLC', 'PHL', 'PIAA',
        'PIBTL', 'PICL', 'PIM', 'PINL', 'PIOC', 'PKGP', 'PKGS', 'PLL', 'PMPK', 'PNL',
        'POL', 'PPL', 'PPVC', 'PRWM', 'PSL', 'PSM', 'PSMC', 'PSO', 'PTC', 'PTCL', 'RANS',
        'RCML', 'REDCO', 'RMPL', 'RUPL', 'SAZEW', 'SBL', 'SCBPL', 'SEARL', 'SEL', 'SGPL',
        'SHEL', 'SIEM', 'SILK', 'SIML', 'SITC', 'SLCL', 'SLL', 'SLYL', 'SMBL', 'SML',
        'SNBL', 'SNGP', 'SPCL', 'SPEL', 'SPLC', 'SPL', 'SSGC', 'SSIC', 'SSL', 'STJT',
        'STPL', 'SYS', 'TGL', 'THALL', 'TM', 'TOMCL', 'TRG', 'TRSM', 'TSML', 'TUL',
        'UBDL', 'UBL', 'UDL', 'UFL', 'UNITY', 'UPFL', 'UVIC', 'WTL', 'YOUW', 'ZIL',
        'ZTL'
    ]

def get_fallback_stock_symbols() -> List[str]:
    """Fallback stock symbols (same as all for now)"""
    return get_fallback_tickers()

def get_fallback_sectors() -> Dict[str, List[str]]:
    """Fallback sector organization"""
    return {
        'Banking': ['UBL', 'MCB', 'HBL', 'ABL', 'NBP', 'BAFL', 'AKBL', 'BAHL', 'SILK', 'KASB', 'BOP', 'JSBL', 'MEBL', 'SNBL'],
        'Oil & Gas': ['OGDC', 'PPL', 'POL', 'MARI', 'MPCL', 'SNGP', 'SSGC', 'PSO', 'HASCOL', 'BYCO'],
        'Cement': ['LUCK', 'DGKC', 'MLCF', 'PIOC', 'ACPL', 'CHCC', 'FCCL', 'GWLC', 'THALL', 'KOHINOOR'],
        'Fertilizer': ['ENGRO', 'FFC', 'FATIMA', 'EFERT'],
        'Power': ['KAPCO', 'HUBC', 'KEL', 'KESC', 'LPGL', 'PKGP'],
        'Telecom': ['PTC', 'TRG', 'NETSOL', 'PACE', 'PTCL'],
        'Food & Beverages': ['NESTLE', 'UFL', 'UNITY', 'PSMC', 'FRSH', 'EFOODS'],
        'Pharmaceutical': ['SEARL', 'GLAXO', 'IBL', 'HINOON', 'ABBOTT', 'GSK'],
        'Steel': ['ISL', 'ASL', 'ASTL', 'ITTEFAQ'],
        'Textile Spinning': ['GATM', 'KTML', 'SITC', 'KOTML', 'YOUW', 'ZIL'],
        'Chemical': ['ICI', 'LOTTE', 'DYNEA', 'COLG', 'GHGL'],
        'Sugar': ['JDW', 'CHSR', 'HABSM', 'THAL'],
        'Insurance': ['ALICO', 'EFU', 'IGI', 'NICL'],
        'Engineering': ['HMB', 'INDU', 'HCL'],
        'Technology': ['NETSOL', 'TRG', 'SYSTEMSLTD', 'AVANCEON']
    }

# Test functions
def test_ticker_manager():
    """Test the ticker manager functionality"""
    print("🚀 Testing PSX Ticker Manager")
    print("=" * 50)
    
    # Test all tickers
    all_tickers = get_all_psx_tickers()
    print(f"📊 Total PSX tickers: {len(all_tickers)}")
    
    # Test stock symbols only
    stock_symbols = get_stock_symbols_only()
    print(f"📈 Stock symbols only: {len(stock_symbols)}")
    
    # Test sectors
    sectors = get_tickers_by_sector()
    print(f"🏢 Available sectors: {len(sectors)}")
    for sector, symbols in list(sectors.items())[:5]:  # Show first 5 sectors
        print(f"   {sector}: {len(symbols)} symbols")
    
    # Test search
    search_results = search_symbols("UBL", limit=5)
    print(f"🔍 Search 'UBL': {search_results}")
    
    # Test top symbols
    top_symbols = get_top_symbols()
    print(f"⭐ Top symbols: {len(top_symbols)} ({top_symbols[:10]}...)")
    
    print("\n✅ Ticker manager test completed!")

if __name__ == "__main__":
    test_ticker_manager()