#!/usr/bin/env python3
"""
Professional Trading System - Installation & Setup
==================================================

Installs all required dependencies and sets up the enhanced trading system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🚀 Professional PSX Trading System - Installation")
    print("=" * 60)
    
    # Core requirements
    core_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "requests>=2.28.0",
        "python-dateutil>=2.8.0"
    ]
    
    # Enhanced features
    enhanced_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "streamlit>=1.25.0",
        "openpyxl>=3.0.0",  # Excel export
        "ta-lib",  # Technical analysis (optional)
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0"  # For advanced analytics
    ]
    
    # Optional packages
    optional_packages = [
        "jupyter>=1.0.0",  # For notebooks
        "dash>=2.0.0",  # Alternative web framework
        "fastapi>=0.95.0",  # API endpoints
        "uvicorn>=0.20.0",  # ASGI server
        "yfinance>=0.2.0",  # Alternative data source
        "ccxt>=3.0.0"  # Crypto exchange APIs
    ]
    
    print("📦 Installing core packages...")
    core_success = all(install_package(pkg) for pkg in core_packages)
    
    print("\n🔧 Installing enhanced features...")
    enhanced_success = all(install_package(pkg) for pkg in enhanced_packages)
    
    print("\n⚡ Installing optional packages...")
    optional_success = sum(install_package(pkg) for pkg in optional_packages)
    
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    print(f"✅ Core packages: {'SUCCESS' if core_success else 'FAILED'}")
    print(f"🔧 Enhanced features: {'SUCCESS' if enhanced_success else 'PARTIAL'}")
    print(f"⚡ Optional packages: {optional_success}/{len(optional_packages)} installed")
    
    if core_success:
        print("\n🎉 INSTALLATION COMPLETE!")
        print("\n🚀 Quick Start Commands:")
        print("1. Basic analysis:")
        print("   python professional_trading_system.py --symbol UBL")
        print("\n2. Portfolio report:")
        print("   python professional_trading_system.py --portfolio-report")
        print("\n3. Web interface:")
        print("   streamlit run streamlit_app.py")
        print("\n4. Multi-symbol scan:")
        print("   python professional_trading_system.py --scan-symbols UBL MCB FFC")
        
        print("\n⚙️ Configuration:")
        print("1. Set your EODHD API key:")
        print("   export EODHD_API_KEY='your_key_here'")
        print("\n2. Edit config.json to customize parameters")
        
        print("\n📖 Documentation:")
        print("- Check README.md for detailed usage")
        print("- Run with --help for all options")
        print("- Logs saved to trading_bot.log")
        
    else:
        print("\n❌ INSTALLATION FAILED")
        print("Please check your Python environment and try again.")
        print("You may need to upgrade pip: python -m pip install --upgrade pip")

if __name__ == '__main__':
    main()