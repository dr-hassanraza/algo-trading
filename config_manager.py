#!/usr/bin/env python3
"""
Configuration Manager
====================

Centralized configuration management for the PSX trading system.
Handles loading settings from config.json and environment variables.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages configuration settings for the trading system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file is missing"""
        return {
            "trading_parameters": {
                "ma_period": 44,
                "bb_period": 20,
                "rsi_period": 14,
                "atr_period": 14
            },
            "risk_management": {
                "default_account_risk_pct": 2.0
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def _setup_logging(self):
        """Configure logging based on config settings"""
        log_config = self.config.get('logging', {})
        
        # Set log level
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(
                    log_config.get('file', 'trading_bot.log'), 
                    mode='a'
                )
            ]
        )
        
        logging.info("Configuration manager initialized")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: get('trading_parameters.ma_period', 44)
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # Check environment variable
            env_key = f"TRADING_{'_'.join(keys).upper()}"
            env_value = os.getenv(env_key)
            if env_value:
                # Try to convert to appropriate type
                try:
                    return float(env_value) if '.' in env_value else int(env_value)
                except ValueError:
                    return env_value
            
            logging.debug(f"Config key '{key_path}' not found, using default: {default}")
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
        logging.info(f"Config updated: {key_path} = {value}")
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def get_trading_params(self) -> Dict[str, Any]:
        """Get all trading parameters"""
        return self.config.get('trading_parameters', {})
    
    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.config.get('risk_management', {})
    
    def get_signal_thresholds(self) -> Dict[str, Any]:
        """Get signal threshold parameters"""
        return self.config.get('signal_thresholds', {})
    
    def get_scoring_weights(self) -> Dict[str, Any]:
        """Get scoring weight parameters"""
        return self.config.get('scoring_weights', {})

# Global config instance
config = ConfigManager()

# Convenience functions
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key_path, default)

def set_config(key_path: str, value: Any):
    """Set configuration value"""
    config.set(key_path, value)

def save_config():
    """Save configuration"""
    config.save()

# Setup logger for this module
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Test configuration manager
    print("Testing Configuration Manager...")
    
    # Test getting values
    ma_period = get_config('trading_parameters.ma_period', 44)
    print(f"MA Period: {ma_period}")
    
    # Test setting values
    set_config('test.value', 123)
    test_value = get_config('test.value')
    print(f"Test Value: {test_value}")
    
    # Test environment override
    os.environ['TRADING_TRADING_PARAMETERS_RSI_PERIOD'] = '21'
    rsi_period = get_config('trading_parameters.rsi_period', 14)
    print(f"RSI Period (env override): {rsi_period}")
    
    print("Configuration manager test complete!")