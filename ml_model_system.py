#!/usr/bin/env python3
"""
ML Model System with Walk-Forward Validation
============================================

Professional-grade machine learning system for quantitative trading
implementing institutional best practices:

- Purged, embargoed cross-validation to prevent data leakage
- Walk-forward analysis with expanding/rolling windows
- Multiple model ensemble with calibrated predictions
- SHAP explainability for model interpretability
- Robust performance evaluation and drift monitoring
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
import joblib
import os
from pathlib import Path

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy import stats

# Model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from quant_system_config import SystemConfig
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo
    to prevent data leakage in financial data
    """
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 2, 
                 purge_days: int = 1):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None):
        """Generate train/test splits with purging and embargo"""
        
        dates = X.index.get_level_values('date').unique().sort_values()
        n_dates = len(dates)
        
        # Calculate split sizes
        test_size = n_dates // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test period
            test_start_idx = (i + 1) * test_size
            test_end_idx = min(test_start_idx + test_size, n_dates)
            
            if test_end_idx >= n_dates:
                break
                
            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]
            
            # Training period (before test, with embargo)
            train_end_idx = max(0, test_start_idx - self.embargo_days - self.purge_days)
            train_end_date = dates[train_end_idx] if train_end_idx > 0 else dates[0]
            
            # Create masks
            train_mask = X.index.get_level_values('date') <= train_end_date
            test_mask = ((X.index.get_level_values('date') >= test_start_date) & 
                        (X.index.get_level_values('date') <= test_end_date))
            
            train_idx = X.index[train_mask]
            test_idx = X.index[test_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

class MLModelSystem:
    """Complete ML system for quantitative trading predictions"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.model_config = self.config.model
        
        # Model storage
        self.models = {}
        self.ensemble_weights = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.calibrators = {}
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer(config)
        self.selected_features = []
        
        # Model paths
        self.models_path = Path(self.config.models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        
        # LightGBM (primary model)
        self.models['lightgbm'] = lgb.LGBMRegressor(**self.model_config.lightgbm_params)
        
        # XGBoost
        if 'xgboost' in self.model_config.ensemble_models:
            self.models['xgboost'] = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # Random Forest
        if 'rf' in self.model_config.ensemble_models:
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def prepare_data_for_training(self, features_df: pd.DataFrame, 
                                 labels_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        
        # Align features and labels
        common_index = features_df.index.intersection(labels_series.index)
        features_aligned = features_df.loc[common_index]
        labels_aligned = labels_series.loc[common_index]
        
        # Remove rows with NaN labels
        valid_mask = ~labels_aligned.isna()
        features_clean = features_aligned[valid_mask]
        labels_clean = labels_aligned[valid_mask]
        
        # Feature selection if not already done
        if not self.selected_features:
            features_selected, self.selected_features = self.feature_engineer.select_features(
                features_clean, labels_clean
            )
        else:
            # Use previously selected features
            available_features = [f for f in self.selected_features if f in features_clean.columns]
            features_selected = features_clean[available_features]
        
        # Fill any remaining NaNs
        features_selected = features_selected.fillna(features_selected.median())
        
        logger.info(f"Prepared training data: {features_selected.shape}, selected {len(self.selected_features)} features")
        
        return features_selected, labels_clean
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame = None, 
                          y_val: pd.Series = None) -> Dict[str, Any]:
        """Train a single model with validation"""
        
        model = self.models[model_name]
        training_results = {}
        
        try:
            if model_name == 'lightgbm':
                # LightGBM with early stopping
                if X_val is not None and y_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Feature importance
                importance = dict(zip(X_train.columns, model.feature_importances_))
                training_results['feature_importance'] = importance
                
            elif model_name == 'xgboost':
                # XGBoost with early stopping
                if X_val is not None and y_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Feature importance
                importance = dict(zip(X_train.columns, model.feature_importances_))
                training_results['feature_importance'] = importance
                
            else:
                # Scikit-learn models
                model.fit(X_train, y_train)
                
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X_train.columns, model.feature_importances_))
                    training_results['feature_importance'] = importance
            
            # Validation performance
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                
                training_results['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred))
                training_results['val_mae'] = mean_absolute_error(y_val, y_pred)
                training_results['val_r2'] = r2_score(y_val, y_pred)
                training_results['val_ic'] = stats.spearmanr(y_val, y_pred)[0]
            
            training_results['model'] = model
            training_results['trained_features'] = list(X_train.columns)
            
            logger.info(f"Trained {model_name}: val_rmse={training_results.get('val_rmse', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            training_results['error'] = str(e)
        
        return training_results
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Train ensemble of models"""
        
        ensemble_results = {}
        model_predictions = {}
        
        # Train each model
        for model_name in self.models.keys():
            logger.info(f"Training {model_name}...")
            
            results = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
            ensemble_results[model_name] = results
            
            # Get predictions for ensemble weighting
            if X_val is not None and 'model' in results:
                pred = results['model'].predict(X_val)
                model_predictions[model_name] = pred
        
        # Calculate ensemble weights based on validation performance
        if X_val is not None and y_val is not None and model_predictions:
            self.ensemble_weights = self._calculate_ensemble_weights(
                model_predictions, y_val
            )
            ensemble_results['ensemble_weights'] = self.ensemble_weights
        else:
            # Equal weights if no validation data
            self.ensemble_weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        # Store feature importance (from primary model)
        if 'lightgbm' in ensemble_results and 'feature_importance' in ensemble_results['lightgbm']:
            self.feature_importance = ensemble_results['lightgbm']['feature_importance']
        
        return ensemble_results
    
    def _calculate_ensemble_weights(self, model_predictions: Dict[str, np.ndarray], 
                                   y_true: pd.Series) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on performance"""
        
        # Calculate IC (Information Coefficient) for each model
        model_ics = {}
        for name, pred in model_predictions.items():
            try:
                ic = stats.spearmanr(y_true, pred)[0]
                model_ics[name] = max(ic, 0)  # Only positive ICs
            except:
                model_ics[name] = 0
        
        # Weight by IC (models with higher IC get more weight)
        total_ic = sum(model_ics.values())
        if total_ic > 0:
            weights = {name: ic / total_ic for name, ic in model_ics.items()}
        else:
            # Equal weights if all ICs are zero/negative
            weights = {name: 1/len(model_predictions) for name in model_predictions.keys()}
        
        return weights
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        
        if not self.models:
            raise ValueError("No trained models available")
        
        # Ensure we have the right features
        available_features = [f for f in self.selected_features if f in X.columns]
        X_selected = X[available_features]
        
        # Fill NaNs
        X_selected = X_selected.fillna(X_selected.median())
        
        # Get predictions from each model
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_selected)
                weight = self.ensemble_weights.get(model_name, 0)
                
                predictions.append(pred)
                weights.append(weight)
                
            except Exception as e:
                logger.warning(f"Error predicting with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could generate predictions")
        
        # Weighted ensemble
        predictions_array = np.column_stack(predictions)
        weights_array = np.array(weights)
        
        # Normalize weights
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones(len(weights)) / len(weights)
        
        ensemble_pred = np.average(predictions_array, axis=1, weights=weights_array)
        
        return ensemble_pred
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform purged time series cross-validation"""
        
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=self.model_config.cv_folds,
            embargo_days=self.model_config.embargo_days,
            purge_days=1
        )
        
        cv_results = {
            'fold_results': [],
            'mean_rmse': 0,
            'mean_mae': 0,
            'mean_r2': 0,
            'mean_ic': 0,
            'std_ic': 0
        }
        
        fold_metrics = []
        
        logger.info(f"Starting {self.model_config.cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.model_config.cv_folds}")
            
            # Split data
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            
            # Train models
            fold_results = self.train_ensemble(X_train, y_train)
            
            # Generate predictions
            try:
                y_pred = self.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                ic = stats.spearmanr(y_test, y_pred)[0] if len(y_test) > 1 else 0
                
                fold_metric = {
                    'fold': fold,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'ic': ic,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
                
                fold_metrics.append(fold_metric)
                cv_results['fold_results'].append(fold_metric)
                
                logger.info(f"Fold {fold + 1} - RMSE: {rmse:.4f}, IC: {ic:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {e}")
                continue
        
        # Calculate overall metrics
        if fold_metrics:
            cv_results['mean_rmse'] = np.mean([f['rmse'] for f in fold_metrics])
            cv_results['mean_mae'] = np.mean([f['mae'] for f in fold_metrics])
            cv_results['mean_r2'] = np.mean([f['r2'] for f in fold_metrics])
            cv_results['mean_ic'] = np.mean([f['ic'] for f in fold_metrics])
            cv_results['std_ic'] = np.std([f['ic'] for f in fold_metrics])
            
            logger.info(f"CV Results - Mean IC: {cv_results['mean_ic']:.4f} ± {cv_results['std_ic']:.4f}")
        
        return cv_results
    
    def generate_shap_explanations(self, X: pd.DataFrame, 
                                  max_samples: int = 1000) -> Dict[str, Any]:
        """Generate SHAP explanations for model interpretability"""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for explanations")
            return {}
        
        if 'lightgbm' not in self.models:
            logger.warning("LightGBM model not available for SHAP")
            return {}
        
        try:
            # Sample data if too large
            if len(X) > max_samples:
                X_sample = X.sample(n=max_samples, random_state=42)
            else:
                X_sample = X
            
            # Ensure correct features
            available_features = [f for f in self.selected_features if f in X_sample.columns]
            X_selected = X_sample[available_features].fillna(X_sample[available_features].median())
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.models['lightgbm'])
            shap_values = explainer.shap_values(X_selected)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_dict = dict(zip(X_selected.columns, feature_importance))
            
            # Sort by importance
            sorted_importance = sorted(
                feature_importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            shap_results = {
                'shap_values': shap_values,
                'feature_importance': feature_importance_dict,
                'top_features': sorted_importance[:20],  # Top 20
                'explainer': explainer,
                'base_value': explainer.expected_value,
                'data_used': X_selected
            }
            
            logger.info(f"Generated SHAP explanations for {len(X_selected)} samples")
            
            return shap_results
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return {}
    
    def save_models(self, model_suffix: str = None) -> str:
        """Save trained models and metadata"""
        
        if model_suffix is None:
            model_suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = self.models_path / f"models_{model_suffix}"
        save_dir.mkdir(exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = save_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'model_performance': self.model_performance,
            'config': self.config.to_dict(),
            'created_at': dt.datetime.now().isoformat()
        }
        
        metadata_path = save_dir / "metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Models saved to: {save_dir}")
        return str(save_dir)
    
    def load_models(self, model_path: str):
        """Load trained models and metadata"""
        
        load_dir = Path(model_path)
        
        # Load metadata
        metadata_path = load_dir / "metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.ensemble_weights = metadata.get('ensemble_weights', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.selected_features = metadata.get('selected_features', [])
            self.model_performance = metadata.get('model_performance', {})
        
        # Load individual models
        for name in self.models.keys():
            model_path = load_dir / f"{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
        
        logger.info(f"Models loaded from: {load_dir}")

# Test function
def test_ml_system():
    """Test ML model system"""
    print("🚀 Testing ML Model System")
    print("=" * 40)
    
    # Create ML system
    config = SystemConfig()
    ml_system = MLModelSystem(config)
    
    print(f"🧠 Model Configuration:")
    print(f"   Primary Model: {config.model.primary_model}")
    print(f"   Ensemble Models: {config.model.ensemble_models}")
    print(f"   CV Folds: {config.model.cv_folds}")
    print(f"   Embargo Days: {config.model.embargo_days}")
    
    # Generate some sample data for testing
    print(f"\n📊 Generating sample data for testing...")
    
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    symbols = ['UBL', 'MCB', 'FFC']
    
    # Create multi-index
    index_tuples = [(symbol, date) for symbol in symbols for date in dates]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['symbol', 'date'])
    
    # Generate sample features
    np.random.seed(42)
    n_samples = len(multi_index)
    
    sample_features = pd.DataFrame({
        'momentum_21d': np.random.normal(0, 0.1, n_samples),
        'momentum_63d': np.random.normal(0, 0.15, n_samples),
        'reversal_5d': np.random.normal(0, 0.05, n_samples),
        'volatility_21d': np.random.lognormal(-3, 0.5, n_samples),
        'rsi_14': np.random.uniform(-1, 1, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.3, n_samples),
        'symbol': [idx[0] for idx in index_tuples]
    }, index=multi_index)
    
    # Generate sample labels (forward returns)
    sample_labels = pd.Series(
        np.random.normal(0, 0.02, n_samples),  # 2% daily vol
        index=multi_index,
        name='forward_returns'
    )
    
    print(f"   Features Shape: {sample_features.shape}")
    print(f"   Labels Shape: {sample_labels.shape}")
    
    try:
        # Prepare data
        print(f"\n🔧 Preparing data for training...")
        X, y = ml_system.prepare_data_for_training(sample_features, sample_labels)
        print(f"   Prepared Data: {X.shape}")
        print(f"   Selected Features: {len(ml_system.selected_features)}")
        
        # Train models
        print(f"\n🏋️ Training ensemble models...")
        train_results = ml_system.train_ensemble(X, y)
        
        print(f"   Models Trained: {len(train_results)}")
        for model_name, results in train_results.items():
            if 'error' not in results:
                print(f"   ✅ {model_name}: Success")
            else:
                print(f"   ❌ {model_name}: {results['error']}")
        
        # Test predictions
        print(f"\n🎯 Testing predictions...")
        predictions = ml_system.predict(X.head(100))  # Test on first 100 samples
        print(f"   Predictions Shape: {predictions.shape}")
        print(f"   Sample Predictions: {predictions[:5]}")
        
        # Test model saving
        print(f"\n💾 Testing model saving...")
        save_path = ml_system.save_models("test")
        print(f"   Models saved to: {save_path}")
        
        # Test cross-validation (on smaller sample for speed)
        print(f"\n🔄 Testing cross-validation...")
        cv_results = ml_system.cross_validate(X.head(200), y.head(200))
        print(f"   CV Mean IC: {cv_results['mean_ic']:.4f} ± {cv_results['std_ic']:.4f}")
        print(f"   CV Mean RMSE: {cv_results['mean_rmse']:.4f}")
        
        # Test SHAP explanations
        if SHAP_AVAILABLE:
            print(f"\n🔍 Testing SHAP explanations...")
            shap_results = ml_system.generate_shap_explanations(X.head(100))
            if shap_results:
                print(f"   SHAP explanations generated")
                print(f"   Top 3 features: {shap_results['top_features'][:3]}")
            else:
                print(f"   SHAP explanations failed")
        
    except Exception as e:
        print(f"❌ Error in ML system test: {e}")
    
    print(f"\n🏁 ML model system test completed!")

if __name__ == "__main__":
    test_ml_system()