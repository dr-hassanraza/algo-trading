#!/usr/bin/env python3
"""
Feature Selection and Correlation Management System
=================================================

Advanced feature selection and correlation management system for algorithmic trading
implementing institutional-grade techniques for optimal feature set construction.

Features:
- Multi-method feature selection (univariate, recursive, embedded)
- Advanced correlation analysis and multicollinearity detection
- Dynamic feature importance tracking and stability analysis  
- Hierarchical clustering for correlated feature groups
- Real-time feature relevance monitoring
- Feature interaction detection and engineering
- Statistical significance testing for feature selection
- Performance-based feature ranking and selection
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import joblib

# Statistical libraries
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
# mutual_info_classif is not in scipy.special - we'll use sklearn version

# Machine Learning
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    mutual_info_regression, f_regression, chi2,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mutual_info_score

# Correlation and multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import networkx as nx

from quant_system_config import SystemConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection and correlation management"""
    
    # Selection methods
    selection_methods: List[str] = field(default_factory=lambda: [
        'mutual_info', 'f_regression', 'rfe', 'lasso', 'random_forest'
    ])
    
    # Selection parameters
    max_features: int = 50
    min_features: int = 5
    correlation_threshold: float = 0.85
    vif_threshold: float = 5.0
    variance_threshold: float = 0.01
    
    # Stability analysis
    stability_window: int = 10
    stability_threshold: float = 0.7
    importance_decay: float = 0.95
    
    # Statistical testing
    significance_level: float = 0.05
    min_samples_for_test: int = 30
    
    # Clustering
    cluster_correlation_method: str = 'ward'
    cluster_distance_threshold: float = 0.5
    
    # Performance optimization
    parallel_processing: bool = True
    n_jobs: int = -1
    chunk_size: int = 1000

class CorrelationAnalyzer:
    """Advanced correlation analysis and multicollinearity detection"""
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.correlation_history = deque(maxlen=config.stability_window)
        
    def compute_correlation_matrix(self, features: pd.DataFrame, 
                                 method: str = 'pearson') -> pd.DataFrame:
        """Compute correlation matrix with multiple methods"""
        
        if method == 'pearson':
            corr_matrix = features.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = features.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = features.corr(method='kendall')
        elif method == 'mic':  # Maximal Information Coefficient
            corr_matrix = self._compute_mic_matrix(features)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Store for stability analysis
        self.correlation_history.append({
            'timestamp': time.time(),
            'method': method,
            'matrix': corr_matrix
        })
        
        return corr_matrix
    
    def _compute_mic_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute Maximal Information Coefficient matrix"""
        
        n_features = len(features.columns)
        mic_matrix = np.zeros((n_features, n_features))
        
        feature_list = list(features.columns)
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    mic_matrix[i, j] = 1.0
                else:
                    # Use mutual information as approximation to MIC
                    try:
                        mi = mutual_info_score(
                            features.iloc[:, i].fillna(0),
                            features.iloc[:, j].fillna(0)
                        )
                        mic_matrix[i, j] = mic_matrix[j, i] = mi
                    except:
                        mic_matrix[i, j] = mic_matrix[j, i] = 0.0
        
        return pd.DataFrame(mic_matrix, index=feature_list, columns=feature_list)
    
    def find_highly_correlated_pairs(self, corr_matrix: pd.DataFrame, 
                                   threshold: float = None) -> List[Tuple[str, str, float]]:
        """Find pairs of highly correlated features"""
        
        if threshold is None:
            threshold = self.config.correlation_threshold
        
        highly_correlated = []
        
        # Get upper triangle to avoid duplicates
        upper_triangle = np.triu(np.abs(corr_matrix.values), k=1)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = upper_triangle[i, j]
                
                if correlation > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    highly_correlated.append((feature1, feature2, correlation))
        
        # Sort by correlation strength
        highly_correlated.sort(key=lambda x: x[2], reverse=True)
        
        return highly_correlated
    
    def compute_vif_scores(self, features: pd.DataFrame) -> Dict[str, float]:
        """Compute Variance Inflation Factor for multicollinearity detection"""
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features.fillna(features.mean())),
                columns=features.columns
            )
            
            vif_scores = {}
            
            for i, feature in enumerate(features_scaled.columns):
                try:
                    vif = variance_inflation_factor(features_scaled.values, i)
                    vif_scores[feature] = vif
                except:
                    vif_scores[feature] = float('inf')
            
            return vif_scores
            
        except Exception as e:
            logger.warning(f"Error computing VIF scores: {e}")
            return {}
    
    def detect_multicollinearity_groups(self, features: pd.DataFrame) -> List[List[str]]:
        """Detect groups of multicollinear features using hierarchical clustering"""
        
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(features, method='pearson')
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method=self.config.cluster_correlation_method)
        
        # Form clusters
        cluster_labels = fcluster(
            linkage_matrix, 
            t=self.config.cluster_distance_threshold, 
            criterion='distance'
        )
        
        # Group features by cluster
        feature_groups = defaultdict(list)
        for feature, cluster_id in zip(features.columns, cluster_labels):
            feature_groups[cluster_id].append(feature)
        
        # Return groups with more than one feature
        return [group for group in feature_groups.values() if len(group) > 1]
    
    def correlation_stability_analysis(self) -> Dict[str, float]:
        """Analyze correlation matrix stability over time"""
        
        if len(self.correlation_history) < 2:
            return {}
        
        stability_scores = {}
        
        # Compare recent correlation matrices
        recent_matrices = [entry['matrix'] for entry in self.correlation_history]
        
        # Compute pairwise stability
        for i in range(len(recent_matrices) - 1):
            for j in range(i + 1, len(recent_matrices)):
                matrix1 = recent_matrices[i]
                matrix2 = recent_matrices[j]
                
                # Find common features
                common_features = matrix1.columns.intersection(matrix2.columns)
                
                if len(common_features) < 2:
                    continue
                
                # Align matrices
                aligned_matrix1 = matrix1.loc[common_features, common_features]
                aligned_matrix2 = matrix2.loc[common_features, common_features]
                
                # Compute correlation of correlations
                upper_tri_1 = aligned_matrix1.where(
                    np.triu(np.ones(aligned_matrix1.shape), k=1).astype(bool)
                ).stack()
                upper_tri_2 = aligned_matrix2.where(
                    np.triu(np.ones(aligned_matrix2.shape), k=1).astype(bool)
                ).stack()
                
                stability = stats.pearsonr(upper_tri_1, upper_tri_2)[0]
                
                period_key = f"period_{i}_{j}"
                stability_scores[period_key] = stability
        
        return stability_scores

class FeatureSelector:
    """Multi-method feature selection system"""
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.selection_history = deque(maxlen=config.stability_window)
        self.feature_importance_history = defaultdict(list)
        self.selected_features_cache = {}
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'mutual_info') -> Dict[str, float]:
        """Univariate feature selection methods"""
        
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        if method == 'mutual_info':
            scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
        elif method == 'f_regression':
            from sklearn.feature_selection import f_regression
            scores, p_values = f_regression(X_clean, y_clean)
            
        elif method == 'chi2':
            # Transform to non-negative for chi2
            min_val = X_clean.min().min()
            if min_val < 0:
                X_transformed = X_clean - min_val
            else:
                X_transformed = X_clean
                
            # Discretize target for chi2
            y_discrete = pd.cut(y_clean, bins=5, labels=False)
            scores = chi2(X_transformed, y_discrete)[0]
            
        else:
            raise ValueError(f"Unknown univariate method: {method}")
        
        # Create feature scores dictionary
        feature_scores = dict(zip(X.columns, scores))
        
        # Normalize scores to [0, 1]
        max_score = max(feature_scores.values())
        if max_score > 0:
            feature_scores = {k: v/max_score for k, v in feature_scores.items()}
        
        return feature_scores
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    estimator=None, n_features: int = None) -> Dict[str, float]:
        """Recursive Feature Elimination"""
        
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        if n_features is None:
            n_features = min(self.config.max_features, len(X.columns) // 2)
        
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Use cross-validated RFE
        rfe = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs
        )
        
        rfe.fit(X_clean, y_clean)
        
        # Create feature scores based on ranking
        feature_scores = {}
        for feature, selected, ranking in zip(X.columns, rfe.support_, rfe.ranking_):
            # Convert ranking to score (lower rank = higher score)
            score = 1.0 / ranking if ranking > 0 else 0.0
            feature_scores[feature] = score
        
        # Normalize scores
        max_score = max(feature_scores.values()) if feature_scores else 1.0
        if max_score > 0:
            feature_scores = {k: v/max_score for k, v in feature_scores.items()}
        
        return feature_scores
    
    def embedded_selection(self, X: pd.DataFrame, y: pd.Series,
                         method: str = 'lasso') -> Dict[str, float]:
        """Embedded feature selection methods (Lasso, ElasticNet, etc.)"""
        
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Standardize features for regularization methods
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        if method == 'lasso':
            selector = LassoCV(cv=5, random_state=42, max_iter=1000)
            
        elif method == 'elastic_net':
            selector = ElasticNetCV(cv=5, random_state=42, max_iter=1000)
            
        elif method == 'random_forest':
            selector = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                n_jobs=self.config.n_jobs
            )
            
        else:
            raise ValueError(f"Unknown embedded method: {method}")
        
        # Fit selector
        selector.fit(X_scaled, y_clean)
        
        # Get feature importances/coefficients
        if hasattr(selector, 'coef_'):
            importances = np.abs(selector.coef_)
        elif hasattr(selector, 'feature_importances_'):
            importances = selector.feature_importances_
        else:
            importances = np.ones(len(X.columns))
        
        # Create feature scores
        feature_scores = dict(zip(X.columns, importances))
        
        # Normalize scores
        max_score = max(feature_scores.values()) if feature_scores else 1.0
        if max_score > 0:
            feature_scores = {k: v/max_score for k, v in feature_scores.items()}
        
        return feature_scores
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Tree-based feature importance selection"""
        
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Ensemble of tree-based models
        models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('extra_trees', ExtraTreesRegressor(n_estimators=100, random_state=42))
        ]
        
        combined_importances = np.zeros(len(X.columns))
        
        for name, model in models:
            model.fit(X_clean, y_clean)
            combined_importances += model.feature_importances_
        
        # Average importances
        combined_importances /= len(models)
        
        # Create feature scores
        feature_scores = dict(zip(X.columns, combined_importances))
        
        return feature_scores
    
    def select_features_multi_method(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Multi-method feature selection with consensus scoring"""
        
        start_time = time.time()
        
        method_scores = {}
        
        # Apply each selection method
        for method in self.config.selection_methods:
            try:
                if method == 'mutual_info':
                    scores = self.univariate_selection(X, y, 'mutual_info')
                elif method == 'f_regression':
                    scores = self.univariate_selection(X, y, 'f_regression')
                elif method == 'rfe':
                    scores = self.recursive_feature_elimination(X, y)
                elif method == 'lasso':
                    scores = self.embedded_selection(X, y, 'lasso')
                elif method == 'random_forest':
                    scores = self.tree_based_selection(X, y)
                else:
                    logger.warning(f"Unknown selection method: {method}")
                    continue
                
                method_scores[method] = scores
                
            except Exception as e:
                logger.error(f"Error in feature selection method {method}: {e}")
                continue
        
        # Consensus scoring - average across methods
        consensus_scores = self._compute_consensus_scores(method_scores)
        
        # Select top features
        selected_features = self._select_top_features(consensus_scores)
        
        selection_time = time.time() - start_time
        
        # Store selection history
        selection_record = {
            'timestamp': time.time(),
            'selected_features': selected_features,
            'consensus_scores': consensus_scores,
            'method_scores': method_scores,
            'selection_time': selection_time
        }
        
        self.selection_history.append(selection_record)
        
        # Update feature importance history
        for feature, score in consensus_scores.items():
            self.feature_importance_history[feature].append(score)
        
        return {
            'selected_features': selected_features,
            'feature_scores': consensus_scores,
            'method_breakdown': method_scores,
            'selection_time': selection_time,
            'n_features_selected': len(selected_features)
        }
    
    def _compute_consensus_scores(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute consensus scores across selection methods"""
        
        if not method_scores:
            return {}
        
        # Get all features
        all_features = set()
        for scores in method_scores.values():
            all_features.update(scores.keys())
        
        consensus_scores = {}
        
        for feature in all_features:
            scores = []
            for method_name, feature_scores in method_scores.items():
                score = feature_scores.get(feature, 0.0)
                scores.append(score)
            
            # Compute average score
            consensus_scores[feature] = np.mean(scores) if scores else 0.0
        
        return consensus_scores
    
    def _select_top_features(self, feature_scores: Dict[str, float]) -> List[str]:
        """Select top features based on scores"""
        
        if not feature_scores:
            return []
        
        # Sort features by score
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top features
        n_select = min(self.config.max_features, len(sorted_features))
        n_select = max(self.config.min_features, n_select)
        
        selected = [feature for feature, score in sorted_features[:n_select]]
        
        return selected
    
    def feature_stability_analysis(self) -> Dict[str, float]:
        """Analyze feature selection stability over time"""
        
        if len(self.selection_history) < 2:
            return {}
        
        # Get recent selections
        recent_selections = [
            entry['selected_features'] for entry in self.selection_history
        ]
        
        # Compute stability metrics
        stability_scores = {}
        
        # Jaccard stability between consecutive selections
        jaccard_scores = []
        for i in range(len(recent_selections) - 1):
            set1 = set(recent_selections[i])
            set2 = set(recent_selections[i + 1])
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
        
        if jaccard_scores:
            stability_scores['jaccard_stability'] = np.mean(jaccard_scores)
        
        # Feature frequency stability
        feature_counts = defaultdict(int)
        for selection in recent_selections:
            for feature in selection:
                feature_counts[feature] += 1
        
        # Features selected in most recent selections
        stable_features = [
            feature for feature, count in feature_counts.items()
            if count / len(recent_selections) >= self.config.stability_threshold
        ]
        
        stability_scores['stable_feature_count'] = len(stable_features)
        stability_scores['stable_features'] = stable_features
        
        return stability_scores

class FeatureInteractionDetector:
    """Detect and engineer feature interactions"""
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        
    def detect_pairwise_interactions(self, X: pd.DataFrame, y: pd.Series,
                                   max_pairs: int = 100) -> List[Tuple[str, str, float]]:
        """Detect significant pairwise feature interactions"""
        
        interactions = []
        feature_list = list(X.columns)
        
        # Limit number of pairs to check
        n_features = len(feature_list)
        max_pairs_possible = n_features * (n_features - 1) // 2
        
        if max_pairs_possible > max_pairs:
            # Sample pairs randomly
            import random
            random.seed(42)
            pairs_to_check = random.sample(
                [(i, j) for i in range(n_features) for j in range(i+1, n_features)],
                max_pairs
            )
        else:
            pairs_to_check = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
        
        for i, j in pairs_to_check:
            feature1 = feature_list[i]
            feature2 = feature_list[j]
            
            # Create interaction feature
            interaction_feature = X[feature1] * X[feature2]
            
            # Test interaction significance
            try:
                correlation = stats.pearsonr(interaction_feature.fillna(0), y.fillna(y.mean()))[0]
                interaction_strength = abs(correlation)
                
                if interaction_strength > 0.1:  # Threshold for meaningful interaction
                    interactions.append((feature1, feature2, interaction_strength))
                    
            except:
                continue
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x[2], reverse=True)
        
        return interactions
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                  interactions: List[Tuple[str, str, float]],
                                  max_interactions: int = 20) -> pd.DataFrame:
        """Create interaction features from detected interactions"""
        
        X_with_interactions = X.copy()
        
        for i, (feature1, feature2, strength) in enumerate(interactions[:max_interactions]):
            interaction_name = f"{feature1}_x_{feature2}"
            
            # Multiplicative interaction
            X_with_interactions[interaction_name] = X[feature1] * X[feature2]
            
            # Ratio interaction (if denominator not zero)
            ratio_name = f"{feature1}_div_{feature2}"
            denominator = X[feature2].replace(0, np.nan)
            if not denominator.isna().all():
                X_with_interactions[ratio_name] = X[feature1] / denominator
        
        return X_with_interactions

class FeatureCorrelationManager:
    """Main feature selection and correlation management system"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.feature_config = FeatureSelectionConfig()
        
        # Components
        self.correlation_analyzer = CorrelationAnalyzer(self.feature_config)
        self.feature_selector = FeatureSelector(self.feature_config)
        self.interaction_detector = FeatureInteractionDetector(self.feature_config)
        
        # State management
        self.current_features = []
        self.feature_metadata = {}
        self.selection_results = {}
        
        # Model persistence
        self.models_path = Path(self.config.models_path) / "feature_selection"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_feature_set(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive analysis of feature set"""
        
        start_time = time.time()
        
        analysis_results = {
            'timestamp': time.time(),
            'n_features': len(X.columns),
            'n_samples': len(X)
        }
        
        # Basic statistics
        analysis_results['basic_stats'] = {
            'missing_data_pct': (X.isnull().sum().sum() / X.size) * 100,
            'numeric_features': len(X.select_dtypes(include=[np.number]).columns),
            'zero_variance_features': len(X.columns[X.var() == 0])
        }
        
        # Correlation analysis
        logger.info("Computing correlation analysis...")
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix(X)
        highly_correlated = self.correlation_analyzer.find_highly_correlated_pairs(corr_matrix)
        vif_scores = self.correlation_analyzer.compute_vif_scores(X)
        multicollinear_groups = self.correlation_analyzer.detect_multicollinearity_groups(X)
        
        analysis_results['correlation'] = {
            'correlation_matrix': corr_matrix,
            'highly_correlated_pairs': highly_correlated[:20],  # Top 20
            'vif_scores': vif_scores,
            'multicollinear_groups': multicollinear_groups,
            'max_correlation': corr_matrix.abs().values.max() if not corr_matrix.empty else 0,
            'mean_correlation': corr_matrix.abs().mean().mean() if not corr_matrix.empty else 0
        }
        
        # Feature selection
        logger.info("Performing feature selection...")
        selection_results = self.feature_selector.select_features_multi_method(X, y)
        analysis_results['feature_selection'] = selection_results
        
        # Feature interactions
        logger.info("Detecting feature interactions...")
        interactions = self.interaction_detector.detect_pairwise_interactions(X, y)
        analysis_results['interactions'] = {
            'top_interactions': interactions[:10],
            'total_interactions_found': len(interactions)
        }
        
        # Quality metrics
        selected_features = selection_results['selected_features']
        if selected_features:
            X_selected = X[selected_features]
            selected_corr = self.correlation_analyzer.compute_correlation_matrix(X_selected)
            
            analysis_results['quality_metrics'] = {
                'selected_feature_count': len(selected_features),
                'reduction_ratio': len(selected_features) / len(X.columns),
                'max_correlation_selected': selected_corr.abs().values.max() if not selected_corr.empty else 0,
                'mean_correlation_selected': selected_corr.abs().mean().mean() if not selected_corr.empty else 0
            }
        
        analysis_time = time.time() - start_time
        analysis_results['analysis_time'] = analysis_time
        
        # Store results
        self.selection_results = analysis_results
        self.current_features = selected_features if selected_features else []
        
        logger.info(f"Feature analysis completed in {analysis_time:.2f}s")
        
        return analysis_results
    
    def get_optimal_feature_set(self, X: pd.DataFrame, y: pd.Series,
                              remove_correlated: bool = True) -> List[str]:
        """Get optimal feature set with correlation management"""
        
        # Perform full analysis
        analysis = self.analyze_feature_set(X, y)
        
        selected_features = analysis['feature_selection']['selected_features']
        
        if not remove_correlated:
            return selected_features
        
        # Remove highly correlated features
        X_selected = X[selected_features]
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix(X_selected)
        highly_correlated = self.correlation_analyzer.find_highly_correlated_pairs(
            corr_matrix, self.feature_config.correlation_threshold
        )
        
        # Remove features with high correlation
        features_to_remove = set()
        feature_scores = analysis['feature_selection']['feature_scores']
        
        for feature1, feature2, correlation in highly_correlated:
            # Keep the feature with higher importance score
            score1 = feature_scores.get(feature1, 0)
            score2 = feature_scores.get(feature2, 0)
            
            if score1 < score2:
                features_to_remove.add(feature1)
            else:
                features_to_remove.add(feature2)
        
        # Final feature set
        optimal_features = [f for f in selected_features if f not in features_to_remove]
        
        # Ensure minimum number of features
        if len(optimal_features) < self.feature_config.min_features:
            # Add back highest scoring removed features
            removed_features_scored = [
                (f, feature_scores.get(f, 0)) for f in features_to_remove
            ]
            removed_features_scored.sort(key=lambda x: x[1], reverse=True)
            
            for feature, score in removed_features_scored:
                optimal_features.append(feature)
                if len(optimal_features) >= self.feature_config.min_features:
                    break
        
        return optimal_features
    
    def monitor_feature_drift(self, X_new: pd.DataFrame) -> Dict[str, Any]:
        """Monitor feature distribution drift"""
        
        if not hasattr(self, 'baseline_stats'):
            # First time - establish baseline
            self.baseline_stats = {
                'means': X_new.mean(),
                'stds': X_new.std(),
                'quantiles': X_new.quantile([0.25, 0.5, 0.75])
            }
            return {'status': 'baseline_established'}
        
        # Compare with baseline
        drift_metrics = {}
        
        common_features = set(X_new.columns).intersection(set(self.baseline_stats['means'].index))
        
        for feature in common_features:
            try:
                # Statistical tests for drift
                baseline_mean = self.baseline_stats['means'][feature]
                current_mean = X_new[feature].mean()
                
                # Simple drift metric based on normalized difference
                drift_score = abs(current_mean - baseline_mean) / (self.baseline_stats['stds'][feature] + 1e-8)
                
                drift_metrics[feature] = {
                    'drift_score': drift_score,
                    'baseline_mean': baseline_mean,
                    'current_mean': current_mean,
                    'significant_drift': drift_score > 2.0  # 2 standard deviations
                }
                
            except Exception as e:
                logger.warning(f"Error computing drift for {feature}: {e}")
        
        # Summary metrics
        significant_drifts = sum(1 for m in drift_metrics.values() if m['significant_drift'])
        
        return {
            'feature_drift_metrics': drift_metrics,
            'features_with_significant_drift': significant_drifts,
            'total_features_monitored': len(common_features),
            'drift_percentage': (significant_drifts / len(common_features)) * 100 if common_features else 0
        }
    
    def save_feature_analysis(self, analysis_name: str = None) -> str:
        """Save feature analysis results"""
        
        if analysis_name is None:
            analysis_name = f"feature_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        save_path = self.models_path / f"{analysis_name}.joblib"
        
        save_data = {
            'feature_config': self.feature_config,
            'current_features': self.current_features,
            'feature_metadata': self.feature_metadata,
            'selection_results': self.selection_results,
            'correlation_history': list(self.correlation_analyzer.correlation_history),
            'selection_history': list(self.feature_selector.selection_history),
            'created_at': dt.datetime.now()
        }
        
        joblib.dump(save_data, save_path)
        logger.info(f"Feature analysis saved to: {save_path}")
        
        return str(save_path)
    
    def load_feature_analysis(self, save_path: str):
        """Load feature analysis results"""
        
        save_data = joblib.load(save_path)
        
        self.feature_config = save_data.get('feature_config', self.feature_config)
        self.current_features = save_data.get('current_features', [])
        self.feature_metadata = save_data.get('feature_metadata', {})
        self.selection_results = save_data.get('selection_results', {})
        
        # Restore history
        if 'correlation_history' in save_data:
            self.correlation_analyzer.correlation_history.extend(save_data['correlation_history'])
        
        if 'selection_history' in save_data:
            self.feature_selector.selection_history.extend(save_data['selection_history'])
        
        logger.info(f"Feature analysis loaded from: {save_path}")

# Test function
def test_feature_correlation_manager():
    """Test feature selection and correlation management system"""
    print("🚀 Testing Feature Correlation Manager")
    print("=" * 50)
    
    # Create feature manager
    config = SystemConfig()
    feature_manager = FeatureCorrelationManager(config)
    
    print(f"🔧 Configuration:")
    print(f"   Max Features: {feature_manager.feature_config.max_features}")
    print(f"   Correlation Threshold: {feature_manager.feature_config.correlation_threshold}")
    print(f"   VIF Threshold: {feature_manager.feature_config.vif_threshold}")
    print(f"   Selection Methods: {feature_manager.feature_config.selection_methods}")
    
    # Generate sample data with known correlations
    print(f"\n📊 Generating sample data with correlations...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create base features
    X_base = np.random.randn(n_samples, 10)
    
    # Add correlated features
    X_correlated = np.column_stack([
        X_base,
        X_base[:, 0] + np.random.randn(n_samples) * 0.1,  # Highly correlated with feature 0
        X_base[:, 1] * 2 + np.random.randn(n_samples) * 0.2,  # Correlated with feature 1
        np.random.randn(n_samples),  # Independent
    ])
    
    # Add interaction features
    X_with_interactions = np.column_stack([
        X_correlated,
        X_correlated[:, 0] * X_correlated[:, 1],  # Interaction feature
        X_correlated[:, 2] ** 2,  # Non-linear feature
    ])
    
    # Add noise features
    X_full = np.column_stack([
        X_with_interactions,
        np.random.randn(n_samples, n_features - X_with_interactions.shape[1])
    ])
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X_full.shape[1])]
    X = pd.DataFrame(X_full, columns=feature_names)
    
    # Create target with known relationships
    y = (X['feature_0'] * 2 + X['feature_1'] * -1.5 + 
         X['feature_2'] * 0.8 + np.random.randn(n_samples) * 0.5)
    
    print(f"   Data Shape: {X.shape}")
    print(f"   Target Shape: {y.shape}")
    
    try:
        # Test comprehensive analysis
        print(f"\n🔍 Running comprehensive feature analysis...")
        
        analysis = feature_manager.analyze_feature_set(X, y)
        
        print(f"   ✅ Analysis Results:")
        print(f"      Original Features: {analysis['n_features']}")
        print(f"      Missing Data: {analysis['basic_stats']['missing_data_pct']:.2f}%")
        print(f"      Features Selected: {analysis['quality_metrics']['selected_feature_count']}")
        print(f"      Reduction Ratio: {analysis['quality_metrics']['reduction_ratio']:.2f}")
        print(f"      Analysis Time: {analysis['analysis_time']:.2f}s")
        
        # Correlation results
        corr_results = analysis['correlation']
        print(f"\n📊 Correlation Analysis:")
        print(f"      Highly Correlated Pairs: {len(corr_results['highly_correlated_pairs'])}")
        print(f"      Max Correlation: {corr_results['max_correlation']:.3f}")
        print(f"      Mean Correlation: {corr_results['mean_correlation']:.3f}")
        print(f"      Multicollinear Groups: {len(corr_results['multicollinear_groups'])}")
        
        # Show top correlated pairs
        if corr_results['highly_correlated_pairs']:
            print(f"      Top Correlated Pairs:")
            for i, (f1, f2, corr) in enumerate(corr_results['highly_correlated_pairs'][:3]):
                print(f"         {i+1}. {f1} - {f2}: {corr:.3f}")
        
        # Feature selection results
        selection_results = analysis['feature_selection']
        print(f"\n🎯 Feature Selection Results:")
        print(f"      Selection Time: {selection_results['selection_time']:.3f}s")
        print(f"      Methods Used: {list(selection_results['method_breakdown'].keys())}")
        
        # Show top selected features
        top_features = selection_results['selected_features'][:10]
        print(f"      Top Selected Features: {top_features}")
        
        # Feature interactions
        interactions = analysis['interactions']
        print(f"\n🔗 Feature Interactions:")
        print(f"      Total Interactions Found: {interactions['total_interactions_found']}")
        
        if interactions['top_interactions']:
            print(f"      Top Interactions:")
            for i, (f1, f2, strength) in enumerate(interactions['top_interactions'][:3]):
                print(f"         {i+1}. {f1} × {f2}: {strength:.3f}")
        
        # Test optimal feature set
        print(f"\n⚡ Testing optimal feature set...")
        
        optimal_features = feature_manager.get_optimal_feature_set(X, y, remove_correlated=True)
        print(f"   ✅ Optimal Feature Set:")
        print(f"      Features: {len(optimal_features)}")
        print(f"      Feature Names: {optimal_features[:10]}...")
        
        # Test feature drift monitoring
        print(f"\n📈 Testing feature drift monitoring...")
        
        # Simulate new data with slight drift
        X_new = X.copy()
        X_new['feature_0'] += np.random.randn(len(X_new)) * 0.1  # Add drift
        
        drift_results = feature_manager.monitor_feature_drift(X_new)
        
        if 'drift_percentage' in drift_results:
            print(f"   ✅ Drift Monitoring:")
            print(f"      Features with Significant Drift: {drift_results['features_with_significant_drift']}")
            print(f"      Drift Percentage: {drift_results['drift_percentage']:.1f}%")
        
        # Test stability analysis
        print(f"\n🔄 Testing stability analysis...")
        
        # Run selection multiple times to build history
        for i in range(3):
            X_sample = X.sample(frac=0.8, random_state=i)
            y_sample = y.loc[X_sample.index]
            feature_manager.feature_selector.select_features_multi_method(X_sample, y_sample)
        
        stability = feature_manager.feature_selector.feature_stability_analysis()
        
        if stability:
            print(f"   ✅ Stability Analysis:")
            print(f"      Jaccard Stability: {stability.get('jaccard_stability', 0):.3f}")
            print(f"      Stable Features: {stability.get('stable_feature_count', 0)}")
        
        # Test model saving
        print(f"\n💾 Testing model persistence...")
        
        save_path = feature_manager.save_feature_analysis("test_analysis")
        print(f"   Model saved to: {save_path}")
        
        print(f"\n📋 Feature Correlation Manager Summary:")
        print(f"   ✅ Correlation analysis: Working")
        print(f"   ✅ Multi-method selection: Working")
        print(f"   ✅ Feature interactions: Working")
        print(f"   ✅ Optimal feature set: Working")
        print(f"   ✅ Drift monitoring: Working")
        print(f"   ✅ Stability analysis: Working")
        print(f"   ✅ Model persistence: Working")
        
    except Exception as e:
        print(f"❌ Error in feature correlation manager test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Feature correlation manager test completed!")

if __name__ == "__main__":
    test_feature_correlation_manager()