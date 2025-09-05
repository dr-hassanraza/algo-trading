#!/usr/bin/env python3
"""
HDBSCAN Clustering Engine for Algorithmic Trading
===============================================

Advanced clustering system for market regime detection and asset grouping
with parameter optimization and real-time processing capabilities.

Features:
- HDBSCAN clustering with parameter optimization
- Market regime detection based on volatility and correlation patterns
- Asset similarity clustering for pair trading strategies
- Real-time cluster updates with minimal latency
- Clustering validation and stability metrics
- Integration with existing ML pipeline
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import joblib
from pathlib import Path

# Clustering libraries
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.stats as stats

# Optimization
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid, ParameterSampler

from quant_system_config import SystemConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters"""
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_epsilon: float = 0.0
    alpha: float = 1.0
    cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'
    metric: str = 'euclidean'
    n_jobs: int = -1
    
    # Parameter optimization
    optimize_parameters: bool = True
    optimization_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    n_trials: int = 50
    
    # Validation
    min_silhouette_score: float = 0.3
    max_noise_ratio: float = 0.3
    stability_threshold: float = 0.7
    
    # Real-time processing
    update_frequency_minutes: int = 15
    batch_size: int = 1000
    max_latency_ms: int = 100

class ClusterValidator:
    """Validates clustering results and measures stability"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.validation_history = []
    
    def validate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          clusterer: hdbscan.HDBSCAN) -> Dict[str, float]:
        """Comprehensive validation of clustering results"""
        
        # Remove noise points for internal validation
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) < 2:
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': 0,
                'noise_ratio': 1.0,
                'stability_score': 0.0,
                'valid': False
            }
        
        X_clean = X[non_noise_mask]
        labels_clean = labels[non_noise_mask]
        
        n_clusters = len(np.unique(labels_clean))
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Silhouette Score
        try:
            if n_clusters > 1:
                silhouette = silhouette_score(X_clean, labels_clean)
            else:
                silhouette = 0.0
        except:
            silhouette = -1.0
        
        # Calinski-Harabasz Score
        try:
            if n_clusters > 1:
                ch_score = calinski_harabasz_score(X_clean, labels_clean)
            else:
                ch_score = 0.0
        except:
            ch_score = 0.0
        
        # Davies-Bouldin Score
        try:
            if n_clusters > 1:
                db_score = davies_bouldin_score(X_clean, labels_clean)
            else:
                db_score = float('inf')
        except:
            db_score = float('inf')
        
        # Cluster stability (using cluster probabilities if available)
        stability_score = 0.0
        if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
            stability_score = np.mean(clusterer.probabilities_[non_noise_mask])
        
        # Overall validation
        is_valid = (
            silhouette >= self.config.min_silhouette_score and
            noise_ratio <= self.config.max_noise_ratio and
            stability_score >= self.config.stability_threshold and
            n_clusters >= 2
        )
        
        validation_result = {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'stability_score': stability_score,
            'valid': is_valid,
            'timestamp': dt.datetime.now()
        }
        
        self.validation_history.append(validation_result)
        
        return validation_result
    
    def get_validation_trends(self, lookback_periods: int = 10) -> Dict[str, float]:
        """Analyze validation trends over time"""
        
        if len(self.validation_history) < 2:
            return {}
        
        recent_history = self.validation_history[-lookback_periods:]
        
        trends = {}
        for metric in ['silhouette_score', 'n_clusters', 'noise_ratio', 'stability_score']:
            values = [h[metric] for h in recent_history if not np.isnan(h[metric])]
            if len(values) >= 2:
                slope, _, r_value, _, _ = stats.linregress(range(len(values)), values)
                trends[f'{metric}_trend'] = slope
                trends[f'{metric}_r2'] = r_value ** 2
        
        return trends

class ParameterOptimizer:
    """Optimizes HDBSCAN parameters for best clustering performance"""
    
    def __init__(self, config: ClusteringConfig, validator: ClusterValidator):
        self.config = config
        self.validator = validator
        
        # Parameter search spaces
        self.param_space = {
            'min_cluster_size': [3, 5, 7, 10, 15, 20],
            'min_samples': [1, 3, 5, 7, 10],
            'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3, 0.5],
            'alpha': [0.5, 1.0, 1.5, 2.0],
            'cluster_selection_method': ['eom', 'leaf']
        }
    
    def objective_function(self, params: Dict[str, Any], X: np.ndarray) -> float:
        """Objective function for parameter optimization"""
        
        try:
            # Create clusterer with parameters
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(params['min_cluster_size']),
                min_samples=int(params['min_samples']),
                cluster_selection_epsilon=float(params['cluster_selection_epsilon']),
                alpha=float(params['alpha']),
                cluster_selection_method=params['cluster_selection_method'],
                metric=self.config.metric,
                n_jobs=self.config.n_jobs
            )
            
            # Fit clustering
            labels = clusterer.fit_predict(X)
            
            # Validate results
            validation = self.validator.validate_clustering(X, labels, clusterer)
            
            # Multi-objective score (lower is better for minimization)
            score = (
                -validation['silhouette_score'] +  # Maximize silhouette
                validation['noise_ratio'] +         # Minimize noise
                -validation['stability_score'] +    # Maximize stability
                abs(validation['n_clusters'] - 5) * 0.1  # Prefer ~5 clusters
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return float('inf')
    
    def grid_search_optimization(self, X: np.ndarray) -> Dict[str, Any]:
        """Grid search parameter optimization"""
        
        best_score = float('inf')
        best_params = None
        best_validation = None
        
        # Generate parameter grid
        param_grid = list(ParameterGrid(self.param_space))
        
        logger.info(f"Starting grid search with {len(param_grid)} combinations...")
        
        for i, params in enumerate(param_grid):
            if i % 20 == 0:
                logger.info(f"Testing parameter combination {i+1}/{len(param_grid)}")
            
            score = self.objective_function(params, X)
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
        
        # Validate best parameters
        if best_params:
            # Create clusterer with compatibility handling
            try:
                clusterer = hdbscan.HDBSCAN(**best_params, metric=self.config.metric, n_jobs=self.config.n_jobs)
            except TypeError:
                clusterer = hdbscan.HDBSCAN(**best_params, metric=self.config.metric)
            labels = clusterer.fit_predict(X)
            best_validation = self.validator.validate_clustering(X, labels, clusterer)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'validation': best_validation,
            'method': 'grid_search'
        }
    
    def random_search_optimization(self, X: np.ndarray) -> Dict[str, Any]:
        """Random search parameter optimization"""
        
        best_score = float('inf')
        best_params = None
        best_validation = None
        
        # Generate random parameter samples
        param_list = list(ParameterSampler(self.param_space, n_iter=self.config.n_trials, random_state=42))
        
        logger.info(f"Starting random search with {len(param_list)} trials...")
        
        for i, params in enumerate(param_list):
            if i % 10 == 0:
                logger.info(f"Testing trial {i+1}/{len(param_list)}")
            
            score = self.objective_function(params, X)
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
        
        # Validate best parameters
        if best_params:
            # Create clusterer with compatibility handling
            try:
                clusterer = hdbscan.HDBSCAN(**best_params, metric=self.config.metric, n_jobs=self.config.n_jobs)
            except TypeError:
                clusterer = hdbscan.HDBSCAN(**best_params, metric=self.config.metric)
            labels = clusterer.fit_predict(X)
            best_validation = self.validator.validate_clustering(X, labels, clusterer)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'validation': best_validation,
            'method': 'random_search'
        }
    
    def optimize_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Main parameter optimization method"""
        
        if not self.config.optimize_parameters:
            # Use default parameters
            default_params = {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'alpha': self.config.alpha,
                'cluster_selection_method': self.config.cluster_selection_method
            }
            
            clusterer = hdbscan.HDBSCAN(**default_params, metric=self.config.metric, n_jobs=self.config.n_jobs)
            labels = clusterer.fit_predict(X)
            validation = self.validator.validate_clustering(X, labels, clusterer)
            
            return {
                'best_params': default_params,
                'best_score': 0.0,
                'validation': validation,
                'method': 'default'
            }
        
        if self.config.optimization_method == 'grid_search':
            return self.grid_search_optimization(X)
        elif self.config.optimization_method == 'random_search':
            return self.random_search_optimization(X)
        else:
            logger.warning(f"Unknown optimization method: {self.config.optimization_method}")
            return self.grid_search_optimization(X)

class ClusteringEngine:
    """Main clustering engine with HDBSCAN and optimization"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.clustering_config = ClusteringConfig()
        
        # Components
        self.validator = ClusterValidator(self.clustering_config)
        self.optimizer = ParameterOptimizer(self.clustering_config, self.validator)
        
        # State
        self.clusterer = None
        self.scaler = RobustScaler()
        self.pca_reducer = None
        self.current_labels = None
        self.current_features = None
        self.optimization_results = {}
        
        # Model persistence
        self.models_path = Path(self.config.models_path) / "clustering"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = []
        
    def preprocess_features(self, features: np.ndarray, 
                          apply_pca: bool = False, 
                          n_components: float = 0.95) -> np.ndarray:
        """Preprocess features for clustering"""
        
        start_time = time.time()
        
        # Handle NaNs
        if np.isnan(features).any():
            features = pd.DataFrame(features).fillna(method='bfill').fillna(method='ffill').values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Optional PCA for dimensionality reduction
        if apply_pca and features_scaled.shape[1] > 10:
            if self.pca_reducer is None:
                self.pca_reducer = PCA(n_components=n_components, random_state=42)
            
            features_scaled = self.pca_reducer.fit_transform(features_scaled)
            logger.info(f"PCA reduced dimensions from {features.shape[1]} to {features_scaled.shape[1]}")
        
        preprocessing_time = (time.time() - start_time) * 1000
        logger.debug(f"Feature preprocessing took {preprocessing_time:.2f}ms")
        
        return features_scaled
    
    def fit_clustering(self, features: np.ndarray, 
                      optimize_params: bool = True,
                      apply_pca: bool = False) -> Dict[str, Any]:
        """Fit HDBSCAN clustering with parameter optimization"""
        
        start_time = time.time()
        
        # Preprocess features
        features_processed = self.preprocess_features(features, apply_pca)
        self.current_features = features_processed
        
        # Parameter optimization
        if optimize_params:
            logger.info("Optimizing HDBSCAN parameters...")
            self.optimization_results = self.optimizer.optimize_parameters(features_processed)
            best_params = self.optimization_results['best_params']
        else:
            best_params = {
                'min_cluster_size': self.clustering_config.min_cluster_size,
                'min_samples': self.clustering_config.min_samples,
                'cluster_selection_epsilon': self.clustering_config.cluster_selection_epsilon,
                'alpha': self.clustering_config.alpha,
                'cluster_selection_method': self.clustering_config.cluster_selection_method
            }
        
        # Create and fit clusterer (remove n_jobs for compatibility)
        clusterer_params = best_params.copy()
        clusterer_params['metric'] = self.clustering_config.metric
        
        # Remove n_jobs if it causes compatibility issues
        try:
            self.clusterer = hdbscan.HDBSCAN(**clusterer_params, n_jobs=self.clustering_config.n_jobs)
        except TypeError:
            # Fallback without n_jobs parameter for older HDBSCAN versions
            self.clusterer = hdbscan.HDBSCAN(**clusterer_params)
        
        self.current_labels = self.clusterer.fit_predict(features_processed)
        
        # Validate clustering
        validation_results = self.validator.validate_clustering(
            features_processed, self.current_labels, self.clusterer
        )
        
        total_time = (time.time() - start_time) * 1000
        
        results = {
            'labels': self.current_labels,
            'n_clusters': len(np.unique(self.current_labels[self.current_labels != -1])),
            'n_noise': np.sum(self.current_labels == -1),
            'optimization_results': self.optimization_results,
            'validation_results': validation_results,
            'parameters_used': best_params,
            'processing_time_ms': total_time,
            'features_shape': features_processed.shape
        }
        
        # Track performance
        self.performance_metrics.append({
            'timestamp': dt.datetime.now(),
            'processing_time_ms': total_time,
            'n_samples': len(features),
            'n_features': features.shape[1],
            'n_clusters': results['n_clusters'],
            'silhouette_score': validation_results.get('silhouette_score', 0)
        })
        
        logger.info(f"Clustering completed in {total_time:.2f}ms - {results['n_clusters']} clusters, "
                   f"{results['n_noise']} noise points")
        
        return results
    
    def predict_clusters(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        
        if self.clusterer is None:
            raise ValueError("Clusterer not fitted. Call fit_clustering first.")
        
        start_time = time.time()
        
        # Preprocess features
        features_processed = self.scaler.transform(features)
        if self.pca_reducer is not None:
            features_processed = self.pca_reducer.transform(features_processed)
        
        # HDBSCAN doesn't have direct predict method, use approximate_predict
        try:
            labels, strengths = hdbscan.approximate_predict(self.clusterer, features_processed)
        except:
            # Fallback: find closest cluster centroids
            labels = self._predict_via_centroids(features_processed)
        
        prediction_time = (time.time() - start_time) * 1000
        logger.debug(f"Cluster prediction took {prediction_time:.2f}ms")
        
        return labels
    
    def _predict_via_centroids(self, features: np.ndarray) -> np.ndarray:
        """Predict clusters using centroid distances (fallback method)"""
        
        # Calculate cluster centroids
        unique_labels = np.unique(self.current_labels[self.current_labels != -1])
        centroids = []
        
        for label in unique_labels:
            cluster_mask = self.current_labels == label
            centroid = np.mean(self.current_features[cluster_mask], axis=0)
            centroids.append(centroid)
        
        if not centroids:
            return np.full(len(features), -1)  # All noise
        
        centroids = np.array(centroids)
        
        # Find closest centroid for each point
        distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        
        return unique_labels[closest_centroids]
    
    def detect_market_regimes(self, volatility_features: np.ndarray, 
                            correlation_features: np.ndarray) -> Dict[str, Any]:
        """Detect market regimes using volatility and correlation patterns"""
        
        # Combine volatility and correlation features
        regime_features = np.column_stack([volatility_features, correlation_features])
        
        # Fit clustering
        results = self.fit_clustering(regime_features, optimize_params=True)
        
        # Interpret regimes
        regime_labels = results['labels']
        regimes = {}
        
        for regime_id in np.unique(regime_labels[regime_labels != -1]):
            regime_mask = regime_labels == regime_id
            
            avg_vol = np.mean(volatility_features[regime_mask])
            avg_corr = np.mean(correlation_features[regime_mask])
            
            # Classify regime type
            if avg_vol > np.median(volatility_features) and avg_corr > np.median(correlation_features):
                regime_type = "Crisis"
            elif avg_vol > np.median(volatility_features):
                regime_type = "High_Volatility"
            elif avg_corr > np.median(correlation_features):
                regime_type = "High_Correlation"
            else:
                regime_type = "Normal"
            
            regimes[regime_id] = {
                'type': regime_type,
                'avg_volatility': avg_vol,
                'avg_correlation': avg_corr,
                'n_observations': np.sum(regime_mask),
                'stability': results['validation_results'].get('stability_score', 0)
            }
        
        return {
            'regime_labels': regime_labels,
            'regimes': regimes,
            'clustering_results': results
        }
    
    def cluster_assets(self, returns_matrix: np.ndarray, 
                      asset_names: List[str]) -> Dict[str, Any]:
        """Cluster assets based on return patterns"""
        
        # Calculate correlation-based features
        correlation_matrix = np.corrcoef(returns_matrix.T)
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Use distance matrix as features
        features = distance_matrix
        
        # Fit clustering
        results = self.fit_clustering(features, optimize_params=True)
        
        # Group assets by clusters
        asset_clusters = {}
        cluster_labels = results['labels']
        
        for i, asset in enumerate(asset_names):
            cluster_id = cluster_labels[i]
            
            if cluster_id not in asset_clusters:
                asset_clusters[cluster_id] = []
            
            asset_clusters[cluster_id].append(asset)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, assets in asset_clusters.items():
            if cluster_id == -1:  # Noise cluster
                continue
            
            asset_indices = [asset_names.index(asset) for asset in assets]
            cluster_returns = returns_matrix[:, asset_indices]
            
            cluster_stats[cluster_id] = {
                'assets': assets,
                'n_assets': len(assets),
                'avg_return': np.mean(cluster_returns),
                'avg_volatility': np.std(cluster_returns),
                'avg_correlation': np.mean(correlation_matrix[np.ix_(asset_indices, asset_indices)]),
                'sharpe_ratio': np.mean(cluster_returns) / (np.std(cluster_returns) + 1e-8)
            }
        
        return {
            'asset_clusters': asset_clusters,
            'cluster_stats': cluster_stats,
            'clustering_results': results,
            'correlation_matrix': correlation_matrix
        }
    
    def update_clustering_realtime(self, new_features: np.ndarray, 
                                 batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Update clustering with new data in real-time"""
        
        start_time = time.time()
        
        if batch_size is None:
            batch_size = self.clustering_config.batch_size
        
        # Process in batches if data is large
        if len(new_features) > batch_size:
            logger.info(f"Processing {len(new_features)} samples in batches of {batch_size}")
            
            batch_results = []
            for i in range(0, len(new_features), batch_size):
                batch_end = min(i + batch_size, len(new_features))
                batch_features = new_features[i:batch_end]
                
                batch_labels = self.predict_clusters(batch_features)
                batch_results.append(batch_labels)
            
            new_labels = np.concatenate(batch_results)
        else:
            new_labels = self.predict_clusters(new_features)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Check if latency requirement is met
        latency_met = processing_time <= self.clustering_config.max_latency_ms
        
        if not latency_met:
            logger.warning(f"Real-time update exceeded latency requirement: {processing_time:.2f}ms > {self.clustering_config.max_latency_ms}ms")
        
        return {
            'labels': new_labels,
            'processing_time_ms': processing_time,
            'latency_requirement_met': latency_met,
            'batch_size_used': batch_size,
            'n_samples_processed': len(new_features)
        }
    
    def save_model(self, model_name: str = None) -> str:
        """Save clustering model and configuration"""
        
        if model_name is None:
            model_name = f"clustering_model_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.models_path / f"{model_name}.joblib"
        
        model_data = {
            'clusterer': self.clusterer,
            'scaler': self.scaler,
            'pca_reducer': self.pca_reducer,
            'current_labels': self.current_labels,
            'current_features': self.current_features,
            'optimization_results': self.optimization_results,
            'clustering_config': self.clustering_config,
            'performance_metrics': self.performance_metrics,
            'created_at': dt.datetime.now(),
            'model_name': model_name
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Clustering model saved to: {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """Load clustering model and configuration"""
        
        model_data = joblib.load(model_path)
        
        self.clusterer = model_data['clusterer']
        self.scaler = model_data['scaler']
        self.pca_reducer = model_data.get('pca_reducer')
        self.current_labels = model_data.get('current_labels')
        self.current_features = model_data.get('current_features')
        self.optimization_results = model_data.get('optimization_results', {})
        self.performance_metrics = model_data.get('performance_metrics', [])
        
        if 'clustering_config' in model_data:
            self.clustering_config = model_data['clustering_config']
            self.validator = ClusterValidator(self.clustering_config)
            self.optimizer = ParameterOptimizer(self.clustering_config, self.validator)
        
        logger.info(f"Clustering model loaded from: {model_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of clustering operations"""
        
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-10:]  # Last 10 operations
        
        processing_times = [m['processing_time_ms'] for m in recent_metrics]
        silhouette_scores = [m['silhouette_score'] for m in recent_metrics if m['silhouette_score'] > -1]
        
        return {
            'total_operations': len(self.performance_metrics),
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'avg_silhouette_score': np.mean(silhouette_scores) if silhouette_scores else 0,
            'latency_violations': sum(1 for t in processing_times if t > self.clustering_config.max_latency_ms),
            'latest_operation': recent_metrics[-1] if recent_metrics else None
        }

# Test function
def test_clustering_engine():
    """Test clustering engine functionality"""
    print("🚀 Testing HDBSCAN Clustering Engine")
    print("=" * 50)
    
    # Create clustering engine
    config = SystemConfig()
    clustering_engine = ClusteringEngine(config)
    
    print(f"🔧 Clustering Configuration:")
    print(f"   Min Cluster Size: {clustering_engine.clustering_config.min_cluster_size}")
    print(f"   Optimization Method: {clustering_engine.clustering_config.optimization_method}")
    print(f"   Max Latency: {clustering_engine.clustering_config.max_latency_ms}ms")
    
    # Generate sample data
    print(f"\n📊 Generating sample data...")
    
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    
    # Create synthetic clustered data
    cluster_centers = np.random.randn(4, n_features) * 2
    data_points = []
    
    for i in range(n_samples):
        center_idx = np.random.choice(4)
        point = cluster_centers[center_idx] + np.random.randn(n_features) * 0.5
        data_points.append(point)
    
    sample_features = np.array(data_points)
    
    print(f"   Features Shape: {sample_features.shape}")
    
    try:
        # Test clustering with optimization
        print(f"\n🧠 Testing clustering with parameter optimization...")
        
        results = clustering_engine.fit_clustering(
            sample_features, 
            optimize_params=True, 
            apply_pca=True
        )
        
        print(f"   ✅ Clustering Results:")
        print(f"      Clusters Found: {results['n_clusters']}")
        print(f"      Noise Points: {results['n_noise']}")
        print(f"      Processing Time: {results['processing_time_ms']:.2f}ms")
        print(f"      Silhouette Score: {results['validation_results']['silhouette_score']:.3f}")
        print(f"      Valid Clustering: {results['validation_results']['valid']}")
        
        if results['optimization_results']:
            print(f"      Best Parameters: {results['optimization_results']['best_params']}")
        
        # Test real-time prediction
        print(f"\n⚡ Testing real-time prediction...")
        
        new_samples = np.random.randn(50, n_features)
        rt_results = clustering_engine.update_clustering_realtime(new_samples)
        
        print(f"   ✅ Real-time Results:")
        print(f"      Samples Processed: {rt_results['n_samples_processed']}")
        print(f"      Processing Time: {rt_results['processing_time_ms']:.2f}ms")
        print(f"      Latency Requirement Met: {rt_results['latency_requirement_met']}")
        
        # Test market regime detection
        print(f"\n📈 Testing market regime detection...")
        
        volatility_features = np.random.lognormal(0, 0.5, n_samples)
        correlation_features = np.random.uniform(0.1, 0.9, n_samples)
        
        regime_results = clustering_engine.detect_market_regimes(
            volatility_features, correlation_features
        )
        
        print(f"   ✅ Market Regime Results:")
        print(f"      Regimes Detected: {len(regime_results['regimes'])}")
        
        for regime_id, regime_info in regime_results['regimes'].items():
            print(f"      Regime {regime_id}: {regime_info['type']} "
                  f"(Vol: {regime_info['avg_volatility']:.3f}, "
                  f"Corr: {regime_info['avg_correlation']:.3f})")
        
        # Test asset clustering
        print(f"\n🏢 Testing asset clustering...")
        
        asset_names = ['UBL', 'MCB', 'HBL', 'FFC', 'ENGRO', 'LUCK', 'PPL', 'OGDC']
        returns_matrix = np.random.randn(100, len(asset_names)) * 0.02  # 2% daily vol
        
        asset_results = clustering_engine.cluster_assets(returns_matrix, asset_names)
        
        print(f"   ✅ Asset Clustering Results:")
        print(f"      Asset Clusters: {len(asset_results['cluster_stats'])}")
        
        for cluster_id, stats in asset_results['cluster_stats'].items():
            print(f"      Cluster {cluster_id}: {stats['assets']} "
                  f"(Sharpe: {stats['sharpe_ratio']:.3f})")
        
        # Test model saving and loading
        print(f"\n💾 Testing model persistence...")
        
        model_path = clustering_engine.save_model("test_clustering")
        print(f"   Model saved to: {model_path}")
        
        # Test performance summary
        performance = clustering_engine.get_performance_summary()
        print(f"\n📊 Performance Summary:")
        print(f"   Total Operations: {performance.get('total_operations', 0)}")
        print(f"   Avg Processing Time: {performance.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"   Latency Violations: {performance.get('latency_violations', 0)}")
        
    except Exception as e:
        print(f"❌ Error in clustering engine test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Clustering engine test completed!")

if __name__ == "__main__":
    test_clustering_engine()