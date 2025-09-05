#!/usr/bin/env python3
"""
Bayesian Statistics Engine for Algorithmic Trading
================================================

Production-ready Bayesian inference system for quantitative trading applications
implementing institutional-grade probabilistic modeling and uncertainty quantification.

Features:
- Bayesian linear regression with conjugate priors
- Hierarchical Bayesian models for multi-asset analysis
- Variational inference for scalable posterior approximation
- Monte Carlo methods for complex posterior sampling
- Bayesian model selection and averaging
- Online Bayesian learning with streaming data
- Uncertainty quantification and confidence intervals
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import joblib
from pathlib import Path

# Bayesian and statistical libraries
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import gammaln, digamma
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Monte Carlo sampling
try:
    import pymc3 as pm
    import theano.tensor as tt
    PYMC3_AVAILABLE = True
except ImportError:
    try:
        import pymc as pm
        import pytensor.tensor as tt
        PYMC3_AVAILABLE = True
    except ImportError:
        PYMC3_AVAILABLE = False
        warnings.warn("PyMC not available. Limited Bayesian functionality.")

from quant_system_config import SystemConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class BayesianConfig:
    """Configuration for Bayesian inference parameters"""
    
    # Prior specifications
    alpha_prior: float = 1.0  # Prior precision for noise
    beta_prior: float = 1.0   # Prior precision for weights
    
    # Variational inference
    max_iter: int = 1000
    tol: float = 1e-6
    learning_rate: float = 0.01
    
    # Monte Carlo settings
    n_samples: int = 2000
    n_burnin: int = 500
    n_chains: int = 2
    
    # Model selection
    use_model_averaging: bool = True
    max_models: int = 10
    
    # Online learning
    forgetting_factor: float = 0.99
    update_frequency: int = 10
    
    # Confidence intervals
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95])
    
    # Performance
    parallel_chains: bool = True
    use_gpu: bool = False

class BayesianLinearRegression:
    """Bayesian linear regression with conjugate priors"""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        
        # Conjugate prior parameters
        self.alpha = config.alpha_prior
        self.beta = config.beta_prior
        
        # Posterior parameters (will be updated)
        self.S_N = None  # Posterior covariance
        self.m_N = None  # Posterior mean
        self.alpha_N = None  # Posterior precision (noise)
        self.beta_N = None  # Posterior precision (weights)
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Model evidence for comparison
        self.log_evidence = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit Bayesian linear regression using conjugate priors"""
        
        start_time = time.time()
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        n, p = X_scaled.shape
        
        # Add bias term
        X_design = np.column_stack([np.ones(n), X_scaled])
        
        # Prior parameters
        S_0_inv = self.beta * np.eye(p + 1)
        m_0 = np.zeros(p + 1)
        
        # Posterior parameters (conjugate updates)
        S_N_inv = S_0_inv + self.alpha * X_design.T @ X_design
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.S_N @ (S_0_inv @ m_0 + self.alpha * X_design.T @ y)
        
        # Update precision parameters
        self.alpha_N = self.alpha + n / 2
        
        residual = y - X_design @ self.m_N
        self.beta_N = self.beta + 0.5 * (
            (y - X_design @ self.m_N).T @ (y - X_design @ self.m_N) +
            (self.m_N - m_0).T @ S_0_inv @ (self.m_N - m_0)
        )
        
        # Compute log marginal likelihood (model evidence)
        self.log_evidence = self._compute_log_evidence(X_design, y, S_0_inv, m_0)
        
        self.fitted = True
        
        fitting_time = (time.time() - start_time) * 1000
        
        return {
            'posterior_mean': self.m_N,
            'posterior_covariance': self.S_N,
            'alpha_posterior': self.alpha_N,
            'beta_posterior': self.beta_N,
            'log_evidence': self.log_evidence,
            'fitting_time_ms': fitting_time,
            'n_features': p
        }
    
    def _compute_log_evidence(self, X: np.ndarray, y: np.ndarray, 
                            S_0_inv: np.ndarray, m_0: np.ndarray) -> float:
        """Compute log marginal likelihood for model comparison"""
        
        n, p = X.shape
        
        try:
            # Terms for log evidence computation
            log_det_S0 = np.linalg.slogdet(S_0_inv)[1]
            log_det_SN = np.linalg.slogdet(self.S_N)[1]
            
            quadratic_term = (
                self.beta * np.sum(y**2) + 
                m_0.T @ S_0_inv @ m_0 -
                self.m_N.T @ np.linalg.inv(self.S_N) @ self.m_N
            )
            
            log_evidence = (
                0.5 * log_det_S0 - 0.5 * log_det_SN +
                0.5 * p * np.log(self.beta / (2 * np.pi)) -
                0.5 * n * np.log(2 * np.pi) -
                0.5 * self.alpha * quadratic_term
            )
            
            return log_evidence
            
        except:
            return -np.inf
    
    def predict(self, X: np.ndarray, 
               return_uncertainty: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict with uncertainty quantification"""
        
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        # Standardize features
        X_scaled = self.scaler.transform(X)
        X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        # Predictive mean
        y_mean = X_design @ self.m_N
        
        if not return_uncertainty:
            return y_mean
        
        # Predictive variance
        predictive_variance = []
        for x in X_design:
            var = (1 / self.alpha_N) * (1 + x.T @ self.S_N @ x)
            predictive_variance.append(var)
        
        y_std = np.sqrt(predictive_variance)
        
        return y_mean, y_std
    
    def get_confidence_intervals(self, X: np.ndarray) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals for predictions"""
        
        y_mean, y_std = self.predict(X, return_uncertainty=True)
        
        intervals = {}
        for confidence in self.config.confidence_levels:
            alpha = 1 - confidence
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = y_mean - z_score * y_std
            upper = y_mean + z_score * y_std
            
            intervals[confidence] = (lower, upper)
        
        return intervals
    
    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """Sample from posterior distribution of weights"""
        
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        # Sample weights from multivariate normal
        samples = np.random.multivariate_normal(self.m_N, self.S_N, n_samples)
        
        return samples

class HierarchicalBayesianModel:
    """Hierarchical Bayesian model for multi-asset analysis"""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.group_models = {}
        self.global_params = {}
        self.fitted = False
        
    def fit(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Fit hierarchical model with group-specific and global parameters"""
        
        start_time = time.time()
        
        all_X = []
        all_y = []
        group_sizes = {}
        
        # Collect all data
        for group_name, (X, y) in data_dict.items():
            all_X.append(X)
            all_y.append(y)
            group_sizes[group_name] = len(y)
        
        # Fit global model first
        X_global = np.vstack(all_X)
        y_global = np.concatenate(all_y)
        
        global_model = BayesianLinearRegression(self.config)
        global_results = global_model.fit(X_global, y_global)
        
        self.global_params = {
            'model': global_model,
            'results': global_results
        }
        
        # Fit group-specific models with informed priors
        for group_name, (X, y) in data_dict.items():
            # Use global parameters as informed priors
            group_config = BayesianConfig()
            group_config.alpha_prior = global_results['alpha_posterior']
            group_config.beta_prior = global_results['beta_posterior']
            
            group_model = BayesianLinearRegression(group_config)
            group_results = group_model.fit(X, y)
            
            self.group_models[group_name] = {
                'model': group_model,
                'results': group_results,
                'n_observations': len(y)
            }
        
        self.fitted = True
        
        fitting_time = (time.time() - start_time) * 1000
        
        return {
            'global_results': global_results,
            'group_results': {name: info['results'] for name, info in self.group_models.items()},
            'fitting_time_ms': fitting_time,
            'n_groups': len(data_dict),
            'total_observations': len(y_global)
        }
    
    def predict(self, X: np.ndarray, group_name: str = None, 
               use_hierarchical: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using hierarchical structure"""
        
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if group_name and group_name in self.group_models:
            # Use group-specific model
            group_model = self.group_models[group_name]['model']
            return group_model.predict(X, return_uncertainty=True)
        
        elif use_hierarchical and group_name:
            # Combine global and group predictions
            global_pred, global_std = self.global_params['model'].predict(X, return_uncertainty=True)
            
            if group_name in self.group_models:
                group_pred, group_std = self.group_models[group_name]['model'].predict(X, return_uncertainty=True)
                
                # Weighted combination based on group sample size
                group_n = self.group_models[group_name]['n_observations']
                global_n = self.global_params['results']['n_features']
                
                weight_group = group_n / (group_n + global_n)
                weight_global = 1 - weight_group
                
                combined_pred = weight_group * group_pred + weight_global * global_pred
                combined_std = np.sqrt(weight_group * group_std**2 + weight_global * global_std**2)
                
                return combined_pred, combined_std
        
        # Fallback to global model
        return self.global_params['model'].predict(X, return_uncertainty=True)

class VariationalInference:
    """Variational inference for scalable Bayesian learning"""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.q_params = {}  # Variational parameters
        self.elbo_history = []
        
    def fit_variational(self, X: np.ndarray, y: np.ndarray, 
                       model_type: str = 'linear') -> Dict[str, Any]:
        """Fit using variational inference"""
        
        start_time = time.time()
        n, p = X.shape
        
        # Initialize variational parameters
        self.q_params = {
            'mu_w': np.zeros(p + 1),  # Mean of weight posterior
            'sigma_w': np.ones(p + 1),  # Std of weight posterior
            'alpha_q': 1.0,  # Precision posterior shape
            'beta_q': 1.0   # Precision posterior rate
        }
        
        # Add bias term
        X_design = np.column_stack([np.ones(n), X])
        
        # Variational optimization
        for iteration in range(self.config.max_iter):
            old_params = {k: v.copy() if isinstance(v, np.ndarray) else v 
                         for k, v in self.q_params.items()}
            
            # Update weight parameters
            precision_matrix = np.diag(1 / self.q_params['sigma_w']**2) + \
                              self.q_params['alpha_q'] * X_design.T @ X_design
            
            covariance_matrix = np.linalg.inv(precision_matrix)
            self.q_params['mu_w'] = covariance_matrix @ (self.q_params['alpha_q'] * X_design.T @ y)
            self.q_params['sigma_w'] = np.sqrt(np.diag(covariance_matrix))
            
            # Update precision parameters
            residual = y - X_design @ self.q_params['mu_w']
            self.q_params['alpha_q'] = self.config.alpha_prior + n / 2
            self.q_params['beta_q'] = self.config.beta_prior + 0.5 * np.sum(residual**2)
            
            # Compute ELBO
            elbo = self._compute_elbo(X_design, y)
            self.elbo_history.append(elbo)
            
            # Check convergence
            param_change = np.sum([np.sum(np.abs(self.q_params[k] - old_params[k]))
                                  for k in self.q_params.keys() 
                                  if isinstance(self.q_params[k], np.ndarray)])
            
            if param_change < self.config.tol:
                logger.info(f"Variational inference converged at iteration {iteration}")
                break
        
        fitting_time = (time.time() - start_time) * 1000
        
        return {
            'q_params': self.q_params,
            'elbo_history': self.elbo_history,
            'converged': param_change < self.config.tol,
            'n_iterations': iteration + 1,
            'fitting_time_ms': fitting_time,
            'final_elbo': elbo
        }
    
    def _compute_elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Evidence Lower Bound (ELBO)"""
        
        n, p = X.shape
        
        # Likelihood term
        expected_log_likelihood = (
            -0.5 * n * np.log(2 * np.pi) +
            0.5 * n * (digamma(self.q_params['alpha_q']) - np.log(self.q_params['beta_q'])) -
            0.5 * self.q_params['alpha_q'] / self.q_params['beta_q'] * (
                np.sum((y - X @ self.q_params['mu_w'])**2) +
                np.sum(self.q_params['sigma_w']**2 * np.diag(X.T @ X))
            )
        )
        
        # Prior terms
        prior_w = -0.5 * np.sum(self.q_params['mu_w']**2 + self.q_params['sigma_w']**2)
        prior_precision = (self.config.alpha_prior - 1) * (
            digamma(self.q_params['alpha_q']) - np.log(self.q_params['beta_q'])
        ) - self.config.beta_prior * self.q_params['alpha_q'] / self.q_params['beta_q']
        
        # Entropy terms
        entropy_w = 0.5 * np.sum(np.log(2 * np.pi * np.e * self.q_params['sigma_w']**2))
        entropy_precision = (
            self.q_params['alpha_q'] - np.log(self.q_params['beta_q']) +
            gammaln(self.q_params['alpha_q']) +
            (1 - self.q_params['alpha_q']) * digamma(self.q_params['alpha_q'])
        )
        
        elbo = expected_log_likelihood + prior_w + prior_precision + entropy_w + entropy_precision
        
        return elbo
    
    def predict(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using variational posterior"""
        
        X_design = np.column_stack([np.ones(len(X)), X])
        
        # Sample from variational posterior
        predictions = []
        for _ in range(n_samples):
            # Sample weights
            w_sample = np.random.normal(self.q_params['mu_w'], self.q_params['sigma_w'])
            
            # Predict
            pred = X_design @ w_sample
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        return pred_mean, pred_std

class OnlineBayesianLearning:
    """Online Bayesian learning for streaming data"""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.model = BayesianLinearRegression(config)
        self.update_count = 0
        self.performance_history = []
        
    def partial_fit(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Update model with new data using exponential forgetting"""
        
        start_time = time.time()
        
        if not self.model.fitted:
            # First update - full fit
            results = self.model.fit(X_new, y_new)
        else:
            # Incremental update with forgetting factor
            results = self._incremental_update(X_new, y_new)
        
        self.update_count += 1
        
        # Track performance
        y_pred, _ = self.model.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        
        performance = {
            'update_count': self.update_count,
            'mse': mse,
            'timestamp': dt.datetime.now(),
            'n_new_samples': len(y_new)
        }
        
        self.performance_history.append(performance)
        
        update_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'performance': performance,
            'update_time_ms': update_time
        }
    
    def _incremental_update(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Incremental Bayesian update with forgetting"""
        
        # Scale new data
        X_scaled = self.model.scaler.transform(X_new)
        X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        # Apply forgetting factor to existing parameters
        forget = self.config.forgetting_factor
        
        self.model.S_N *= forget
        self.model.alpha_N *= forget
        self.model.beta_N *= forget
        
        # Update with new data
        n_new = len(y_new)
        
        # Posterior updates
        S_N_inv_new = np.linalg.inv(self.model.S_N) + self.model.alpha * X_design.T @ X_design
        self.model.S_N = np.linalg.inv(S_N_inv_new)
        
        m_N_old = self.model.m_N
        self.model.m_N = self.model.S_N @ (
            np.linalg.inv(self.model.S_N) @ m_N_old + 
            self.model.alpha * X_design.T @ y_new
        )
        
        # Update precision
        self.model.alpha_N += n_new / 2
        residual = y_new - X_design @ self.model.m_N
        self.model.beta_N += 0.5 * np.sum(residual**2)
        
        return {
            'posterior_mean': self.model.m_N,
            'posterior_covariance': self.model.S_N,
            'alpha_posterior': self.model.alpha_N,
            'beta_posterior': self.model.beta_N,
            'forgetting_factor_applied': forget,
            'n_new_samples': n_new
        }
    
    def get_learning_curve(self) -> pd.DataFrame:
        """Get learning curve showing online performance"""
        
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df['rolling_mse'] = df['mse'].rolling(window=10, min_periods=1).mean()
        
        return df

class BayesianModelSelection:
    """Bayesian model selection and averaging"""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.models = {}
        self.model_evidences = {}
        self.model_weights = {}
        
    def fit_multiple_models(self, X: np.ndarray, y: np.ndarray, 
                          feature_subsets: List[List[int]]) -> Dict[str, Any]:
        """Fit multiple models for different feature subsets"""
        
        start_time = time.time()
        
        for i, feature_subset in enumerate(feature_subsets[:self.config.max_models]):
            model_name = f"model_{i}"
            
            # Select features
            X_subset = X[:, feature_subset]
            
            # Fit model
            model = BayesianLinearRegression(self.config)
            results = model.fit(X_subset, y)
            
            self.models[model_name] = {
                'model': model,
                'feature_subset': feature_subset,
                'results': results
            }
            
            self.model_evidences[model_name] = results['log_evidence']
        
        # Compute model weights (normalized evidence)
        max_evidence = max(self.model_evidences.values())
        log_weights = {name: evidence - max_evidence 
                      for name, evidence in self.model_evidences.items()}
        
        weights = {name: np.exp(log_weight) for name, log_weight in log_weights.items()}
        total_weight = sum(weights.values())
        
        self.model_weights = {name: weight / total_weight 
                            for name, weight in weights.items()}
        
        fitting_time = (time.time() - start_time) * 1000
        
        return {
            'n_models': len(self.models),
            'model_weights': self.model_weights,
            'model_evidences': self.model_evidences,
            'fitting_time_ms': fitting_time,
            'best_model': max(self.model_evidences.keys(), key=lambda k: self.model_evidences[k])
        }
    
    def predict_averaged(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian model averaging predictions"""
        
        predictions = []
        uncertainties = []
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            feature_subset = model_info['feature_subset']
            weight = self.model_weights[model_name]
            
            # Make prediction with selected features
            X_subset = X[:, feature_subset]
            pred_mean, pred_std = model.predict(X_subset, return_uncertainty=True)
            
            predictions.append(weight * pred_mean)
            uncertainties.append(weight * pred_std**2)  # Variance
        
        # Combine predictions
        averaged_pred = np.sum(predictions, axis=0)
        averaged_std = np.sqrt(np.sum(uncertainties, axis=0))
        
        return averaged_pred, averaged_std

class BayesianEngine:
    """Main Bayesian statistics engine"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.bayesian_config = BayesianConfig()
        
        # Components
        self.linear_models = {}
        self.hierarchical_models = {}
        self.variational_models = {}
        self.online_models = {}
        self.model_selection = None
        
        # Model persistence
        self.models_path = Path(self.config.models_path) / "bayesian"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def fit_bayesian_regression(self, X: np.ndarray, y: np.ndarray, 
                              model_name: str = "default") -> Dict[str, Any]:
        """Fit Bayesian linear regression"""
        
        model = BayesianLinearRegression(self.bayesian_config)
        results = model.fit(X, y)
        
        self.linear_models[model_name] = model
        
        return results
    
    def fit_hierarchical_model(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                             model_name: str = "hierarchical") -> Dict[str, Any]:
        """Fit hierarchical Bayesian model"""
        
        model = HierarchicalBayesianModel(self.bayesian_config)
        results = model.fit(data_dict)
        
        self.hierarchical_models[model_name] = model
        
        return results
    
    def fit_variational_model(self, X: np.ndarray, y: np.ndarray, 
                            model_name: str = "variational") -> Dict[str, Any]:
        """Fit using variational inference"""
        
        model = VariationalInference(self.bayesian_config)
        results = model.fit_variational(X, y)
        
        self.variational_models[model_name] = model
        
        return results
    
    def create_online_learner(self, model_name: str = "online") -> OnlineBayesianLearning:
        """Create online Bayesian learner"""
        
        learner = OnlineBayesianLearning(self.bayesian_config)
        self.online_models[model_name] = learner
        
        return learner
    
    def model_selection_analysis(self, X: np.ndarray, y: np.ndarray, 
                               feature_subsets: List[List[int]]) -> Dict[str, Any]:
        """Perform Bayesian model selection"""
        
        self.model_selection = BayesianModelSelection(self.bayesian_config)
        results = self.model_selection.fit_multiple_models(X, y, feature_subsets)
        
        return results
    
    def predict_with_uncertainty(self, X: np.ndarray, 
                               model_name: str = "default",
                               model_type: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification"""
        
        if model_type == "linear" and model_name in self.linear_models:
            return self.linear_models[model_name].predict(X, return_uncertainty=True)
        
        elif model_type == "variational" and model_name in self.variational_models:
            return self.variational_models[model_name].predict(X)
        
        elif model_type == "hierarchical" and model_name in self.hierarchical_models:
            return self.hierarchical_models[model_name].predict(X)
        
        elif model_type == "averaged" and self.model_selection:
            return self.model_selection.predict_averaged(X)
        
        else:
            raise ValueError(f"Model {model_name} of type {model_type} not found")
    
    def get_confidence_intervals(self, X: np.ndarray, 
                               model_name: str = "default") -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals"""
        
        if model_name in self.linear_models:
            return self.linear_models[model_name].get_confidence_intervals(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def save_models(self, model_suffix: str = None) -> str:
        """Save all Bayesian models"""
        
        if model_suffix is None:
            model_suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = self.models_path / f"bayesian_models_{model_suffix}.joblib"
        
        model_data = {
            'linear_models': self.linear_models,
            'hierarchical_models': self.hierarchical_models,
            'variational_models': self.variational_models,
            'online_models': self.online_models,
            'model_selection': self.model_selection,
            'bayesian_config': self.bayesian_config,
            'created_at': dt.datetime.now()
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Bayesian models saved to: {save_path}")
        
        return str(save_path)
    
    def load_models(self, model_path: str):
        """Load Bayesian models"""
        
        model_data = joblib.load(model_path)
        
        self.linear_models = model_data.get('linear_models', {})
        self.hierarchical_models = model_data.get('hierarchical_models', {})
        self.variational_models = model_data.get('variational_models', {})
        self.online_models = model_data.get('online_models', {})
        self.model_selection = model_data.get('model_selection')
        
        if 'bayesian_config' in model_data:
            self.bayesian_config = model_data['bayesian_config']
        
        logger.info(f"Bayesian models loaded from: {model_path}")

# Test function
def test_bayesian_engine():
    """Test Bayesian statistics engine"""
    print("🚀 Testing Bayesian Statistics Engine")
    print("=" * 50)
    
    # Create Bayesian engine
    config = SystemConfig()
    bayesian_engine = BayesianEngine(config)
    
    print(f"🔧 Bayesian Configuration:")
    print(f"   Alpha Prior: {bayesian_engine.bayesian_config.alpha_prior}")
    print(f"   Beta Prior: {bayesian_engine.bayesian_config.beta_prior}")
    print(f"   Max Iterations: {bayesian_engine.bayesian_config.max_iter}")
    print(f"   Confidence Levels: {bayesian_engine.bayesian_config.confidence_levels}")
    
    # Generate sample data
    print(f"\n📊 Generating sample data...")
    
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    # Create synthetic regression data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -0.8, 0.3, 2.1, -1.2])
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_weights + noise
    
    print(f"   Data Shape: X={X.shape}, y={y.shape}")
    print(f"   True Weights: {true_weights}")
    
    try:
        # Test Bayesian linear regression
        print(f"\n🧠 Testing Bayesian Linear Regression...")
        
        results = bayesian_engine.fit_bayesian_regression(X, y, "test_model")
        
        print(f"   ✅ Fitting Results:")
        print(f"      Log Evidence: {results['log_evidence']:.4f}")
        print(f"      Fitting Time: {results['fitting_time_ms']:.2f}ms")
        print(f"      Posterior Mean: {results['posterior_mean'][:3]}...")  # Show first 3
        
        # Test predictions with uncertainty
        print(f"\n🎯 Testing Predictions with Uncertainty...")
        
        X_test = np.random.randn(50, n_features)
        y_test_true = X_test @ true_weights
        
        pred_mean, pred_std = bayesian_engine.predict_with_uncertainty(X_test, "test_model")
        
        print(f"   ✅ Prediction Results:")
        print(f"      Mean Prediction Error: {np.mean(np.abs(pred_mean - y_test_true)):.4f}")
        print(f"      Avg Uncertainty: {np.mean(pred_std):.4f}")
        
        # Test confidence intervals
        print(f"\n📏 Testing Confidence Intervals...")
        
        intervals = bayesian_engine.get_confidence_intervals(X_test[:10], "test_model")
        
        print(f"   ✅ Confidence Intervals:")
        for conf_level, (lower, upper) in intervals.items():
            coverage = np.mean((y_test_true[:10] >= lower) & (y_test_true[:10] <= upper))
            print(f"      {conf_level*100:.0f}% CI Coverage: {coverage*100:.1f}%")
        
        # Test hierarchical model
        print(f"\n🏗️ Testing Hierarchical Bayesian Model...")
        
        # Create grouped data
        data_dict = {
            'group1': (X[:70], y[:70]),
            'group2': (X[70:140], y[70:140] + 0.5),  # Slight shift
            'group3': (X[140:], y[140:] - 0.3)       # Another shift
        }
        
        hier_results = bayesian_engine.fit_hierarchical_model(data_dict, "hierarchical_test")
        
        print(f"   ✅ Hierarchical Results:")
        print(f"      Global Model Evidence: {hier_results['global_results']['log_evidence']:.4f}")
        print(f"      Groups: {hier_results['n_groups']}")
        print(f"      Fitting Time: {hier_results['fitting_time_ms']:.2f}ms")
        
        # Test variational inference
        print(f"\n🔄 Testing Variational Inference...")
        
        var_results = bayesian_engine.fit_variational_model(X, y, "variational_test")
        
        print(f"   ✅ Variational Results:")
        print(f"      Converged: {var_results['converged']}")
        print(f"      Iterations: {var_results['n_iterations']}")
        print(f"      Final ELBO: {var_results['final_elbo']:.4f}")
        print(f"      Fitting Time: {var_results['fitting_time_ms']:.2f}ms")
        
        # Test online learning
        print(f"\n⚡ Testing Online Bayesian Learning...")
        
        online_learner = bayesian_engine.create_online_learner("online_test")
        
        # Simulate streaming updates
        batch_size = 20
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            
            update_results = online_learner.partial_fit(X_batch, y_batch)
            
            if i == 0:  # Print first update results
                print(f"   ✅ First Update Results:")
                print(f"      MSE: {update_results['performance']['mse']:.4f}")
                print(f"      Update Time: {update_results['update_time_ms']:.2f}ms")
        
        learning_curve = online_learner.get_learning_curve()
        print(f"      Total Updates: {len(learning_curve)}")
        print(f"      Final MSE: {learning_curve['mse'].iloc[-1]:.4f}")
        
        # Test model selection
        print(f"\n🎯 Testing Bayesian Model Selection...")
        
        # Create different feature subsets
        feature_subsets = [
            [0, 1, 2],
            [0, 1, 3, 4],
            [1, 2, 3],
            [0, 2, 4],
            list(range(n_features))  # All features
        ]
        
        selection_results = bayesian_engine.model_selection_analysis(X, y, feature_subsets)
        
        print(f"   ✅ Model Selection Results:")
        print(f"      Models Evaluated: {selection_results['n_models']}")
        print(f"      Best Model: {selection_results['best_model']}")
        
        # Show model weights
        for model_name, weight in selection_results['model_weights'].items():
            print(f"      {model_name}: {weight:.3f}")
        
        # Test model averaging
        pred_avg, std_avg = bayesian_engine.predict_with_uncertainty(
            X_test[:20], model_type="averaged"
        )
        
        avg_error = np.mean(np.abs(pred_avg - y_test_true[:20]))
        print(f"      Averaged Model Error: {avg_error:.4f}")
        
        # Test model saving
        print(f"\n💾 Testing Model Persistence...")
        
        model_path = bayesian_engine.save_models("test")
        print(f"   Models saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error in Bayesian engine test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Bayesian statistics engine test completed!")

if __name__ == "__main__":
    test_bayesian_engine()