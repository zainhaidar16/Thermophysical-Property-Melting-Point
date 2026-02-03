"""ML models for melting point prediction."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


class BaseModel:
    """Base class for melting point prediction models."""
    
    def __init__(self, model_type: str, **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: Type of model to use
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_fitted = False
        
    def _create_model(self, model_type: str, **kwargs):
        """Create model instance based on type."""
        model_classes = {
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model_classes[model_type](**kwargs)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y_true: True target values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }


class EnsembleModel:
    """Ensemble of multiple models for melting point prediction."""
    
    def __init__(self, models: list):
        """
        Initialize ensemble.
        
        Args:
            models: List of BaseModel instances
        """
        self.models = models
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(X_train, y_train)
            
    def predict(self, X: np.ndarray, weights: Optional[list] = None) -> np.ndarray:
        """
        Make predictions by averaging model outputs.
        
        Args:
            X: Features
            weights: Optional weights for each model (defaults to equal weighting)
            
        Returns:
            Averaged predictions
        """
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
            
        predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, weights):
            predictions += weight * model.predict(X)
            
        return predictions
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                weights: Optional[list] = None) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Features
            y_true: True target values
            weights: Optional weights for each model
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X, weights)
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }


def get_default_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for different models."""
    return {
        'ridge': {
            'alpha': 1.0,
            'random_state': 42
        },
        'lasso': {
            'alpha': 0.1,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'lightgbm': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }
