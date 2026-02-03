"""Training pipeline for melting point prediction models."""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_loader import MeltingPointDataLoader
from src.models import BaseModel, EnsembleModel, get_default_model_configs


class TrainingPipeline:
    """Pipeline for training melting point prediction models."""
    
    def __init__(self,
                 train_path: str,
                 test_path: Optional[str] = None,
                 model_dir: str = 'models',
                 val_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize training pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            model_dir: Directory to save trained models
            val_size: Validation set size
            random_state: Random seed
        """
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = model_dir
        self.val_size = val_size
        self.random_state = random_state
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = MeltingPointDataLoader(train_path, test_path)
        
        # Storage for trained models
        self.trained_models = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare training and validation data.
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        print("Loading training data...")
        train_df, _ = self.data_loader.load_data()
        
        print("Preprocessing features...")
        X, y, _ = self.data_loader.preprocess_features(train_df, fit=True)
        
        print("Creating train/validation split...")
        X_train, X_val, y_train, y_val = self.data_loader.create_train_val_split(
            X, y, val_size=self.val_size, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val
    
    def train_single_model(self,
                          model_type: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          **kwargs) -> Dict:
        """
        Train a single model and evaluate on validation set.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary with model and metrics
        """
        print(f"\nTraining {model_type} model...")
        
        # Get default config and update with provided kwargs
        default_config = get_default_model_configs().get(model_type, {})
        config = {**default_config, **kwargs}
        
        # Create and train model
        model = BaseModel(model_type, **config)
        model.train(X_train, y_train)
        
        # Evaluate
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        
        print(f"Train MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # Save model
        self.trained_models[model_type] = model
        
        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
    
    def train_all_models(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        model_types: Optional[List[str]] = None) -> Dict:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_types: List of model types to train (defaults to all)
            
        Returns:
            Dictionary of results for each model
        """
        if model_types is None:
            model_types = ['ridge', 'lasso', 'random_forest', 'gradient_boosting', 
                          'xgboost', 'lightgbm']
        
        results = {}
        
        for model_type in model_types:
            try:
                result = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                results[model_type] = result
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                
        return results
    
    def create_ensemble(self,
                       model_types: List[str],
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray) -> EnsembleModel:
        """
        Create and evaluate an ensemble model.
        
        Args:
            model_types: List of model types to include in ensemble
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained ensemble model
        """
        print("\nCreating ensemble model...")
        
        models = [self.trained_models[mt] for mt in model_types if mt in self.trained_models]
        
        if not models:
            raise ValueError("No trained models available for ensemble")
        
        ensemble = EnsembleModel(models)
        
        # Evaluate ensemble
        train_metrics = ensemble.evaluate(X_train, y_train)
        val_metrics = ensemble.evaluate(X_val, y_val)
        
        print(f"Ensemble Train MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Ensemble Val MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        self.trained_models['ensemble'] = ensemble
        
        return ensemble
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.trained_models.items():
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.data_loader.scaler, f)
        print(f"Saved scaler to {scaler_path}")
    
    def generate_predictions(self,
                           model_name: str = 'ensemble',
                           output_path: str = 'submission.csv') -> pd.DataFrame:
        """
        Generate predictions on test set.
        
        Args:
            model_name: Name of model to use for predictions
            output_path: Path to save submission file
            
        Returns:
            DataFrame with predictions
        """
        if self.test_path is None:
            raise ValueError("Test path not provided")
        
        print(f"\nGenerating predictions using {model_name} model...")
        
        # Load test data
        _, test_df = self.data_loader.load_data()
        
        # Preprocess
        X_test, _, test_ids = self.data_loader.preprocess_features(
            test_df, fit=False
        )
        
        # Get model
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.trained_models[model_name]
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': test_ids,
            'melting_point': predictions
        })
        
        # Save submission
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return submission
