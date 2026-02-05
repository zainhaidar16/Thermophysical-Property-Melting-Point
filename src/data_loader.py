"""Data loading and preprocessing utilities for melting point prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MeltingPointDataLoader:
    """Load and preprocess melting point data."""
    
    def __init__(self, train_path: str, test_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV (optional)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load training and test data.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path) if self.test_path else None
        
        return train_df, test_df
    
    def preprocess_features(self, 
                           df: pd.DataFrame, 
                           target_col: str = 'melting_point',
                           id_col: str = 'id',
                           fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], pd.Index]:
        """
        Preprocess features by scaling and separating target.
        
        Args:
            df: DataFrame with features and optionally target
            target_col: Name of target column
            id_col: Name of ID column
            fit: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            Tuple of (X_scaled, y, ids)
        """
        # Separate IDs
        ids = df[id_col] if id_col in df.columns else df.index
        
        # Check for alternative target column names (Tm is commonly used for melting point)
        if target_col not in df.columns and 'Tm' in df.columns:
            target_col = 'Tm'
        
        # Separate target if exists
        y = df[target_col].values if target_col in df.columns else None
        
        # Get feature columns (exclude id, target, and non-numeric columns like SMILES)
        exclude_cols = [id_col, target_col, 'SMILES', 'smiles']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y, ids
    
    def create_train_val_split(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               val_size: float = 0.2,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            val_size: Validation set size (fraction)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        return train_test_split(X, y, test_size=val_size, random_state=random_state)
