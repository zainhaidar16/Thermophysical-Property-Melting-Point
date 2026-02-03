"""Create sample data files for demonstration."""

import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    
    # Create sample training data with group contribution features
    n_train = 100
    n_features = 20
    
    # Generate random molecular descriptor features
    feature_names = [f'group_{i}' for i in range(n_features)]
    
    X_train = np.random.randint(0, 10, size=(n_train, n_features))
    
    # Generate synthetic melting points (nonlinear relationship)
    # Melting point depends on various group contributions
    weights = np.random.randn(n_features) * 10
    melting_points = X_train @ weights + np.random.randn(n_train) * 15 + 150
    
    # Create training dataframe
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df.insert(0, 'id', range(1, n_train + 1))
    train_df['melting_point'] = melting_points
    
    # Create sample test data (without target)
    n_test = 30
    X_test = np.random.randint(0, 10, size=(n_test, n_features))
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df.insert(0, 'id', range(n_train + 1, n_train + n_test + 1))
    
    return train_df, test_df


if __name__ == '__main__':
    import os
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate and save sample data
    train_df, test_df = create_sample_data()
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("Sample data created successfully!")
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTraining data columns: {list(train_df.columns)}")
    print(f"\nFirst few rows of training data:")
    print(train_df.head())
