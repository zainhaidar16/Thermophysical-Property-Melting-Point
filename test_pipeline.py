"""Test script to validate the melting point prediction pipeline."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import MeltingPointDataLoader
from src.models import BaseModel, EnsembleModel, get_default_model_configs
from src.train import TrainingPipeline


def test_data_loader():
    """Test data loading functionality."""
    print("Testing Data Loader...")
    
    # Create sample data
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 10 + 100
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['id'] = range(1, n_samples + 1)
    df['melting_point'] = y
    
    # Save temporary CSV
    temp_path = '/tmp/test_train.csv'
    df.to_csv(temp_path, index=False)
    
    # Test loader
    loader = MeltingPointDataLoader(temp_path)
    train_df, _ = loader.load_data()
    
    assert train_df.shape[0] == n_samples, "Data loading failed"
    
    X_scaled, y_loaded, ids = loader.preprocess_features(train_df, fit=True)
    
    assert X_scaled.shape == (n_samples, n_features), "Feature preprocessing failed"
    assert len(y_loaded) == n_samples, "Target extraction failed"
    assert len(ids) == n_samples, "ID extraction failed"
    
    # Test train/val split
    X_train, X_val, y_train, y_val = loader.create_train_val_split(X_scaled, y_loaded)
    
    assert X_train.shape[0] + X_val.shape[0] == n_samples, "Train/val split failed"
    
    # Cleanup
    os.remove(temp_path)
    
    print("✓ Data Loader tests passed\n")


def test_models():
    """Test model creation and training."""
    print("Testing Models...")
    
    # Create sample data
    n_samples = 100
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples) * 10 + 100
    X_test = np.random.randn(20, n_features)
    
    # Test each model type
    model_types = ['ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm']
    
    trained_models = []
    
    for model_type in model_types:
        config = get_default_model_configs()[model_type]
        
        # Reduce size for faster testing
        if 'n_estimators' in config:
            config['n_estimators'] = 10
        
        model = BaseModel(model_type, **config)
        model.train(X_train, y_train)
        
        assert model.is_fitted, f"{model_type} model not fitted"
        
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0], f"{model_type} prediction shape mismatch"
        
        metrics = model.evaluate(X_train, y_train)
        assert 'mae' in metrics, f"{model_type} MAE not in metrics"
        assert 'rmse' in metrics, f"{model_type} RMSE not in metrics"
        assert 'r2' in metrics, f"{model_type} R² not in metrics"
        
        trained_models.append(model)
        print(f"  ✓ {model_type} model OK")
    
    # Test ensemble
    ensemble = EnsembleModel(trained_models)
    ensemble_preds = ensemble.predict(X_test)
    assert ensemble_preds.shape[0] == X_test.shape[0], "Ensemble prediction shape mismatch"
    
    ensemble_metrics = ensemble.evaluate(X_train, y_train)
    assert 'mae' in ensemble_metrics, "Ensemble MAE not in metrics"
    
    print("  ✓ Ensemble model OK")
    print("✓ Model tests passed\n")


def test_training_pipeline():
    """Test the full training pipeline."""
    print("Testing Training Pipeline...")
    
    # Create sample data files
    n_train = 80
    n_test = 20
    n_features = 10
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randn(n_train) * 10 + 100
    X_test = np.random.randn(n_test, n_features)
    
    train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(n_features)])
    train_df['id'] = range(1, n_train + 1)
    train_df['melting_point'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(n_features)])
    test_df['id'] = range(n_train + 1, n_train + n_test + 1)
    
    # Save to temp files
    temp_train = '/tmp/test_pipeline_train.csv'
    temp_test = '/tmp/test_pipeline_test.csv'
    temp_model_dir = '/tmp/test_models'
    temp_submission = '/tmp/test_submission.csv'
    
    os.makedirs(temp_model_dir, exist_ok=True)
    
    train_df.to_csv(temp_train, index=False)
    test_df.to_csv(temp_test, index=False)
    
    # Test pipeline
    pipeline = TrainingPipeline(
        train_path=temp_train,
        test_path=temp_test,
        model_dir=temp_model_dir,
        val_size=0.25
    )
    
    # Load and prepare data
    X_tr, X_val, y_tr, y_val = pipeline.load_and_prepare_data()
    
    assert X_tr.shape[0] > 0, "Training data loading failed"
    assert X_val.shape[0] > 0, "Validation data loading failed"
    
    # Train models (subset for speed)
    results = pipeline.train_all_models(
        X_tr, y_tr, X_val, y_val,
        model_types=['ridge', 'lasso']
    )
    
    assert len(results) > 0, "No models trained"
    
    for model_type, result in results.items():
        assert 'val_metrics' in result, f"{model_type} missing validation metrics"
        assert result['val_metrics']['mae'] >= 0, f"{model_type} invalid MAE"
        print(f"  ✓ {model_type} pipeline OK")
    
    # Test ensemble creation
    ensemble = pipeline.create_ensemble(
        list(results.keys()),
        X_tr, y_tr, X_val, y_val
    )
    
    assert ensemble is not None, "Ensemble creation failed"
    print("  ✓ Ensemble creation OK")
    
    # Test model saving
    pipeline.save_models()
    
    assert os.path.exists(os.path.join(temp_model_dir, 'ridge_model.pkl')), "Model saving failed"
    print("  ✓ Model saving OK")
    
    # Test prediction generation
    submission = pipeline.generate_predictions('ridge', temp_submission)
    
    assert os.path.exists(temp_submission), "Submission file not created"
    assert submission.shape[0] == n_test, "Incorrect number of predictions"
    assert 'id' in submission.columns, "Missing ID column in submission"
    assert 'melting_point' in submission.columns, "Missing melting_point column in submission"
    print("  ✓ Prediction generation OK")
    
    # Cleanup
    os.remove(temp_train)
    os.remove(temp_test)
    os.remove(temp_submission)
    
    import shutil
    shutil.rmtree(temp_model_dir)
    
    print("✓ Training Pipeline tests passed\n")


def main():
    """Run all tests."""
    print("="*80)
    print("MELTING POINT PREDICTION - VALIDATION TESTS")
    print("="*80 + "\n")
    
    try:
        test_data_loader()
        test_models()
        test_training_pipeline()
        
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
