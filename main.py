#!/usr/bin/env python
"""Main script for training and predicting melting points."""

import argparse
import os
from src.train import TrainingPipeline


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(
        description='Train models for melting point prediction'
    )
    
    parser.add_argument(
        '--train',
        type=str,
        default='data/train.csv',
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        default='data/test.csv',
        help='Path to test data CSV'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to train (ridge, lasso, random_forest, gradient_boosting, xgboost, lightgbm)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Validation set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Create ensemble model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate predictions on test set'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output path for predictions'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.train):
        print(f"Error: Training data not found at {args.train}")
        print("Please place your training data CSV file in the specified location.")
        print("Expected format: CSV with 'id' column, feature columns, and 'melting_point' target column")
        return
    
    # Initialize pipeline
    test_path = args.test if os.path.exists(args.test) else None
    pipeline = TrainingPipeline(
        train_path=args.train,
        test_path=test_path,
        model_dir=args.model_dir,
        val_size=args.val_size
    )
    
    # Load and prepare data
    X_train, X_val, y_train, y_val = pipeline.load_and_prepare_data()
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    results = pipeline.train_all_models(
        X_train, y_train, X_val, y_val,
        model_types=args.models
    )
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    for model_type, result in sorted(
        results.items(),
        key=lambda x: x[1]['val_metrics']['mae']
    ):
        val_mae = result['val_metrics']['mae']
        val_rmse = result['val_metrics']['rmse']
        val_r2 = result['val_metrics']['r2']
        print(f"{model_type:20s} - Val MAE: {val_mae:8.4f}, RMSE: {val_rmse:8.4f}, R²: {val_r2:7.4f}")
    
    # Create ensemble if requested
    if args.ensemble:
        ensemble_models = list(results.keys())
        pipeline.create_ensemble(
            ensemble_models, X_train, y_train, X_val, y_val
        )
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    pipeline.save_models()
    
    # Generate predictions if requested
    if args.predict:
        if test_path is None:
            print("\nWarning: Test data not found. Skipping prediction generation.")
        else:
            print("\n" + "="*80)
            print("GENERATING PREDICTIONS")
            print("="*80)
            
            model_name = 'ensemble' if args.ensemble else min(
                results.items(),
                key=lambda x: x[1]['val_metrics']['mae']
            )[0]
            
            print(f"Using {model_name} model for predictions")
            pipeline.generate_predictions(model_name, args.output)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
