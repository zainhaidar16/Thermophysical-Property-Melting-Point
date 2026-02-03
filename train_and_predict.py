#!/usr/bin/env python
"""
Standalone script for training melting point prediction models and generating submissions.
This script implements the complete ML pipeline for the melting point prediction challenge.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.data_loader import MeltingPointDataLoader
from src.models import BaseModel, EnsembleModel, get_default_model_configs
from src.train import TrainingPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


def create_engineered_features(df, feature_cols):
    """Create new engineered features for group contribution analysis."""
    df_eng = df.copy()
    
    # 1. Polynomial features for top features
    for col in feature_cols[:5]:
        df_eng[f'{col}_squared'] = df_eng[col] ** 2
    
    # 2. Interaction terms
    df_eng['group_interaction_12'] = df_eng[feature_cols[0]] * df_eng[feature_cols[1]]
    df_eng['group_interaction_23'] = df_eng[feature_cols[1]] * df_eng[feature_cols[2]]
    
    # 3. Sum of all group contributions
    df_eng['total_groups'] = df_eng[feature_cols].sum(axis=1)
    
    # 4. Mean group contribution
    df_eng['mean_group'] = df_eng[feature_cols].mean(axis=1)
    
    # 5. Std of group contributions
    df_eng['std_group'] = df_eng[feature_cols].std(axis=1)
    
    # 6. Max and Min group contributions
    df_eng['max_group'] = df_eng[feature_cols].max(axis=1)
    df_eng['min_group'] = df_eng[feature_cols].min(axis=1)
    df_eng['max_min_ratio'] = df_eng['max_group'] / (df_eng['min_group'] + 1e-6)
    
    # 7. Skewness and entropy
    from scipy.stats import skew
    df_eng['skewness_groups'] = df_eng[feature_cols].apply(lambda x: skew(x), axis=1)
    
    def calculate_entropy(row):
        total = row.sum()
        if total == 0:
            return 0
        probs = row / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    
    df_eng['entropy_groups'] = df_eng[feature_cols].apply(calculate_entropy, axis=1)
    
    return df_eng


def main():
    """Main pipeline execution."""
    print("="*80)
    print("MELTING POINT PREDICTION - ML PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # File paths
    train_path = 'train.csv'
    test_path = 'test.csv'
    model_dir = 'models'
    submission_dir = 'submissions'
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    # ========== STEP 1: LOAD DATA ==========
    print("Step 1: Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col not in ['id', 'SMILES', 'Tm']]
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Features: {len(feature_cols)}\n")
    
    # ========== STEP 2: FEATURE ENGINEERING ==========
    print("Step 2: Feature engineering...")
    train_eng = create_engineered_features(train_df, feature_cols)
    test_eng = create_engineered_features(test_df, feature_cols)
    
    new_features = [col for col in train_eng.columns if col not in train_df.columns]
    all_feature_cols = feature_cols + new_features
    print(f"  New features created: {len(new_features)}")
    print(f"  Total features: {len(all_feature_cols)}\n")
    
    # ========== STEP 3: PREPARE DATA ==========
    print("Step 3: Preparing data...")
    X = train_eng[all_feature_cols].values
    y = train_eng['Tm'].values
    X_test = test_eng[all_feature_cols].values
    
    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples\n")
    
    # ========== STEP 4: TRAIN MODELS ==========
    print("Step 4: Training models...")
    
    models = {}
    results = {}
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_val_pred_xgb = xgb_model.predict(X_val)
    mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
    r2_xgb = r2_score(y_val, y_val_pred_xgb)
    models['xgboost'] = xgb_model
    results['xgboost'] = {'mae': mae_xgb, 'r2': r2_xgb}
    print(f"    MAE: {mae_xgb:.4f}, R²: {r2_xgb:.4f}")
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    y_val_pred_lgb = lgb_model.predict(X_val)
    mae_lgb = mean_absolute_error(y_val, y_val_pred_lgb)
    r2_lgb = r2_score(y_val, y_val_pred_lgb)
    models['lightgbm'] = lgb_model
    results['lightgbm'] = {'mae': mae_lgb, 'r2': r2_lgb}
    print(f"    MAE: {mae_lgb:.4f}, R²: {r2_lgb:.4f}\n")
    
    # ========== STEP 5: SELECT BEST MODEL ==========
    print("Step 5: Model selection...")
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_model = models[best_model_name]
    print(f"  Best model: {best_model_name.upper()}")
    print(f"  Validation MAE: {results[best_model_name]['mae']:.4f}\n")
    
    # ========== STEP 6: TRAIN FINAL MODEL ==========
    print("Step 6: Training final model on full dataset...")
    best_model.fit(X_scaled, y)
    print("  ✓ Model trained on full training set\n")
    
    # ========== STEP 7: GENERATE PREDICTIONS ==========
    print("Step 7: Generating predictions...")
    y_test_pred = best_model.predict(X_test_scaled)
    print(f"  Predictions generated for {len(y_test_pred)} test samples")
    print(f"  Mean prediction: {y_test_pred.mean():.2f} K")
    print(f"  Range: [{y_test_pred.min():.2f}, {y_test_pred.max():.2f}] K\n")
    
    # ========== STEP 8: CREATE SUBMISSION ==========
    print("Step 8: Creating submission file...")
    submission_df = pd.DataFrame({
        'id': test_df['id'].values,
        'Tm': y_test_pred
    })
    
    submission_path = os.path.join(submission_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"  ✓ Submission saved to: {submission_path}\n")
    
    # ========== STEP 9: SAVE MODELS ==========
    print("Step 9: Saving models...")
    
    # Save best model
    model_path = os.path.join(model_dir, f'{best_model_name}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved to: {scaler_path}")
    
    # Save feature information
    feature_info = {
        'all_features': all_feature_cols,
        'original_features': feature_cols,
        'engineered_features': new_features,
        'n_features': len(all_feature_cols)
    }
    info_path = os.path.join(model_dir, 'feature_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"  ✓ Feature info saved to: {info_path}\n")
    
    # ========== SUMMARY ==========
    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Summary:")
    print(f"  Best Model: {best_model_name.upper()}")
    print(f"  Validation MAE: {results[best_model_name]['mae']:.4f}")
    print(f"  Submission File: {submission_path}")
    print(f"  Model Path: {model_path}")
    print("\nReady for submission to Kaggle!")


if __name__ == '__main__':
    main()
