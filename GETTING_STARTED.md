# Getting Started with Melting Point Prediction

This guide will help you get started with the melting point prediction pipeline.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/zainhaidar16/Thermophysical-Property-Melting-Point.git
cd Thermophysical-Property-Melting-Point
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Prepare Your Data

1. **Create data directory**:
   ```bash
   mkdir -p data
   ```

2. **Prepare your data files**:
   - Place your training data in `data/train.csv`
   - Place your test data in `data/test.csv`

3. **Data format requirements**:
   
   Training data (`train.csv`):
   ```
   id,group_0,group_1,...,group_N,melting_point
   1,5,3,...,7,150.23
   2,2,8,...,4,120.45
   ...
   ```
   
   Test data (`test.csv`):
   ```
   id,group_0,group_1,...,group_N
   101,3,5,...,2
   102,7,1,...,9
   ...
   ```

4. **Run the pipeline**:
   ```bash
   python main.py --train data/train.csv --test data/test.csv --ensemble --predict
   ```

## Usage Examples

### Train Specific Models

Train only specific models:
```bash
python main.py --train data/train.csv --models xgboost lightgbm random_forest
```

### Adjust Validation Split

Change the validation set size:
```bash
python main.py --train data/train.csv --val-size 0.3
```

### Train Without Predictions

Just train and save models without generating predictions:
```bash
python main.py --train data/train.csv
```

### Use Custom Output Path

Specify a custom path for the submission file:
```bash
python main.py --train data/train.csv --test data/test.csv --predict --output my_predictions.csv
```

### Save Models to Custom Directory

Save trained models to a specific directory:
```bash
python main.py --train data/train.csv --model-dir my_models/
```

## Understanding the Output

After training, you'll see:

1. **Training Progress**: Real-time progress for each model
2. **Validation Results**: Performance metrics (MAE, RMSE, R²) for each model
3. **Model Ranking**: Models sorted by validation MAE
4. **Saved Models**: All models saved to `models/` directory
5. **Predictions**: Submission file saved to `submission.csv`

Example output:
```
VALIDATION RESULTS SUMMARY
================================================================================
xgboost              - Val MAE:  11.2345, RMSE:  14.3456, R²:  0.8456
lightgbm             - Val MAE:  11.4567, RMSE:  14.5678, R²:  0.8423
ridge                - Val MAE:  15.7506, RMSE:  19.4555, R²:  0.7956
...
```

## Submission Format

The output file (`submission.csv`) will have this format:
```
id,melting_point
101,123.45
102,98.76
103,145.23
...
```

## Jupyter Notebook

For interactive exploration and analysis:

```bash
cd notebooks
jupyter notebook exploratory_analysis.ipynb
```

The notebook includes:
- Data exploration and visualization
- Feature correlation analysis
- Model training and comparison
- Prediction generation
- Results analysis

## Running Tests

Validate the pipeline installation:

```bash
python test_pipeline.py
```

All tests should pass with:
```
================================================================================
ALL TESTS PASSED ✓
================================================================================
```

## Troubleshooting

### Missing Data Files

**Error**: `Training data not found at data/train.csv`

**Solution**: Place your own data files in the `data/` directory following the format specified in the "Prepare Your Data" section.

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Memory Issues

If you encounter memory issues with large datasets:

1. Reduce the number of estimators in tree-based models
2. Train models one at a time instead of all at once
3. Use a smaller validation set size

## Model Performance Tips

1. **Feature Engineering**: Create meaningful group contribution features
2. **Hyperparameter Tuning**: Experiment with different model parameters
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Cross-Validation**: Use k-fold CV for more robust evaluation

## Next Steps

1. **Explore the data**: Use the Jupyter notebook to understand your dataset
2. **Experiment with models**: Try different model types and parameters
3. **Feature engineering**: Create new features from existing ones
4. **Hyperparameter tuning**: Use grid search or random search
5. **Submit predictions**: Upload `submission.csv` to the competition platform

## Support

For issues or questions:
- Check the main [README.md](README.md) for detailed documentation
- Review the code in the `src/` directory
- Run tests with `python test_pipeline.py` to verify installation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
