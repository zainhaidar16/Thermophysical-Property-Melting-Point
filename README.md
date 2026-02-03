# Thermophysical Property: Melting Point Prediction

Machine learning solution for predicting melting points (°C) of organic compounds using molecular descriptors and group contribution features.

## Overview

This project implements multiple machine learning models to predict melting points from group contribution features - subgroup counts that represent functional groups within each molecule. The models capture complex, nonlinear relationships between molecular structure and melting behavior.

**Evaluation Metric**: Mean Absolute Error (MAE)

## Project Structure

```
.
├── data/                      # Data directory
│   ├── train.csv             # Training data with melting points
│   └── test.csv              # Test data for predictions
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── models.py            # ML model implementations
│   └── train.py             # Training pipeline
├── models/                   # Saved trained models
├── notebooks/                # Jupyter notebooks for exploration
├── main.py                  # Main script for training
├── create_sample_data.py   # Script to generate sample data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zainhaidar16/Thermophysical-Property-Melting-Point.git
cd Thermophysical-Property-Melting-Point
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

### Training Data (train.csv)
- `id`: Unique identifier for each compound
- `group_0`, `group_1`, ..., `group_N`: Group contribution features (molecular descriptors)
- `melting_point`: Target variable (melting point in °C)

### Test Data (test.csv)
- `id`: Unique identifier for each compound
- `group_0`, `group_1`, ..., `group_N`: Group contribution features (molecular descriptors)

## Usage

### Quick Start with Sample Data

Generate sample data and train models:

```bash
# Create sample data
python create_sample_data.py

# Train all models and generate predictions
python main.py --train data/train.csv --test data/test.csv --ensemble --predict
```

### Training Models

Train specific models:

```bash
# Train specific models
python main.py --train data/train.csv --models xgboost lightgbm random_forest

# Train all models with custom validation split
python main.py --train data/train.csv --val-size 0.25

# Create ensemble and save models
python main.py --train data/train.csv --ensemble
```

### Generate Predictions

```bash
# Generate predictions using ensemble
python main.py --train data/train.csv --test data/test.csv --ensemble --predict --output submission.csv

# Generate predictions using best single model
python main.py --train data/train.csv --test data/test.csv --predict
```

## Available Models

The pipeline supports the following models:

1. **Ridge Regression**: Linear model with L2 regularization
2. **Lasso Regression**: Linear model with L1 regularization
3. **Random Forest**: Ensemble of decision trees
4. **Gradient Boosting**: Sequential boosting algorithm
5. **XGBoost**: Optimized gradient boosting
6. **LightGBM**: Fast gradient boosting framework
7. **Ensemble**: Weighted average of multiple models

## Model Training Pipeline

The training pipeline automatically:

1. **Loads and preprocesses data**
   - Handles missing values
   - Standardizes features using StandardScaler
   - Splits data into train/validation sets

2. **Trains multiple models**
   - Each model uses optimized hyperparameters
   - Tracks training and validation metrics

3. **Evaluates performance**
   - MAE (Mean Absolute Error) - primary metric
   - RMSE (Root Mean Squared Error)
   - R² score

4. **Creates ensemble model** (optional)
   - Combines predictions from multiple models
   - Often achieves better performance than single models

5. **Saves trained models**
   - All models saved in `models/` directory
   - Includes preprocessing scaler

6. **Generates predictions** (optional)
   - Produces submission file in required format

## Example Output

```
Loading training data...
Preprocessing features...
Creating train/validation split...
Training set size: 80
Validation set size: 20
Number of features: 20

Training ridge model...
Train MAE: 12.3456, RMSE: 15.2341, R²: 0.8234
Val MAE: 13.4567, RMSE: 16.3452, R²: 0.7956

Training xgboost model...
Train MAE: 8.2341, RMSE: 10.4567, R²: 0.9234
Val MAE: 11.2345, RMSE: 14.3456, R²: 0.8456

...

VALIDATION RESULTS SUMMARY
================================================================================
xgboost              - Val MAE:  11.2345, RMSE:  14.3456, R²:  0.8456
lightgbm             - Val MAE:  11.4567, RMSE:  14.5678, R²:  0.8423
random_forest        - Val MAE:  12.1234, RMSE:  15.2345, R²:  0.8234
...

Creating ensemble model...
Ensemble Val MAE: 10.8765, RMSE: 13.9876, R²: 0.8567
```

## Customization

### Adding New Models

To add a new model, edit `src/models.py`:

```python
def _create_model(self, model_type: str, **kwargs):
    models = {
        # Add your model here
        'your_model': YourModel(**kwargs),
        ...
    }
```

### Tuning Hyperparameters

Modify default configurations in `src/models.py`:

```python
def get_default_model_configs():
    return {
        'xgboost': {
            'n_estimators': 500,  # Increase trees
            'learning_rate': 0.03,  # Decrease learning rate
            # Add more parameters
        }
    }
```

## Performance Tips

1. **Feature Engineering**: The quality of group contribution features significantly impacts performance
2. **Ensemble Models**: Combining multiple models often improves predictions
3. **Hyperparameter Tuning**: Use grid search or random search for optimal parameters
4. **Cross-Validation**: Consider k-fold cross-validation for more robust evaluation

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{melting_point_prediction,
  author = {Zain Haidar},
  title = {Thermophysical Property: Melting Point Prediction},
  year = {2026},
  url = {https://github.com/zainhaidar16/Thermophysical-Property-Melting-Point}
}
```

## Acknowledgments

- Group contribution methods for property prediction
- Scikit-learn, XGBoost, and LightGBM communities
- Machine learning community for best practices