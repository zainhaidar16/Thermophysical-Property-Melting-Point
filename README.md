# Thermophysical Property: Melting Point Prediction

Machine learning solution for predicting melting points (Tm, in Kelvin) of organic compounds using group contribution features. The goal is a low **MAE** on Kaggle’s held-out test set.

## Project Structure (current)

```
.
├── train.csv / test.csv          # Kaggle competition data (id, SMILES, group features, Tm only in train)
├── sample_submission.csv         # Kaggle format example
├── submissions/                  # Generated submissions
├── train_and_predict.py          # Scripted end-to-end pipeline
├── notebooks/
│   └── melting_point_prediction.ipynb  # Clean notebook for EDA → modeling → submission
├── src/
│   ├── data_loader.py            # Data loading / scaling helpers
│   ├── models.py                 # Model wrappers and defaults
│   └── train.py                  # Training pipeline utilities
├── requirements.txt
└── README.md (this file)
```

## Quickstart

1) Install dependencies (use your venv):
```bash
pip install -r requirements.txt
```

2) Run the scripted pipeline and produce a submission:
```bash
python train_and_predict.py
# outputs: submissions/submission.csv
```

3) (Optional) Work interactively: open `notebooks/melting_point_prediction.ipynb` and run top → bottom to explore, validate, and create a submission.

## Data Schema

- `id`: unique identifier
- `SMILES`: molecular string (not used directly here)
- `Group x`: group contribution feature columns
- `Tm`: target melting point (K) — **train only**

## Modeling Outline

- Feature engineering: polynomial terms on top groups, pairwise interactions, and aggregated stats (sum/mean/std/max/min, ratios, skewness, entropy).
- Scaling: StandardScaler on engineered features.
- Models: LightGBM, XGBoost, Random Forest, plus a weighted VotingRegressor ensemble.
- Validation: 5-fold CV for robustness + hold-out split for quick checks.
- Submission: predictions written to `submissions/submission.csv` with columns `[id, Tm]`.

## Tips for a Better Score

- Tune LightGBM/XGBoost (learning_rate, depth, leaves/estimators) via RandomizedSearchCV or Optuna.
- Increase engineered interactions for top-correlated group features.
- Try stacking/blending (e.g., add SVR/CatBoost) if time permits.
- Use more CV folds (e.g., 10-fold) for stability; average multiple seeds.

## License

MIT License. See `LICENSE` for details.