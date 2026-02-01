# Poisson Models (Artifacts)

This folder stores trained artifacts for the Poisson goal models used in this project.

## Files

After running the training script, you should see:

- `poisson_home_model.joblib`  
  Trained **PoissonRegressor** pipeline for predicting **home goals** (λ_home).

- `poisson_away_model.joblib`  
  Trained **PoissonRegressor** pipeline for predicting **away goals** (λ_away).

- `poisson_feature_columns.joblib`  
  The list of feature columns (schema) used during training.  
  This ensures inference uses the exact same feature order as training.

## How to generate these files

From the project root:

```bash
python scripts/train_poisson.py \
  --feat-path data/processed/pl_training_dataset.csv \
  --match-path data/raw/pl_matches.csv \
  --model-dir models/poisson
