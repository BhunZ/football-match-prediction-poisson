# scripts/train_poisson.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import PoissonRegressor
import joblib


# -----------------------------
# Helpers
# -----------------------------
def parse_score(score: str):
    """Parse score like '2–1' or '2-1' into (home_goals, away_goals)."""
    if pd.isna(score):
        return (np.nan, np.nan)
    parts = re.split(r"[-–—]", str(score))
    if len(parts) < 2:
        return (np.nan, np.nan)
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return (np.nan, np.nan)


def neg_poisson_dev_safe(y_true, y_pred):
    """GridSearch maximizes score -> return negative deviance."""
    y_pred = np.maximum(y_pred, 1e-9)
    return -mean_poisson_deviance(y_true, y_pred)


poisson_scorer = make_scorer(neg_poisson_dev_safe, greater_is_better=True)


def tune_poisson_alpha(X, y, cv_splits=3):
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
        ("model", PoissonRegressor(max_iter=20000, fit_intercept=True))
    ])

    param_grid = {
        "model__alpha": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 150.0, 200.0]
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring=poisson_scorer,
        cv=tscv,
        n_jobs=-1
    )
    gs.fit(X, y)
    best = gs.best_estimator_
    best_alpha = gs.best_params_["model__alpha"]
    best_dev = -gs.best_score_  # convert back to deviance
    return best, best_alpha, best_dev


def evaluate_goals(model, X_train, y_train, X_test, y_test, label):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print(f"\n[{label}]")
    print("Train MAE:", mean_absolute_error(y_train, pred_train))
    print("Test  MAE:", mean_absolute_error(y_test, pred_test))
    return pred_train, pred_test


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Poisson models for football goals (home/away).")
    parser.add_argument("--feat-path", type=str, default="data/processed/pl_training_dataset.csv", help="Training dataset path.")
    parser.add_argument("--match-path", type=str, default="data/raw/pl_matches.csv", help="Matches raw path (for goals).")
    parser.add_argument("--model-dir", type=str, default="models/poisson", help="Directory to save trained models.")
    parser.add_argument("--cv-splits", type=int, default=3, help="TimeSeriesSplit folds for GridSearch.")
    args = parser.parse_args()

    feat_path = Path(args.feat_path)
    match_path = Path(args.match_path)
    model_dir = Path(args.model_dir)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature dataset: {feat_path}")
    if not match_path.exists():
        raise FileNotFoundError(f"Missing matches file: {match_path}")

    feat = pd.read_csv(feat_path)
    matches = pd.read_csv(match_path)

    feat["Date"] = pd.to_datetime(feat["Date"])
    matches["Date"] = pd.to_datetime(matches["Date"])

    # Parse goals from Score
    hg, ag = zip(*matches["Score"].map(parse_score))
    matches["home_goals"] = hg
    matches["away_goals"] = ag

    # Merge goals into feature dataset
    df = feat.merge(
        matches[["Season", "Date", "Home", "Away", "home_goals", "away_goals"]],
        on=["Season", "Date", "Home", "Away"],
        how="inner",
    )
    df = df.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

    # Targets
    y_home = df["home_goals"].astype(int)
    y_away = df["away_goals"].astype(int)

    # Drop non-features
    DROP_COLS = ["Season", "Date", "Home", "Away", "Wk", "home_goals", "away_goals", "y"]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    # Numeric only (PoissonRegressor expects numeric)
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()

    # Sort by time
    df_ord = df.sort_values(["Season", "Date"])
    ord_idx = df_ord.index

    X_ord = X.loc[ord_idx].reset_index(drop=True)
    y_home_ord = y_home.loc[ord_idx].reset_index(drop=True)
    y_away_ord = y_away.loc[ord_idx].reset_index(drop=True)

    season_ord = df_ord["Season"].reset_index(drop=True)
    last_season = season_ord.max()
    test_mask = (season_ord == last_season).values

    X_train, X_test = X_ord[~test_mask], X_ord[test_mask]
    y_home_train, y_home_test = y_home_ord[~test_mask], y_home_ord[test_mask]
    y_away_train, y_away_test = y_away_ord[~test_mask], y_away_ord[test_mask]

    print("Train seasons:", sorted(df_ord.loc[~test_mask, "Season"].unique()))
    print("Test season:", last_season)
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)

    # Tune + train
    home_model, alpha_home, cv_dev_home = tune_poisson_alpha(X_train, y_home_train, cv_splits=args.cv_splits)
    away_model, alpha_away, cv_dev_away = tune_poisson_alpha(X_train, y_away_train, cv_splits=args.cv_splits)

    print("\nBest HOME alpha:", alpha_home, "| CV Poisson deviance:", cv_dev_home)
    print("Best AWAY alpha:", alpha_away, "| CV Poisson deviance:", cv_dev_away)

    # Evaluate
    evaluate_goals(home_model, X_train, y_home_train, X_test, y_home_test, "HOME GOALS")
    evaluate_goals(away_model, X_train, y_away_train, X_test, y_away_test, "AWAY GOALS")

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(home_model, model_dir / "poisson_home_model.joblib")
    joblib.dump(away_model, model_dir / "poisson_away_model.joblib")
    joblib.dump(list(X_train.columns), model_dir / "poisson_feature_columns.joblib")

    print("\n[OK] Saved models to:", model_dir.resolve())


if __name__ == "__main__":
    main()
