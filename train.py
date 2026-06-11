"""
train.py
========
Trains two classifiers on the historical feature matrix:
    1. Logistic Regression  (interpretable baseline)
    2. Random Forest         (non-linear, captures interactions)

Both models use per-sample weights so that recent seasons count more.
Trained models are saved to models/ as joblib files.

Usage:
    python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from fetch_data import fetch_all_seasons
from features import build_feature_matrix, FEATURE_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LR_PATH = os.path.join(MODELS_DIR, "lr_model.joblib")
RF_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")


# ── Training ──────────────────────────────────────────────────────────────────

def train(raw_df: pd.DataFrame | None = None) -> dict:
    """
    Build features, evaluate via manual CV without data leakage, 
    train final models on full data, and save.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if raw_df is None:
        raw_df = fetch_all_seasons()

    print("Building feature matrix …")
    X, y, weights = build_feature_matrix(raw_df)
    
    # Convert to numpy arrays for easier slicing in the CV loop
    X_arr = X.values
    y_arr = y.values
    weights_arr = weights.values

    # TimeSeriesSplit keeps chronological order, preventing future data leakage
    cv = TimeSeriesSplit(n_splits=5)

    # Base models for CV
    lr_base = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20, 
        max_features="sqrt", class_weight="balanced", 
        random_state=42, n_jobs=-1
    )

    lr_scores = []
    rf_scores = []

    print("\nRunning Manual Cross-Validation (Time-Series) …")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        # 1. Split the data
        X_train, y_train, w_train = X_arr[train_idx], y_arr[train_idx], weights_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]
        
        # 2. Scale the data (fit ONLY on training fold to prevent leakage)
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_test_scaled = fold_scaler.transform(X_test)
        
        # 3. Train models with sample weights
        lr_base.fit(X_train_scaled, y_train, sample_weight=w_train)
        rf_base.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        # 4. Score
        lr_acc = lr_base.score(X_test_scaled, y_test)
        rf_acc = rf_base.score(X_test_scaled, y_test)
        
        lr_scores.append(lr_acc)
        rf_scores.append(rf_acc)
        
        print(f"  Fold {fold+1}: LR = {lr_acc:.4f} | RF = {rf_acc:.4f}")

    lr_cv_mean, lr_cv_std = np.mean(lr_scores), np.std(lr_scores)
    rf_cv_mean, rf_cv_std = np.mean(rf_scores), np.std(rf_scores)
    
    print(f"\nFinal CV Results:")
    print(f"  LR CV accuracy: {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")
    print(f"  RF CV accuracy: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")

    # ── Final Training & Saving (On all available data) ───────────────────────
    print("\nFitting final models on the entire dataset …")
    
    # Fit and save the scaler for predict.py
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X_arr)
    joblib.dump(final_scaler, SCALER_PATH)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    # Train and save the final Logistic Regression model
    lr_final = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    lr_final.fit(X_scaled_full, y_arr, sample_weight=weights_arr)
    joblib.dump(lr_final, LR_PATH)
    print(f"  ✓ LR model saved → {LR_PATH}")

    # Train and save the final Random Forest model
    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20, 
        max_features="sqrt", class_weight="balanced", 
        random_state=42, n_jobs=-1
    )
    rf_final.fit(X_scaled_full, y_arr, sample_weight=weights_arr)
    joblib.dump(rf_final, RF_PATH)
    print(f"  ✓ RF model saved → {RF_PATH}")

    # Feature Importance
    coef_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": lr_final.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    print("\n  LR Coefficients (top features):")
    print(coef_df.to_string(index=False))

    imp_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": rf_final.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\n  RF Feature Importances:")
    print(imp_df.to_string(index=False))

    results = {
        "lr_cv_mean": float(lr_cv_mean),
        "lr_cv_std": float(lr_cv_std),
        "rf_cv_mean": float(rf_cv_mean),
        "rf_cv_std": float(rf_cv_std),
        "n_games": len(X),
    }
    print(f"\n✓ Training complete.  {results['n_games']:,} games used.")
    return results


def load_models():
    """Load and return (lr, rf, scaler). Raises FileNotFoundError if not trained yet."""
    for path in (LR_PATH, RF_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python train.py` first to train the models."
            )
    lr = joblib.load(LR_PATH)
    rf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return lr, rf, scaler


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
