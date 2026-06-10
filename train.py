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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    Build features, evaluate via CV without data leakage, train final models on full data, and save.

    Parameters
    ----------
    raw_df : optional pre-loaded raw games DataFrame.
             If None, fetch_all_seasons() is called automatically.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if raw_df is None:
        raw_df = fetch_all_seasons()

    print("Building feature matrix …")
    X, y, weights = build_feature_matrix(raw_df)

    # TimeSeriesSplit keeps chronological order, preventing future data leakage
    cv = TimeSeriesSplit(n_splits=5)

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    print("\nEvaluating Logistic Regression (with Pipeline to prevent leakage) …")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    
    # The Pipeline ensures scaling is done only on the training folds
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", lr_model)
    ])

    # Using 'params' for sklearn >= 1.4
    lr_cv = cross_val_score(
        lr_pipeline, X, y,
        cv=cv,
        scoring="accuracy",
        params={"lr__sample_weight": weights},
    )
    print(f"  CV accuracy: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\nEvaluating Random Forest (with Pipeline) …")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", rf_model)
    ])

    rf_cv = cross_val_score(
        rf_pipeline, X, y,
        cv=cv,
        scoring="accuracy",
        params={"rf__sample_weight": weights},
    )
    print(f"  CV accuracy: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # ── 3. Final Training & Saving (On all available data) ────────────────────
    print("\nFitting final models on the entire dataset …")
    
    # Fit and save the scaler for predict.py
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    # Train and save the final Logistic Regression model
    lr_model.fit(X_scaled, y, sample_weight=weights)
    joblib.dump(lr_model, LR_PATH)
    print(f"  ✓ LR model saved → {LR_PATH}")

    # Train and save the final Random Forest model
    rf_model.fit(X_scaled, y, sample_weight=weights)
    joblib.dump(rf_model, RF_PATH)
    print(f"  ✓ RF model saved → {RF_PATH}")

    # Feature Importance
    coef_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": lr_model.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    print("\n  LR Coefficients (top features):")
    print(coef_df.to_string(index=False))

    imp_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\n  RF Feature Importances:")
    print(imp_df.to_string(index=False))

    results = {
        "lr_cv_mean": float(lr_cv.mean()),
        "lr_cv_std": float(lr_cv.std()),
        "rf_cv_mean": float(rf_cv.mean()),
        "rf_cv_std": float(rf_cv.std()),
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

