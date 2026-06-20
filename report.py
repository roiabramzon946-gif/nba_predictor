"""
report.py
=========
Generates an accuracy and spread evaluation report for a specific calendar year.
Uses cached data and saved models — no API calls required.

Usage:
    python report.py          # defaults to current year (2026)
    python report.py 2025     # run for 2025
    python report.py 2024     # run for 2024

Note: these are IN-SAMPLE results (models were trained on this data).
For honest out-of-sample accuracy, refer to the CV scores printed by train.py.
"""

import warnings
import argparse
import pandas as pd
import numpy as np
from train import load_models
import features
from fetch_data import GAMES_FILE, PLAYER_LOGS_FILE

warnings.filterwarnings("ignore")


def generate_report(year: int):
    print(f"Loading models and cached data...")
    try:
        lr, rf, scaler, margin_model = load_models()
        raw_df = pd.read_csv(GAMES_FILE, parse_dates=["GAME_DATE"])
        player_logs = pd.read_csv(PLAYER_LOGS_FILE)
    except FileNotFoundError:
        print("Error: Missing data or models. Run fetch_data.py and train.py first.")
        return

    print("Building full historical feature matrix (this takes a few seconds)...")
    X, _, _, margins = features.build_feature_matrix(raw_df, player_logs_df=player_logs)

    # Reconstruct game metadata (GAME_DATE, home_win) in the exact same row
    # order that build_feature_matrix produces. Both sides must sort by
    # ["GAME_DATE", "GAME_ID"] so games sharing the same date are always
    # resolved deterministically — otherwise pandas' stable sort produces
    # a different tie-breaking order here vs. inside build_feature_matrix,
    # misaligning predictions with outcomes.
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_binary"] = (df["WL"] == "W").astype(int)
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

    home = (
        df[df["is_home"] == 1][["GAME_ID", "GAME_DATE", "WL_binary"]]
        .rename(columns={"WL_binary": "home_win"})
    )
    away = (
        df[df["is_home"] == 0][["GAME_ID", "WL_binary"]]
        .rename(columns={"WL_binary": "away_win"})
    )

    paired = (
        home.merge(away, on="GAME_ID")
        .sort_values(["GAME_DATE", "GAME_ID"])   # deterministic secondary key
        .reset_index(drop=True)
    )
    paired = paired[paired["home_win"] + paired["away_win"] == 1].copy()

    # Run predictions across all historical data
    X_scaled = scaler.transform(X)
    paired["pred_home_prob_ens"] = (
        lr.predict_proba(X_scaled)[:, 1] + rf.predict_proba(X_scaled)[:, 1]
    ) / 2
    paired["pred_home_win"] = (paired["pred_home_prob_ens"] >= 0.5).astype(int)
    paired["actual_margin"] = margins.values
    paired["pred_margin"] = margin_model.predict(X_scaled)

    # Filter to the requested calendar year
    df_year = paired[paired["GAME_DATE"].dt.year == year].copy()

    if df_year.empty:
        print(f"\nNo completed games found for {year}.")
        return

    # --- Compute report statistics ---
    n_games = len(df_year)

    # 1. Winner accuracy
    df_year["correct_winner"] = (df_year["pred_home_win"] == df_year["home_win"]).astype(int)
    win_acc = df_year["correct_winner"].mean() * 100

    # 2. Spread error
    rmse = np.sqrt(((df_year["pred_margin"] - df_year["actual_margin"]) ** 2).mean())
    mae = (df_year["pred_margin"] - df_year["actual_margin"]).abs().mean()

    # 3. Spread proximity
    df_year["margin_error"] = (df_year["pred_margin"] - df_year["actual_margin"]).abs()
    within_5 = (df_year["margin_error"] <= 5).mean() * 100
    within_10 = (df_year["margin_error"] <= 10).mean() * 100

    print("\n" + "=" * 50)
    print(f"📊 NBA PREDICTOR REPORT — {year}  (in-sample)")
    print(f"🏀 Games evaluated: {n_games}")
    print("=" * 50)
    print(f"🏆 Winner Accuracy:      {win_acc:.1f}%")
    print(f"📉 Spread RMSE:          {rmse:.2f} pts")
    print(f"📉 Spread MAE:           {mae:.2f} pts")
    print("-" * 50)
    print(f"🎯 Margin within  5 pts: {within_5:.1f}% of games")
    print(f"🎯 Margin within 10 pts: {within_10:.1f}% of games")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Predictor accuracy report.")
    parser.add_argument(
        "year",
        type=int,
        nargs="?",
        default=2026,
        help="Calendar year to evaluate (e.g. 2024, 2025, 2026)",
    )
    args = parser.parse_args()
    generate_report(year=args.year)
