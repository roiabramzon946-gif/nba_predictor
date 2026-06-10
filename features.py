"""
features.py
===========
Transforms raw nba_api game-log rows into a model-ready feature matrix.

Each row in the output represents ONE game from the HOME team's perspective:
    • Target  : home_win  (1 = home team won, 0 = away team won)
    • Features:
        - home_advantage   : always 1 (flag; keeps parity with prediction-time rows)
        - month            : calendar month of the game (1–12)
        - home_win_pct     : home team's season win% BEFORE this game
        - away_win_pct     : away team's season win% before this game
        - win_pct_diff     : home_win_pct − away_win_pct
        - home_form5       : home team's win rate in last 5 games (any opponent)
        - away_form5       : away team's win rate in last 5 games
        - form5_diff       : home_form5 − away_form5
        - home_h2h         : home team's win rate vs THIS away team (all-time in data)
        - away_h2h         : away team's win rate vs THIS home team (= 1 − home_h2h)
        - season_weight    : recency weight used during training (not a model feature)

Usage:
    from features import build_feature_matrix
    X, y, weights = build_feature_matrix(raw_df)
"""

import numpy as np
import pandas as pd

# Feature columns fed to the model (order matters for saved models)
FEATURE_COLS = [
    "home_advantage",
    "month",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_form5",
    "away_form5",
    "form5_diff",
    "home_h2h",
    "away_h2h",
]

# Map season string → numeric recency weight (oldest = 1, newest = 6)
def _build_season_weights(num_seasons: int = 6) -> dict:
    from datetime import date
    today = date.today()
    if today.month >= 10:
        end_year = today.year
    else:
        end_year = today.year - 1
    weights = {}
    for i in range(num_seasons - 1, -1, -1):
        start = end_year - i
        season = f"{start}-{str(start + 1)[-2:]}"
        weights[season] = num_seasons - i
    return weights

SEASON_WEIGHTS = _build_season_weights(num_seasons=6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_win_pct(team_games: pd.DataFrame) -> pd.Series:
    """
    For a single team's games (sorted by date), compute cumulative win%
    BEFORE each game. The very first game gets 0.5 as a prior.
    Returns a Series indexed like team_games.
    """
    wins = team_games["WL_binary"].values
    cum_wins = np.concatenate([[0], np.cumsum(wins[:-1])])
    cum_games = np.arange(len(wins))
    
    with np.errstate(invalid="ignore"):
        pct = np.where(cum_games == 0, 0.5, cum_wins / cum_games)
    return pd.Series(pct, index=team_games.index)


def _rolling_form5(team_games: pd.DataFrame) -> pd.Series:
    """
    Win rate in the last 5 games BEFORE each game.
    Optimized via pandas vectorization.
    """
    form = team_games["WL_binary"].shift(1).rolling(window=5, min_periods=1).mean()
    return form.fillna(0.5)


def _h2h_win_rate(games_df: pd.DataFrame) -> pd.Series:
    """
    For each row (a paired home vs away game), look at ALL earlier meetings
    between the same pair and return the home team's historical win rate.
    """
    h2h = []
    records: dict = {}

    for _, row in games_df.iterrows():
        key = (int(row["home_team_id"]), int(row["away_team_id"]))
        rev_key = (int(row["away_team_id"]), int(row["home_team_id"]))

        wins, total = records.get(key, [0, 0])
        rev_wins, rev_total = records.get(rev_key, [0, 0])

        combined_total = total + rev_total
        if combined_total == 0:
            h2h.append(0.5)
        else:
            combined_home_wins = wins + (rev_total - rev_wins)
            h2h.append(combined_home_wins / combined_total)

        # Update record for this game
        records[key] = [wins + int(row["home_win"]), total + 1]

    return pd.Series(h2h, index=games_df.index)


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_feature_matrix(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build model-ready features from raw nba_api game logs.
    """
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_binary"] = (df["WL"] == "W").astype(int)

    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    df["month"] = df["GAME_DATE"].dt.month

    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # ── Per-team rolling stats ────────────────────────────────────────────────
    win_pct_map: dict = {}
    form5_map: dict = {}

    # 1. WIN PERCENTAGE: Grouped by BOTH Team and Season (Resets every season)
    for (team_id, season), group in df.groupby(["TEAM_ID", "SEASON"]):
        group_sorted = group.sort_values("GAME_DATE")
        wp = _rolling_win_pct(group_sorted)
        for idx, w in zip(group_sorted.index, wp):
            win_pct_map[idx] = w

    # 2. FORM (Last 5 Games): Grouped by Team only (Carries over between seasons)
    for team_id, group in df.groupby("TEAM_ID"):
        group_sorted = group.sort_values("GAME_DATE")
        f5 = _rolling_form5(group_sorted)
        for idx, f in zip(group_sorted.index, f5):
            form5_map[idx] = f

    df["win_pct_before"] = df.index.map(win_pct_map)
    df["form5"] = df.index.map(form5_map)

    # ── Pair home and away rows by GAME_ID ───────────────────────────────────
    home = df[df["is_home"] == 1][[
        "GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON", "month",
        "WL_binary", "win_pct_before", "form5",
    ]].rename(columns={
        "TEAM_ID": "home_team_id",
        "WL_binary": "home_win",
        "win_pct_before": "home_win_pct",
        "form5": "home_form5",
    })

    away = df[df["is_home"] == 0][[
        "GAME_ID", "TEAM_ID",
        "WL_binary", "win_pct_before", "form5",
    ]].rename(columns={
        "TEAM_ID": "away_team_id",
        "WL_binary": "away_win",
        "win_pct_before": "away_win_pct",
        "form5": "away_form5",
    })

    paired = home.merge(away, on="GAME_ID").sort_values("GAME_DATE").reset_index(drop=True)

    # Sanity: home_win should be complement of away_win
    paired = paired[paired["home_win"] + paired["away_win"] == 1].copy()

    # ── Head-to-head ─────────────────────────────────────────────────────────
    paired["home_h2h"] = _h2h_win_rate(paired)
    paired["away_h2h"] = 1.0 - paired["home_h2h"]

    # ── Derived diffs ─────────────────────────────────────────────────────────
    paired["win_pct_diff"] = paired["home_win_pct"] - paired["away_win_pct"]
    paired["form5_diff"] = paired["home_form5"] - paired["away_form5"]
    paired["home_advantage"] = 1

    # ── Season recency weight ─────────────────────────────────────────────────
    paired["season_weight"] = paired["SEASON"].map(SEASON_WEIGHTS).fillna(1)

    X = paired[FEATURE_COLS].astype(float)
    y = paired["home_win"].astype(int)
    weights = paired["season_weight"].astype(float)

    print(f"✓ Feature matrix: {X.shape[0]:,} games × {X.shape[1]} features")
    return X, y, weights


def build_prediction_row(
    home_team_id: int,
    away_team_id: int,
    game_date,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single feature row for an UPCOMING game (no result yet).
    """
    # Normalize timestamps to safely align types
    game_date = pd.Timestamp(game_date).normalize()
    month = game_date.month

    hist_df = historical_df.copy()
    hist_df["GAME_DATE"] = pd.to_datetime(hist_df["GAME_DATE"]).dt.normalize()
    hist_df["WL_binary"] = (hist_df["WL"] == "W").astype(int)

    # ── Home team rolling stats ───────────────────────────────────────────────
    home_hist = hist_df[hist_df["TEAM_ID"] == home_team_id].sort_values("GAME_DATE")
    home_hist = home_hist[home_hist["GAME_DATE"] < game_date]

    home_season = home_hist["SEASON"].max() if len(home_hist) > 0 else None
    home_season_games = home_hist[home_hist["SEASON"] == home_season]
    
    home_win_pct = float(home_season_games["WL_binary"].mean()) if len(home_season_games) > 0 else 0.5
    home_form5 = float(home_hist["WL_binary"].tail(5).mean()) if len(home_hist) > 0 else 0.5

    # ── Away team rolling stats ───────────────────────────────────────────────
    away_hist = hist_df[hist_df["TEAM_ID"] == away_team_id].sort_values("GAME_DATE")
    away_hist = away_hist[away_hist["GAME_DATE"] < game_date]

    away_season = away_hist["SEASON"].max() if len(away_hist) > 0 else None
    away_season_games = away_hist[away_hist["SEASON"] == away_season]
    
    away_win_pct = float(away_season_games["WL_binary"].mean()) if len(away_season_games) > 0 else 0.5
    away_form5 = float(away_hist["WL_binary"].tail(5).mean()) if len(away_hist) > 0 else 0.5

    # ── H2H (Robust implementation using GAME_ID intersection) ────────────────
    shared_game_ids = set(home_hist["GAME_ID"]).intersection(set(away_hist["GAME_ID"]))
    
    if len(shared_game_ids) > 0:
        h2h_games = home_hist[home_hist["GAME_ID"].isin(shared_game_ids)]
        home_h2h = float(h2h_games["WL_binary"].mean())
    else:
        home_h2h = 0.5

    row = {
        "home_advantage": 1.0,
        "month": float(month),
        "home_win_pct": float(home_win_pct),
        "away_win_pct": float(away_win_pct),
        "win_pct_diff": float(home_win_pct - away_win_pct),
        "home_form5": float(home_form5),
        "away_form5": float(away_form5),
        "form5_diff": float(home_form5 - away_form5),
        "home_h2h": float(home_h2h),
        "away_h2h": float(1.0 - home_h2h),
    }
    return pd.DataFrame([row])[FEATURE_COLS]