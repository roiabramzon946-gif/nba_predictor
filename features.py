"""
features.py
===========
Transforms raw nba_api game-log rows into a model-ready feature matrix.

Each row in the output represents ONE game from the HOME team's perspective:
    • Target  : home_win  (1 = home team won, 0 = away team won)
    • Features:
        - month                  : calendar month of the game (1–12)
        - home_win_pct_at_home   : home team's season win% AT HOME BEFORE this game
        - away_win_pct_on_road   : away team's season win% ON THE ROAD before this game
        - win_pct_diff           : home_win_pct_at_home − away_win_pct_on_road
        - home_form5             : home team's win rate in last 5 games (any opponent)
        - away_form5             : away team's win rate in last 5 games
        - form5_diff             : home_form5 − away_form5
        - home_h2h               : home team's win rate vs THIS away team (all-time in data)
        - away_h2h               : away team's win rate vs THIS home team (= 1 − home_h2h)
        - home_rest_days         : home team's rest days before this game (0 for B2B, max 7)
        - away_rest_days         : away team's rest days before this game (0 for B2B, max 7)
        - season_weight          : recency weight used during training (not a model feature)

Usage:
    from features import build_feature_matrix
    X, y, weights, margins = build_feature_matrix(raw_df)
"""

import numpy as np
import pandas as pd

# Feature columns fed to the model (order matters for saved models)
FEATURE_COLS = [
    "month",
    "home_win_pct_at_home",
    "away_win_pct_on_road",
    "win_pct_diff",
    "home_form5",
    "away_form5",
    "form5_diff",
    "home_h2h",
    "away_h2h",
    "home_rest_days",
    "away_rest_days",
    "home_missing_mins",   # total avg mins/game of injured-out home players
    "away_missing_mins",   # total avg mins/game of injured-out away players
]

# Prior "pseudo-games" added to every team's record before computing
# season win% (home_win_pct_at_home / away_win_pct_on_road). Without this,
# a team's win% after just 1-2 games of a new season is either 0% or 100%,
# which is an extreme/noisy signal. Adding 4 fictional wins + 4 fictional
# losses (a neutral 50% prior, worth 8 "games") smooths this out — it gets
# diluted toward the team's real record as the season goes on.
WIN_PCT_PRIOR_WINS = 4
WIN_PCT_PRIOR_LOSSES = 4
WIN_PCT_PRIOR_GAMES = WIN_PCT_PRIOR_WINS + WIN_PCT_PRIOR_LOSSES

# Players averaging fewer than this many minutes per game are not considered
# "key" rotation players — their absence doesn't register as missing minutes.
KEY_PLAYER_MIN_THRESHOLD = 15.0

# A player must have logged at least this many games for a team in a season
# before they are considered "established" and their absence is tracked.
# This excludes G-League call-ups, 10-day contracts, and mid-season trade
# pickups who haven't yet played enough games to be reliable rotation pieces.
KEY_PLAYER_MIN_GAMES = 30


# Map season string → exponential recency weight (oldest = 1, newest = 32)
# Pattern: [1, 2, 4, 8, 16, 32] — each season is weighted 2× the one before it
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
        weights[season] = 2 ** (num_seasons - 1 - i)
    return weights

SEASON_WEIGHTS = _build_season_weights(num_seasons=6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_win_pct(team_games: pd.DataFrame) -> pd.Series:
    """
    Cumulative win% before each game, with a Bayesian prior of
    WIN_PCT_PRIOR_WINS wins and WIN_PCT_PRIOR_LOSSES losses added to every
    team's record (see constants above). The first game of a season now
    starts at exactly 0.5 (4 / 8), and each additional game's win% moves
    gradually toward the team's real record instead of jumping straight
    to 0% or 100%.
    """
    wins = team_games["WL_binary"].values
    cum_wins = np.concatenate([[0], np.cumsum(wins[:-1])])
    cum_games = np.arange(len(wins))

    pct = (cum_wins + WIN_PCT_PRIOR_WINS) / (cum_games + WIN_PCT_PRIOR_GAMES)
    return pd.Series(pct, index=team_games.index)


def _rolling_form5(team_games: pd.DataFrame) -> pd.Series:
    form = team_games["WL_binary"].shift(1).rolling(window=5, min_periods=1).mean()
    return form.fillna(0.5)


def _calculate_rest_days(team_games: pd.DataFrame) -> pd.Series:
    diff_days = (team_games["GAME_DATE"] - team_games["GAME_DATE"].shift(1)).dt.days
    rest_days = diff_days - 1
    rest_days = rest_days.fillna(3)
    return rest_days.clip(lower=0, upper=7)


def _h2h_win_rate(games_df: pd.DataFrame) -> pd.Series:
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

        records[key] = [wins + int(row["home_win"]), total + 1]

    return pd.Series(h2h, index=games_df.index)


def compute_missing_mins(player_logs_df: pd.DataFrame) -> dict:
    """
    Pre-compute, for every (GAME_ID, TEAM_ID) pair that appears in the player
    logs, the total average minutes of key players who did NOT appear in that game.

    A player is "key" if they logged at least KEY_PLAYER_MIN_GAMES games AND
    averaged at least KEY_PLAYER_MIN_THRESHOLD minutes per game for that team
    in that season.  The two-part threshold filters out:
      - G-League call-ups and 10-day contracts (too few games)
      - End-of-bench players (too few minutes)
      - Mid-season trade acquisitions who haven't established themselves yet

    Temporal guard: we only expect a player at games on or after their FIRST
    game with that team-season.  Without this, a player traded in at game 50
    would be marked as "missing" from all 49 games before they arrived —
    generating completely spurious missing-minutes totals.

    Returns
    -------
    dict mapping (game_id_str, team_id_int) → float (total missing minutes).
    Games where every key player appeared will be absent from the dict
    (look up with .get(..., 0.0)).
    """
    logs = player_logs_df.copy()
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    logs["TEAM_ID"] = logs["TEAM_ID"].astype(int)
    logs["PLAYER_ID"] = logs["PLAYER_ID"].astype(int)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    # ── Step 1: compute each player's profile for each (team, season) ─────────
    # avg_min is over the games they actually played — not the full roster.
    # first_date is when they first appeared for this team-season (arrival date).
    player_profiles = (
        logs.groupby(["PLAYER_ID", "TEAM_ID", "SEASON"])
        .agg(
            games_played=("GAME_ID", "count"),
            avg_min=("MIN_float", "mean"),
            first_date=("GAME_DATE", "min"),
        )
        .reset_index()
    )

    # ── Step 2: filter to established key players only ─────────────────────────
    key_players = player_profiles[
        (player_profiles["games_played"] >= KEY_PLAYER_MIN_GAMES)
        & (player_profiles["avg_min"] >= KEY_PLAYER_MIN_THRESHOLD)
    ].copy()

    # ── Step 3: all (GAME_ID, TEAM_ID, SEASON, GAME_DATE) combos ──────────────
    game_team_info = (
        logs[["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]]
        .drop_duplicates()
    )

    # ── Step 4: expand — pair each key player with every game their team played
    expanded = key_players.merge(game_team_info, on=["TEAM_ID", "SEASON"], how="inner")

    # Temporal guard: only expect the player from the date they first appeared
    expanded = expanded[expanded["GAME_DATE"] >= expanded["first_date"]]

    # ── Step 5: anti-join — key players who did NOT appear in a game ──────────
    appeared = (
        logs[logs["MIN_float"] > 0][["GAME_ID", "TEAM_ID", "PLAYER_ID"]]
        .drop_duplicates()
    )
    appeared["_appeared"] = True

    merged = expanded.merge(
        appeared, on=["GAME_ID", "TEAM_ID", "PLAYER_ID"], how="left"
    )
    missing = merged[merged["_appeared"].isna()]

    # ── Step 6: sum missing minutes per (GAME_ID, TEAM_ID) ───────────────────
    result = (
        missing.groupby(["GAME_ID", "TEAM_ID"])["avg_min"]
        .sum()
        .to_dict()
    )
    return {(gid, int(tid)): v for (gid, tid), v in result.items()}


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_feature_matrix(
    raw_df: pd.DataFrame,
    player_logs_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_binary"] = (df["WL"] == "W").astype(int)

    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    df["month"] = df["GAME_DATE"].dt.month

    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    win_pct_home_map: dict = {}
    win_pct_road_map: dict = {}
    form5_map: dict = {}
    rest_days_map: dict = {}

    for (team_id, season), group in df.groupby(["TEAM_ID", "SEASON"]):
        # Split to home and road for precise win percentages
        home_group = group[group["is_home"] == 1].sort_values("GAME_DATE")
        if not home_group.empty:
            wp_home = _rolling_win_pct(home_group)
            for idx, w in zip(home_group.index, wp_home):
                win_pct_home_map[idx] = w

        road_group = group[group["is_home"] == 0].sort_values("GAME_DATE")
        if not road_group.empty:
            wp_road = _rolling_win_pct(road_group)
            for idx, w in zip(road_group.index, wp_road):
                win_pct_road_map[idx] = w

    for team_id, group in df.groupby("TEAM_ID"):
        group_sorted = group.sort_values("GAME_DATE")
        f5 = _rolling_form5(group_sorted)
        rd = _calculate_rest_days(group_sorted)
        for idx, f in zip(group_sorted.index, f5):
            form5_map[idx] = f
        for idx, r in zip(group_sorted.index, rd):
            rest_days_map[idx] = r

    _prior = WIN_PCT_PRIOR_WINS / WIN_PCT_PRIOR_GAMES
    df["win_pct_at_home_before"] = df.index.map(win_pct_home_map).fillna(_prior)
    df["win_pct_on_road_before"] = df.index.map(win_pct_road_map).fillna(_prior)
    df["form5"] = df.index.map(form5_map)
    df["rest_days_before"] = df.index.map(rest_days_map)

    home = df[df["is_home"] == 1][[
        "GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON", "month",
        "WL_binary", "win_pct_at_home_before", "form5", "rest_days_before", "PTS"
    ]].rename(columns={
        "TEAM_ID": "home_team_id",
        "WL_binary": "home_win",
        "win_pct_at_home_before": "home_win_pct_at_home",
        "form5": "home_form5",
        "rest_days_before": "home_rest_days",
        "PTS": "home_pts",
    })

    away = df[df["is_home"] == 0][[
        "GAME_ID", "TEAM_ID",
        "WL_binary", "win_pct_on_road_before", "form5", "rest_days_before", "PTS"
    ]].rename(columns={
        "TEAM_ID": "away_team_id",
        "WL_binary": "away_win",
        "win_pct_on_road_before": "away_win_pct_on_road",
        "form5": "away_form5",
        "rest_days_before": "away_rest_days",
        "PTS": "away_pts",
    })

    paired = home.merge(away, on="GAME_ID").sort_values("GAME_DATE").reset_index(drop=True)
    paired = paired[paired["home_win"] + paired["away_win"] == 1].copy()

    paired["home_h2h"] = _h2h_win_rate(paired)
    paired["away_h2h"] = 1.0 - paired["home_h2h"]

    paired["win_pct_diff"] = paired["home_win_pct_at_home"] - paired["away_win_pct_on_road"]
    paired["form5_diff"] = paired["home_form5"] - paired["away_form5"]

    paired["season_weight"] = paired["SEASON"].map(SEASON_WEIGHTS).fillna(1)

    # Point differential from the home team's perspective (positive = home team won by that margin)
    paired["point_diff"] = paired["home_pts"] - paired["away_pts"]

    # ── Injury / missing-player minutes ───────────────────────────────────────
    # If player_logs_df is provided, compute how many avg-minutes worth of key
    # players each team was missing for each game.  Defaults to 0.0 otherwise
    # so the feature always exists and models trained without it are still valid
    # (though they'll benefit from retraining once player logs are available).
    if player_logs_df is not None and not player_logs_df.empty:
        print("  Computing missing-player minutes …")
        missing_map = compute_missing_mins(player_logs_df)
        paired["GAME_ID_str"] = paired["GAME_ID"].astype(str)
        paired["home_missing_mins"] = paired.apply(
            lambda r: missing_map.get((r["GAME_ID_str"], int(r["home_team_id"])), 0.0),
            axis=1,
        )
        paired["away_missing_mins"] = paired.apply(
            lambda r: missing_map.get((r["GAME_ID_str"], int(r["away_team_id"])), 0.0),
            axis=1,
        )
    else:
        paired["home_missing_mins"] = 0.0
        paired["away_missing_mins"] = 0.0

    X = paired[FEATURE_COLS].astype(float)
    y = paired["home_win"].astype(int)
    weights = paired["season_weight"].astype(float)
    margins = paired["point_diff"].astype(float)

    print(f"✓ Feature matrix: {X.shape[0]:,} games × {X.shape[1]} features")
    return X, y, weights, margins


def build_prediction_row(
    home_team_id: int,
    away_team_id: int,
    game_date,
    historical_df: pd.DataFrame,
    home_missing_mins: float = 0.0,
    away_missing_mins: float = 0.0,
) -> pd.DataFrame:
    game_date = pd.Timestamp(game_date).normalize()
    month = game_date.month

    hist_df = historical_df.copy()
    hist_df["GAME_DATE"] = pd.to_datetime(hist_df["GAME_DATE"]).dt.normalize()
    hist_df["WL_binary"] = (hist_df["WL"] == "W").astype(int)
    hist_df["is_home"] = hist_df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

    # ── Home team stats ───────────────────────────────────────────────────────
    home_hist = hist_df[hist_df["TEAM_ID"] == home_team_id].sort_values("GAME_DATE")
    home_hist = home_hist[home_hist["GAME_DATE"] < game_date]

    home_season = home_hist["SEASON"].max() if len(home_hist) > 0 else None
    
    # Home win% at home this season, using the same +4W/+4L prior as
    # build_feature_matrix so training and prediction are consistent.
    home_hist_at_home = home_hist[home_hist["is_home"] == 1]
    home_season_games_at_home = home_hist_at_home[home_hist_at_home["SEASON"] == home_season]
    home_wins_at_home = float(home_season_games_at_home["WL_binary"].sum())
    home_games_at_home = len(home_season_games_at_home)
    home_win_pct_at_home = (home_wins_at_home + WIN_PCT_PRIOR_WINS) / (home_games_at_home + WIN_PCT_PRIOR_GAMES)
    
    home_form5 = float(home_hist["WL_binary"].tail(5).mean()) if len(home_hist) > 0 else 0.5
    
    if not home_hist.empty:
        last_home_game = home_hist["GAME_DATE"].max()
        home_rest = (game_date - last_home_game).days - 1
        home_rest = min(max(home_rest, 0), 7)
    else:
        home_rest = 3.0

    # ── Away team stats ───────────────────────────────────────────────────────
    away_hist = hist_df[hist_df["TEAM_ID"] == away_team_id].sort_values("GAME_DATE")
    away_hist = away_hist[away_hist["GAME_DATE"] < game_date]

    away_season = away_hist["SEASON"].max() if len(away_hist) > 0 else None
    
    # Away win% on the road this season (same +4W/+4L prior as above).
    away_hist_on_road = away_hist[away_hist["is_home"] == 0]
    away_season_games_on_road = away_hist_on_road[away_hist_on_road["SEASON"] == away_season]
    away_wins_on_road = float(away_season_games_on_road["WL_binary"].sum())
    away_games_on_road = len(away_season_games_on_road)
    away_win_pct_on_road = (away_wins_on_road + WIN_PCT_PRIOR_WINS) / (away_games_on_road + WIN_PCT_PRIOR_GAMES)
    
    away_form5 = float(away_hist["WL_binary"].tail(5).mean()) if len(away_hist) > 0 else 0.5
    
    if not away_hist.empty:
        last_away_game = away_hist["GAME_DATE"].max()
        away_rest = (game_date - last_away_game).days - 1
        away_rest = min(max(away_rest, 0), 7)
    else:
        away_rest = 3.0

    # ── H2H ───────────────────────────────────────────────────────────────────
    shared_game_ids = set(home_hist["GAME_ID"]).intersection(set(away_hist["GAME_ID"]))
    if len(shared_game_ids) > 0:
        h2h_games = home_hist[home_hist["GAME_ID"].isin(shared_game_ids)]
        home_h2h = float(h2h_games["WL_binary"].mean())
    else:
        home_h2h = 0.5

    row = {
        "month": float(month),
        "home_win_pct_at_home": float(home_win_pct_at_home),
        "away_win_pct_on_road": float(away_win_pct_on_road),
        "win_pct_diff": float(home_win_pct_at_home - away_win_pct_on_road),
        "home_form5": float(home_form5),
        "away_form5": float(away_form5),
        "form5_diff": float(home_form5 - away_form5),
        "home_h2h": float(home_h2h),
        "away_h2h": float(1.0 - home_h2h),
        "home_rest_days": float(home_rest),
        "away_rest_days": float(away_rest),
        "home_missing_mins": float(home_missing_mins),
        "away_missing_mins": float(away_missing_mins),
    }
    return pd.DataFrame([row])[FEATURE_COLS]