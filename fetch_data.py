"""
fetch_data.py
=============
Fetches regular-season game logs for the last 6 NBA seasons using nba_api.
Results are cached locally so subsequent runs don't re-hit the API unless
you pass force_refresh=True (or delete the cache file).

Usage (standalone):
    python fetch_data.py              # uses cache if available
    python fetch_data.py --refresh    # forces a fresh pull from the API
"""

import os
import time
import argparse
from datetime import date
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, leaguegamelog

# ── Configuration ────────────────────────────────────────────────────────────

def _current_nba_season() -> str:
    """
    Return the current NBA season string (e.g. '2025-26').
    The NBA season starts in October; if we're in Oct–Dec the season year
    is the current calendar year, otherwise it's the previous calendar year.
    """
    today = date.today()
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _build_seasons(num_seasons: int = 6) -> list:
    """
    Build a list of the last `num_seasons` NBA season strings ending with
    the current season, e.g. ['2020-21', '2021-22', ..., '2025-26'].
    """
    today = date.today()
    if today.month >= 10:
        end_year = today.year
    else:
        end_year = today.year - 1
    seasons = []
    for i in range(num_seasons - 1, -1, -1):
        start = end_year - i
        seasons.append(f"{start}-{str(start + 1)[-2:]}")
    return seasons


# Automatically computed — always ends with the current NBA season
SEASONS = _build_seasons(num_seasons=6)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GAMES_FILE = os.path.join(DATA_DIR, "games_raw.csv")
PLAYER_LOGS_FILE = os.path.join(DATA_DIR, "player_game_logs.csv")

# Seconds to wait between API calls to avoid rate-limiting
API_SLEEP = 1.5


# ── Core Functions ────────────────────────────────────────────────────────────

def fetch_season(season: str) -> pd.DataFrame:
    """
    Pull all regular-season game rows for one season from LeagueGameFinder.
    Each row is one team's record for one game (two rows per actual game).
    """
    print(f"  Fetching season {season} from nba_api …", flush=True)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    time.sleep(API_SLEEP)
    return df


def fetch_all_seasons(
    seasons: list = SEASONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of all game rows for every season in `seasons`.

    If `data/games_raw.csv` already exists and force_refresh is False,
    the cached file is returned immediately.  Otherwise every season is
    fetched from nba_api and the result is saved to disk.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(GAMES_FILE) and not force_refresh:
        print(f"✓ Loading cached game data from {GAMES_FILE}")
        df = pd.read_csv(GAMES_FILE)
        print(f"  {len(df):,} rows across {df['SEASON'].nunique()} seasons.")
        return df

    print(f"Fetching {len(seasons)} seasons from nba_api …")
    dfs = []
    for season in seasons:
        season_df = fetch_season(season)
        dfs.append(season_df)
        print(f"  ✓ {season}: {len(season_df):,} rows")

    all_games = pd.concat(dfs, ignore_index=True)

    # Normalise column types
    all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])

    all_games.to_csv(GAMES_FILE, index=False)
    print(f"\n✓ Saved {len(all_games):,} rows to {GAMES_FILE}")
    return all_games


def update_with_new_games(existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch only the current season and merge any game rows that aren't
    already in `existing_df`.  Called by daily_run.py every morning to
    incorporate last night's results without re-pulling 5+ old seasons.
    """
    current_season = SEASONS[-1]
    fresh = fetch_season(current_season)
    fresh["GAME_DATE"] = pd.to_datetime(fresh["GAME_DATE"])

    # Identify new rows by GAME_ID
    existing_ids = set(existing_df["GAME_ID"].astype(str))
    new_rows = fresh[~fresh["GAME_ID"].astype(str).isin(existing_ids)]

    if new_rows.empty:
        print("  No new games found since last update.")
        return existing_df

    print(f"  Adding {len(new_rows)} new game rows from {current_season}.")
    updated = pd.concat([existing_df, new_rows], ignore_index=True)
    updated.to_csv(GAMES_FILE, index=False)
    return updated


# ── Player game logs ─────────────────────────────────────────────────────────

def _parse_minutes(min_val) -> float:
    """
    Convert a minutes value to a float.
    nba_api can return minutes as a 'MM:SS' string or as a float depending
    on the endpoint version — this handles both.
    """
    if min_val is None or (isinstance(min_val, float) and pd.isna(min_val)):
        return 0.0
    s = str(min_val).strip()
    if not s or s in ("None", "nan"):
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_player_game_logs(
    seasons: list = SEASONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch player-level game logs for all seasons using LeagueGameLog.
    Cached to data/player_game_logs.csv — only re-fetched if force_refresh=True
    or the file is missing.

    Returns a DataFrame with columns:
        PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID, GAME_DATE, SEASON, MIN_float
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PLAYER_LOGS_FILE) and not force_refresh:
        print(f"✓ Loading cached player game logs from {PLAYER_LOGS_FILE}")
        df = pd.read_csv(PLAYER_LOGS_FILE)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        print(f"  {len(df):,} player-game rows across {df['SEASON'].nunique()} seasons.")
        return df

    print(f"Fetching player game logs for {len(seasons)} seasons from nba_api …")
    dfs = []
    for season in seasons:
        print(f"  Fetching {season} …", flush=True)
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            league_id="00",
        )
        df = log.get_data_frames()[0]
        df["SEASON"] = season
        dfs.append(df)
        print(f"    ✓ {len(df):,} player-game rows")
        time.sleep(API_SLEEP)

    all_logs = pd.concat(dfs, ignore_index=True)
    all_logs["GAME_DATE"] = pd.to_datetime(all_logs["GAME_DATE"])
    all_logs["MIN_float"] = all_logs["MIN"].apply(_parse_minutes)

    # Keep only the columns needed downstream — discard box-score stats
    keep_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID",
                 "GAME_DATE", "SEASON", "MIN_float"]
    all_logs = all_logs[[c for c in keep_cols if c in all_logs.columns]]
    all_logs.to_csv(PLAYER_LOGS_FILE, index=False)
    print(f"✓ Saved {len(all_logs):,} player-game rows to {PLAYER_LOGS_FILE}")
    return all_logs


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game data")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache and re-fetch all seasons from nba_api",
    )
    args = parser.parse_args()
    fetch_all_seasons(force_refresh=args.refresh)
