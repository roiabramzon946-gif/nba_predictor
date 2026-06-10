"""
daily_run.py
============
Cron-ready orchestrator.  Run this every morning before games start.

What it does, in order:
  1. Check whether the NBA regular season is currently active.
  2. Pull any new game results since the last run into the local cache.
  3. Retrain both models on the updated dataset.
  4. Predict today's games and write the HTML dashboard.
  5. Print a summary to stdout (captured by cron logs).

Recommended cron schedule (9 AM local time every day):
    0 9 * * * /path/to/venv/bin/python /path/to/nba_predictor/daily_run.py >> /path/to/nba_predictor/logs/daily.log 2>&1

Usage:
    python daily_run.py                # normal morning run
    python daily_run.py --force-train  # retrain even if no new games
    python daily_run.py --date 2025-03-15   # run for a specific date
"""

import os
import sys
import argparse
import logging
from datetime import date, datetime

import pandas as pd

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from fetch_data import fetch_all_seasons, update_with_new_games, GAMES_FILE
from train import train
from predict import run_predictions

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "daily.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)


# ── Season helpers ────────────────────────────────────────────────────────────

# Approximate regular-season date ranges (month/day, inclusive)
# Oct 1 – Apr 20 of the following year
REG_SEASON_START_MONTH = 10   # October
REG_SEASON_START_DAY = 1
REG_SEASON_END_MONTH = 4      # April
REG_SEASON_END_DAY = 20


def is_regular_season(d: date = None) -> bool:
    """
    Heuristic check — returns True if `d` falls within the usual
    regular-season window (Oct 1 – Apr 20).
    This avoids re-training and predicting during the off-season.
    """
    if d is None:
        d = date.today()
    m = d.month
    # Oct–Dec: start of season
    if m >= REG_SEASON_START_MONTH:
        return True
    # Jan–Apr: end of season
    if m < REG_SEASON_END_MONTH:
        return True
    if m == REG_SEASON_END_MONTH and d.day <= REG_SEASON_END_DAY:
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run(target_date: date = None, force_train: bool = False) -> None:
    if target_date is None:
        target_date = date.today()

    log.info("=" * 60)
    log.info(f"NBA Predictor — daily run for {target_date.isoformat()}")
    log.info("=" * 60)

    # ── 1. Season check ───────────────────────────────────────────────────────
    if not is_regular_season(target_date):
        log.info("Off-season detected — no regular-season games expected.")
        log.info("Skipping update and prediction.  Will resume in October.")
        return

    # ── 2. Update game data ───────────────────────────────────────────────────
    log.info("Step 1/3 — Updating game data …")
    new_games_added = False
    try:
        if os.path.exists(GAMES_FILE):
            raw_df = pd.read_csv(GAMES_FILE, parse_dates=["GAME_DATE"])
            before = len(raw_df)
            raw_df = update_with_new_games(raw_df)
            after = len(raw_df)
            new_games_added = after > before
            log.info(f"  {after - before} new game rows ingested (total: {after:,}).")
        else:
            log.info("  No cache found — performing full fetch for all 6 seasons …")
            raw_df = fetch_all_seasons()
            new_games_added = True
    except Exception as exc:
        log.error(f"  Data fetch failed: {exc}", exc_info=True)
        log.warning("  Proceeding with existing cache (if any) …")
        if os.path.exists(GAMES_FILE):
            raw_df = pd.read_csv(GAMES_FILE, parse_dates=["GAME_DATE"])
        else:
            log.error("  No data available — aborting.")
            return

    # ── 3. Retrain ────────────────────────────────────────────────────────────
    log.info("Step 2/3 — Training models …")
    if new_games_added or force_train:
        try:
            results = train(raw_df=raw_df)
            log.info(
                f"  LR  CV acc: {results['lr_cv_mean']:.4f} ± {results['lr_cv_std']:.4f}"
            )
            log.info(
                f"  RF  CV acc: {results['rf_cv_mean']:.4f} ± {results['rf_cv_std']:.4f}"
            )
        except Exception as exc:
            log.error(f"  Training failed: {exc}", exc_info=True)
            log.warning("  Proceeding with previously saved models …")
    else:
        log.info("  No new games — skipping retrain (use --force-train to override).")

    # ── 4. Predict + write HTML ───────────────────────────────────────────────
    log.info("Step 3/3 — Generating predictions …")
    try:
        html_path = run_predictions(target_date)
        if html_path:
            log.info(f"  Dashboard → {html_path}")
        else:
            log.info("  No games today — no dashboard generated.")
    except Exception as exc:
        log.error(f"  Prediction failed: {exc}", exc_info=True)

    log.info("Daily run complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA predictor — daily update + predict")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain models even if no new game data was found",
    )
    args = parser.parse_args()

    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date else None
    )
    run(target_date=target, force_train=args.force_train)
