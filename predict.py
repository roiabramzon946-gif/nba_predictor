"""
predict.py
==========
1. Fetches today's NBA schedule from ScoreboardV2.
2. Builds feature rows for each game using historical data.
3. Runs both LR and RF models, averages probabilities into an ensemble.
4. Writes a polished HTML dashboard to outputs/predictions_YYYY-MM-DD.html.

Usage:
    python predict.py              # today's date
    python predict.py --date 2025-03-15   # specific date (for back-testing)
"""

import os
import argparse
import warnings
from datetime import date, datetime

import requests
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams_static

from fetch_data import fetch_all_seasons, fetch_player_game_logs, GAMES_FILE
from features import build_prediction_row, FEATURE_COLS, KEY_PLAYER_MIN_THRESHOLD
from train import load_models

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ── Team ID → Name lookup ─────────────────────────────────────────────────────

def _build_team_lookup() -> dict:
    all_teams = nba_teams_static.get_teams()
    return {t["id"]: t["full_name"] for t in all_teams}

TEAM_LOOKUP = _build_team_lookup()

# Reverse lookup: full team name → team ID (used for injury matching)
NAME_TO_ID = {v: k for k, v in TEAM_LOOKUP.items()}


# ── Injury report ─────────────────────────────────────────────────────────────

ESPN_INJURY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)
# Statuses from ESPN that mean the player definitely won't play
OUT_STATUSES = {"Out", "Injured Reserve"}


def fetch_injury_report() -> dict:
    """
    Fetch today's NBA injury report from ESPN's public JSON API.

    Returns
    -------
    dict mapping team_full_name → list of player display names who are Out.
    Example: {"Miami Heat": ["Jimmy Butler", "Tyler Herro"], ...}

    On any network or parse error, returns an empty dict so predictions
    proceed normally with home_missing_mins = away_missing_mins = 0.
    """
    try:
        r = requests.get(ESPN_INJURY_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  ⚠ ESPN injury fetch failed: {exc}")
        print("  Continuing without injury data (missing mins will be 0).")
        return {}

    out_by_team: dict = {}
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        out_names = [
            inj["athlete"]["displayName"]
            for inj in team_entry.get("injuries", [])
            if inj.get("status") in OUT_STATUSES
            and inj.get("athlete", {}).get("displayName")
        ]
        if out_names:
            out_by_team[team_name] = out_names

    n_out = sum(len(v) for v in out_by_team.values())
    print(f"  ESPN injury report: {n_out} player(s) listed as Out across "
          f"{len(out_by_team)} team(s).")
    return out_by_team


def _missing_mins_for_team(
    team_name: str,
    team_id: int,
    out_by_team: dict,
    player_logs_df: pd.DataFrame,
) -> float:
    """
    Given the Out-player list for one team (from ESPN), look up each player's
    season-average minutes in player_logs_df and return the total.
    Only players averaging >= KEY_PLAYER_MIN_THRESHOLD minutes count.
    """
    out_names = out_by_team.get(team_name, [])
    if not out_names or player_logs_df is None or player_logs_df.empty:
        return 0.0

    team_logs = player_logs_df[player_logs_df["TEAM_ID"] == team_id]
    if team_logs.empty:
        return 0.0

    # Use the most recent season in the cache
    latest_season = team_logs["SEASON"].max()
    season_avgs = (
        team_logs[team_logs["SEASON"] == latest_season]
        .groupby("PLAYER_NAME")["MIN_float"]
        .mean()
        .reset_index()
    )

    total_missing = 0.0
    for out_name in out_names:
        # 1. Exact case-insensitive match
        match = season_avgs[
            season_avgs["PLAYER_NAME"].str.lower() == out_name.lower()
        ]
        if match.empty:
            # 2. Last-name fallback (handles "Jr.", suffix differences)
            last = out_name.split()[-1].lower()
            match = season_avgs[
                season_avgs["PLAYER_NAME"].str.lower().str.split().str[-1] == last
            ]
        if not match.empty:
            avg_min = float(match.iloc[0]["MIN_float"])
            if avg_min >= KEY_PLAYER_MIN_THRESHOLD:
                total_missing += avg_min
                print(f"    ↳ Out: {out_name} ({avg_min:.1f} avg min/game)")

    return total_missing


# ── Fetch today's schedule ────────────────────────────────────────────────────

def get_todays_games(game_date: date) -> pd.DataFrame:
    """
    Return a DataFrame of today's games with columns:
        GAME_ID, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_TEAM_NAME, AWAY_TEAM_NAME
    Only regular-season games are included (game_id starts with '002').
    Returns an empty DataFrame if no games today.
    """
    date_str = game_date.strftime("%m/%d/%Y")
    print(f"Fetching schedule for {date_str} …")
    
    # השתקת האזהרה הלא-רלוונטית של nba_api
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        board = scoreboardv2.ScoreboardV2(game_date=date_str, league_id="00")
        
    games_df = board.get_data_frames()[0]  # GameHeader

    if games_df.empty:
        print("  No games scheduled today.")
        return pd.DataFrame()

    # Filter regular season only (game_id prefix "002")
    games_df = games_df[games_df["GAME_ID"].str.startswith("002")].copy()

    if games_df.empty:
        print("  No regular-season games today.")
        return pd.DataFrame()

    result = pd.DataFrame({
        "GAME_ID": games_df["GAME_ID"].values,
        "HOME_TEAM_ID": games_df["HOME_TEAM_ID"].values,
        "AWAY_TEAM_ID": games_df["VISITOR_TEAM_ID"].values,
    })
    result["HOME_TEAM_NAME"] = result["HOME_TEAM_ID"].map(TEAM_LOOKUP).fillna("Unknown")
    result["AWAY_TEAM_NAME"] = result["AWAY_TEAM_ID"].map(TEAM_LOOKUP).fillna("Unknown")
    print(f"  Found {len(result)} regular-season game(s).")
    return result


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_games(games: pd.DataFrame, game_date: date) -> pd.DataFrame:
    """
    Given a DataFrame from get_todays_games(), run both models and return
    predictions with probability columns added.
    """
    lr, rf, scaler, margin_model = load_models()

    if not os.path.exists(GAMES_FILE):
        raise FileNotFoundError(
            "Historical game data not found. Run `python fetch_data.py` first."
        )
    historical = pd.read_csv(GAMES_FILE, parse_dates=["GAME_DATE"])

    # ── Injury report ─────────────────────────────────────────────────────────
    print("Fetching injury report …")
    out_by_team = fetch_injury_report()
    player_logs_df = fetch_player_game_logs()   # loads from cache; fast

    rows = []
    actual_winners = []
    actual_scores = []    # e.g. "112 – 108" (home – away)
    actual_margins = []   # home_pts - away_pts, or None if pending

    for _, game in games.iterrows():
        # --- מציאת המנצחת בפועל (אם המשחק כבר התקיים ונמצא בהיסטוריה) ---
        hist_game = historical[historical["GAME_ID"].astype(int) == int(game["GAME_ID"])]
        home_pts_actual = None
        away_pts_actual = None
        if not hist_game.empty:
            home_row = hist_game[hist_game["TEAM_ID"] == int(game["HOME_TEAM_ID"])]
            away_row = hist_game[hist_game["TEAM_ID"] == int(game["AWAY_TEAM_ID"])]
            if not home_row.empty and pd.notna(home_row.iloc[0]["WL"]):
                home_pts_actual = int(home_row.iloc[0]["PTS"])
                away_pts_actual = int(away_row.iloc[0]["PTS"]) if not away_row.empty else None
                if home_row.iloc[0]["WL"] == "W":
                    actual_winners.append(game["HOME_TEAM_NAME"])
                else:
                    actual_winners.append(game["AWAY_TEAM_NAME"])
            else:
                actual_winners.append("Pending")
        else:
            actual_winners.append("Pending")

        if home_pts_actual is not None and away_pts_actual is not None:
            actual_scores.append(f"{home_pts_actual} – {away_pts_actual}")
            actual_margins.append(home_pts_actual - away_pts_actual)
        else:
            actual_scores.append("Pending")
            actual_margins.append(None)
        # -----------------------------------------------------------------

        # Compute missing minutes for each team using the injury report
        home_name = game["HOME_TEAM_NAME"]
        away_name = game["AWAY_TEAM_NAME"]
        home_mm = _missing_mins_for_team(
            home_name, int(game["HOME_TEAM_ID"]), out_by_team, player_logs_df
        )
        away_mm = _missing_mins_for_team(
            away_name, int(game["AWAY_TEAM_ID"]), out_by_team, player_logs_df
        )

        feat_row = build_prediction_row(
            home_team_id=int(game["HOME_TEAM_ID"]),
            away_team_id=int(game["AWAY_TEAM_ID"]),
            game_date=game_date,
            historical_df=historical,
            home_missing_mins=home_mm,
            away_missing_mins=away_mm,
        )
        rows.append(feat_row)

    X_pred = pd.concat(rows, ignore_index=True)[FEATURE_COLS].astype(float)
    X_scaled = scaler.transform(X_pred)

    lr_probs = lr.predict_proba(X_scaled)[:, 1]   # P(home wins)
    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    ensemble_probs = (lr_probs + rf_probs) / 2.0

    # Point margin prediction (positive = home team favoured by that many points)
    raw_margins = margin_model.predict(X_scaled)

    games = games.copy()
    games["lr_home_prob"] = np.round(lr_probs * 100, 1)
    games["rf_home_prob"] = np.round(rf_probs * 100, 1)
    games["ensemble_home_prob"] = np.round(ensemble_probs * 100, 1)
    games["predicted_winner"] = np.where(
        ensemble_probs >= 0.5,
        games["HOME_TEAM_NAME"],
        games["AWAY_TEAM_NAME"],
    )
    games["confidence"] = np.where(
        ensemble_probs >= 0.5,
        np.round(ensemble_probs * 100, 1),
        np.round((1 - ensemble_probs) * 100, 1),
    )
    # Spread label: e.g. "Lakers by 4.5" or "Toss-up"
    spread_labels = []
    for i, margin in enumerate(raw_margins):
        abs_margin = abs(margin)
        if abs_margin < 1.5:
            spread_labels.append("Pick'em")
        elif margin > 0:
            spread_labels.append(f"{games.iloc[i]['HOME_TEAM_NAME']} by {abs_margin:.1f}")
        else:
            spread_labels.append(f"{games.iloc[i]['AWAY_TEAM_NAME']} by {abs_margin:.1f}")
    games["projected_spread"] = spread_labels
    games["raw_margin"] = np.round(raw_margins, 1)
    games["actual_winner"] = actual_winners
    games["actual_score"] = actual_scores
    games["actual_margin"] = actual_margins
    return games


# ── HTML Dashboard ────────────────────────────────────────────────────────────

def _prob_bar(home_prob: float) -> str:
    """Return an inline HTML probability bar."""
    away_prob = 100 - home_prob
    return f"""
        <div class="prob-bar">
            <div class="prob-home" style="width:{home_prob}%">{home_prob:.0f}%</div>
            <div class="prob-away" style="width:{away_prob}%">{away_prob:.0f}%</div>
        </div>"""


def _confidence_badge(conf: float) -> str:
    if conf >= 70:
        cls = "badge-high"
        label = "High"
    elif conf >= 60:
        cls = "badge-med"
        label = "Medium"
    else:
        cls = "badge-low"
        label = "Toss-up"
    return f'<span class="badge {cls}">{label} ({conf:.0f}%)</span>'


def build_html(predictions: pd.DataFrame, game_date: date) -> str:
    """Render the full HTML dashboard string."""
    # Cross-platform safe date formatting (avoids Linux-only %-d)
    date_str = f"{game_date.strftime('%A, %B')} {game_date.day}, {game_date.year}"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards_html = ""
    for _, row in predictions.iterrows():
        home = row["HOME_TEAM_NAME"]
        away = row["AWAY_TEAM_NAME"]
        winner = row["predicted_winner"]
        actual = row["actual_winner"]
        actual_score = row["actual_score"]
        actual_margin = row["actual_margin"]
        raw_margin = row["raw_margin"]
        home_is_fav = row["ensemble_home_prob"] >= 50

        lr_h = row["lr_home_prob"]
        rf_h = row["rf_home_prob"]
        ens_h = row["ensemble_home_prob"]
        conf = row["confidence"]
        spread = row["projected_spread"]

        # --- actual result + score + spread accuracy ---
        result_html = ""
        spread_result_html = ""
        if actual != "Pending" and actual_margin is not None:
            is_correct = (winner == actual)
            result_html = (
                f'<div class="actual-result correct">✅ <strong>Correct!</strong> '
                f'{home} {actual_score} {away}</div>'
                if is_correct else
                f'<div class="actual-result incorrect">❌ <strong>Incorrect.</strong> '
                f'{home} {actual_score} {away}</div>'
            )
            # Spread accuracy: compare predicted margin direction & magnitude
            pred = float(raw_margin)
            act = float(actual_margin)
            # Determine favoured team name and margins from their perspective
            if pred >= 0:
                fav_name = home
                pred_margin = abs(pred)
                act_margin_fav = act          # positive = home won by that much
            else:
                fav_name = away
                pred_margin = abs(pred)
                act_margin_fav = -act         # positive = away won by that much

            if act_margin_fav > pred_margin:
                spread_result_html = (
                    f'<div class="spread-result spread-more">'
                    f'📈 {fav_name} won by <strong>{abs(act_margin_fav):.0f}</strong> '
                    f'— more than predicted ({pred_margin:.1f})</div>'
                )
            elif act_margin_fav >= 0:
                spread_result_html = (
                    f'<div class="spread-result spread-less">'
                    f'📉 {fav_name} won by <strong>{abs(act_margin_fav):.0f}</strong> '
                    f'— less than predicted ({pred_margin:.1f})</div>'
                )
            else:
                spread_result_html = (
                    f'<div class="spread-result spread-upset">'
                    f'🔄 Upset — {actual} won by <strong>{abs(act_margin_fav):.0f}</strong> '
                    f'(predicted {fav_name} by {pred_margin:.1f})</div>'
                )
        # -----------------------------------------------

        cards_html += f"""
        <div class="card">
            <div class="matchup">
                <div class="team {'winner' if home_is_fav else ''}">
                    <div class="team-name">{home}</div>
                    <div class="team-label">HOME</div>
                </div>
                <div class="vs">VS</div>
                <div class="team {'winner' if not home_is_fav else ''}">
                    <div class="team-name">{away}</div>
                    <div class="team-label">AWAY</div>
                </div>
            </div>

            <div class="prediction-row">
                <span class="pick-label">Predicted winner:</span>
                <span class="pick-team">{winner}</span>
                {_confidence_badge(conf)}
            </div>
            <div class="spread-row">
                <span class="pick-label">Projected spread:</span>
                <span class="spread-value">{spread}</span>
            </div>

            {result_html}
            {spread_result_html}

            <div class="models-section">
                <div class="model-row">
                    <span class="model-name">Logistic Regression</span>
                    {_prob_bar(lr_h)}
                </div>
                <div class="model-row">
                    <span class="model-name">Random Forest</span>
                    {_prob_bar(rf_h)}
                </div>
                <div class="model-row ensemble-row">
                    <span class="model-name">⭐ Ensemble</span>
                    {_prob_bar(ens_h)}
                </div>
            </div>
        </div>"""

    if not cards_html:
        cards_html = '<p class="no-games">No regular-season games scheduled for this date.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Predictions — {date_str}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
    padding: 24px;
  }}

  header {{
    text-align: center;
    margin-bottom: 32px;
  }}
  header h1 {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c8102e, #1d428a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  header .subtitle {{
    margin-top: 6px;
    color: #8b949e;
    font-size: 0.9rem;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
    gap: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }}

  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    transition: transform 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); }}

  .matchup {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
  }}
  .team {{
    flex: 1;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    border: 2px solid transparent;
    opacity: 0.65;
    transition: opacity 0.2s;
  }}
  .team.winner {{
    opacity: 1;
    border-color: #238636;
    background: rgba(35, 134, 54, 0.08);
  }}
  .team-name {{ font-size: 1.05rem; font-weight: 700; line-height: 1.2; }}
  .team-label {{ font-size: 0.7rem; color: #8b949e; margin-top: 4px; letter-spacing: 0.08em; text-transform: uppercase; }}
  .vs {{ font-size: 1rem; font-weight: 700; color: #8b949e; padding: 0 12px; }}

  .prediction-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .pick-label {{ color: #8b949e; font-size: 0.85rem; }}
  .pick-team {{ font-weight: 700; color: #58a6ff; font-size: 1rem; }}

  .spread-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .spread-value {{ font-weight: 600; color: #d29922; font-size: 0.95rem; }}

  .spread-result {{
    margin-top: 4px;
    margin-bottom: 12px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .spread-more  {{ background: rgba(35,134,54,0.1);   border: 1px solid #238636; color: #e6edf3; }}
  .spread-less  {{ background: rgba(210,153,34,0.12); border: 1px solid #9e6a03; color: #e6edf3; }}
  .spread-upset {{ background: rgba(139,148,158,0.1); border: 1px solid #484f58; color: #e6edf3; }}

  .actual-result {{
    margin-top: 4px;
    margin-bottom: 16px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    text-align: left;
  }}
  .actual-result.correct {{
    background: rgba(35,134,54,0.1);
    border: 1px solid #238636;
    color: #e6edf3;
  }}
  .actual-result.incorrect {{
    background: rgba(200,16,46,0.1);
    border: 1px solid #c8102e;
    color: #e6edf3;
  }}

  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-high   {{ background: rgba(35,134,54,0.2);  color: #3fb950; border: 1px solid #238636; }}
  .badge-med    {{ background: rgba(210,153,34,0.2); color: #d29922; border: 1px solid #9e6a03; }}
  .badge-low    {{ background: rgba(139,148,158,0.15); color: #8b949e; border: 1px solid #484f58; }}

  .models-section {{ display: flex; flex-direction: column; gap: 8px; }}
  .model-row {{ display: flex; align-items: center; gap: 10px; }}
  .model-name {{ font-size: 0.78rem; color: #8b949e; width: 140px; flex-shrink: 0; }}
  .ensemble-row .model-name {{ color: #e6edf3; font-weight: 600; }}

  .prob-bar {{
    flex: 1;
    display: flex;
    border-radius: 4px;
    overflow: hidden;
    height: 22px;
    font-size: 0.72rem;
    font-weight: 600;
  }}
  .prob-home {{
    background: #1d428a;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 5px;
    min-width: 28px;
  }}
  .prob-away {{
    background: #c8102e;
    color: #fff;
    display: flex;
    align-items: center;
    padding-left: 5px;
    min-width: 28px;
  }}

  .no-games {{
    text-align: center;
    color: #8b949e;
    font-size: 1.1rem;
    padding: 60px;
  }}

  footer {{
    text-align: center;
    margin-top: 36px;
    color: #484f58;
    font-size: 0.78rem;
  }}
</style>
</head>
<body>
<header>
  <h1>🏀 NBA Game Predictions</h1>
  <div class="subtitle">{date_str} &nbsp;·&nbsp; Generated {generated_at} &nbsp;·&nbsp; Blue = Home &nbsp;|&nbsp; Red = Away</div>
</header>

<div class="grid">
{cards_html}
</div>

<footer>
  Predictions from Logistic Regression + Random Forest ensemble trained on 6 seasons of NBA data.<br>
  Regular season games only. For informational purposes only.
</footer>
</body>
</html>"""
    return html


def run_predictions(game_date: date | None = None) -> str | None:
    """
    Full pipeline: fetch schedule → build features → predict → write HTML.
    Returns the path to the generated HTML file, or None if no games.
    """
    if game_date is None:
        game_date = date.today()

    games = get_todays_games(game_date)
    if games.empty:
        print("No predictions to make.")
        return None

    predictions = predict_games(games, game_date)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = build_html(predictions, game_date)
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{game_date.isoformat()}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Dashboard saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict today's NBA games")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date else date.today()
    )
    run_predictions(target_date)