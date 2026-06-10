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

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams_static

from fetch_data import fetch_all_seasons, GAMES_FILE
from features import build_prediction_row, FEATURE_COLS
from train import load_models

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ── Team ID → Name lookup ─────────────────────────────────────────────────────

def _build_team_lookup() -> dict:
    all_teams = nba_teams_static.get_teams()
    return {t["id"]: t["full_name"] for t in all_teams}

TEAM_LOOKUP = _build_team_lookup()


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
    lr, rf, scaler = load_models()

    if not os.path.exists(GAMES_FILE):
        raise FileNotFoundError(
            "Historical game data not found. Run `python fetch_data.py` first."
        )
    historical = pd.read_csv(GAMES_FILE, parse_dates=["GAME_DATE"])

    rows = []
    actual_winners = []

    for _, game in games.iterrows():
        # --- מציאת המנצחת בפועל (אם המשחק כבר התקיים ונמצא בהיסטוריה) ---
        hist_game = historical[historical["GAME_ID"].astype(int) == int(game["GAME_ID"])]
        if not hist_game.empty:
            home_row = hist_game[hist_game["TEAM_ID"] == int(game["HOME_TEAM_ID"])]
            if not home_row.empty and pd.notna(home_row.iloc[0]["WL"]):
                if home_row.iloc[0]["WL"] == "W":
                    actual_winners.append(game["HOME_TEAM_NAME"])
                else:
                    actual_winners.append(game["AWAY_TEAM_NAME"])
            else:
                actual_winners.append("Pending")
        else:
            actual_winners.append("Pending")
        # -----------------------------------------------------------------

        feat_row = build_prediction_row(
            home_team_id=int(game["HOME_TEAM_ID"]),
            away_team_id=int(game["AWAY_TEAM_ID"]),
            game_date=game_date,
            historical_df=historical,
        )
        rows.append(feat_row)

    X_pred = pd.concat(rows, ignore_index=True)[FEATURE_COLS].astype(float)
    X_scaled = scaler.transform(X_pred)

    lr_probs = lr.predict_proba(X_scaled)[:, 1]   # P(home wins)
    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    ensemble_probs = (lr_probs + rf_probs) / 2.0

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
    games["actual_winner"] = actual_winners
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
    date_str = game_date.strftime("%A, %B %-d %Y")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards_html = ""
    for _, row in predictions.iterrows():
        home = row["HOME_TEAM_NAME"]
        away = row["AWAY_TEAM_NAME"]
        winner = row["predicted_winner"]
        actual = row["actual_winner"]
        home_is_fav = row["ensemble_home_prob"] >= 50

        lr_h = row["lr_home_prob"]
        rf_h = row["rf_home_prob"]
        ens_h = row["ensemble_home_prob"]
        conf = row["confidence"]

        # --- יצירת תווית "התוצאה בפועל" ---
        result_html = ""
        if actual != "Pending":
            is_correct = (winner == actual)
            if is_correct:
                result_html = f'<div class="actual-result correct">✅ <strong>Correct Prediction!</strong> Actual Winner: {actual}</div>'
            else:
                result_html = f'<div class="actual-result incorrect">❌ <strong>Incorrect.</strong> Actual Winner: {actual}</div>'
        # ----------------------------------

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
            
            {result_html}

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