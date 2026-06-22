# 🏀 NBA Game Predictor

Predicts the winner and point spread of every NBA regular-season game using a Logistic Regression + Random Forest ensemble trained on 6 seasons of data, with recent seasons weighted exponentially more heavily.

---

## Features used (13 total)

| Feature | Description |
|---|---|
| `month` | Calendar month of the game (captures fatigue/pace across the season) |
| `home_win_pct_at_home` | Home team's season win% at home before this game (Bayesian-smoothed) |
| `away_win_pct_on_road` | Away team's season win% on the road before this game (Bayesian-smoothed) |
| `win_pct_diff` | `home_win_pct_at_home` − `away_win_pct_on_road` |
| `home_form5` | Home team's win rate in their last 5 games |
| `away_form5` | Away team's win rate in their last 5 games |
| `form5_diff` | `home_form5` − `away_form5` |
| `home_h2h` | Home team's historical win rate vs this specific opponent |
| `away_h2h` | Away team's historical win rate vs this specific opponent (= 1 − `home_h2h`) |
| `home_rest_days` | Home team's rest days before this game (0 for back-to-back, max 7) |
| `away_rest_days` | Away team's rest days before this game (0 for back-to-back, max 7) |
| `home_missing_mins` | Total avg min/game of injured-out key players for the home team |
| `away_missing_mins` | Total avg min/game of injured-out key players for the away team |

**Bayesian smoothing**: Every team starts the season with a +4W / +4L prior so early-season win% doesn't spike to 0% or 100% after 1–2 games.

**Recency weights**: `[1, 3, 9, 27, 81, 243]` — the current season has 243× the influence of the oldest season. Each season back is one third as influential as the one before it.

**Injury feature**: A player counts as a "key" player if they averaged ≥ 15 min/game over ≥ 30 games for their team that season. Their absence is only tracked from the game they first joined the team (prevents mid-season trade arrivals from generating false "missing" signals).

---

## Project structure

```
nba_predictor/
├── fetch_data.py     # pulls team + player game logs from nba_api, caches to data/
├── features.py       # feature engineering pipeline (13 features + margin)
├── train.py          # trains LR + RF (win/loss) + Ridge (point margin), saves to models/
├── predict.py        # fetches today's schedule, predicts, writes HTML dashboard
├── daily_run.py      # ← RUN THIS daily (cron entry point)
├── requirements.txt
├── data/             # auto-created — games_raw.csv + player_game_logs.csv
├── models/           # auto-created — lr_model, rf_model, scaler, margin_model (.joblib)
├── outputs/          # auto-created — HTML dashboards per date
└── logs/             # auto-created — daily.log
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. First-time data fetch + model training

Pulls ~6 seasons of team and player game logs from the NBA API (~3–5 minutes due to rate limiting):

```bash
python fetch_data.py          # downloads data/games_raw.csv
python train.py               # also fetches player_game_logs.csv, then trains all models
```

`train.py` prints cross-validated accuracy for LR and RF, plus the point margin RMSE.

### 4. Run today's predictions

```bash
python predict.py
# → writes outputs/predictions_YYYY-MM-DD.html
```

---

## Daily automation (cron)

Add this line to your crontab (`crontab -e`) to run every morning at 9 AM:

```
0 9 * * * /full/path/to/venv/bin/python /full/path/to/nba_predictor/daily_run.py
```

The script automatically:
- Skips runs during the off-season (May–September)
- Only retrains if new game results have been added since the last run
- Fetches today's injury report from ESPN and factors it into predictions
- Writes a fresh `outputs/predictions_YYYY-MM-DD.html` each day

**Windows Task Scheduler** equivalent:
- Program: `C:\path\to\venv\Scripts\python.exe`
- Arguments: `C:\path\to\nba_predictor\daily_run.py`
- Trigger: Daily at 9:00 AM

---

## Manual usage

```bash
# Force a full re-fetch of all 6 seasons of game data
python fetch_data.py --refresh

# Force retrain even if no new games were added
python daily_run.py --force-train

# Back-test on a specific past date
python predict.py --date 2026-04-12
python daily_run.py --date 2026-04-12
```

---

## Output — HTML Dashboard

Each run produces `outputs/predictions_YYYY-MM-DD.html`. Open it in any browser.

For each game you'll see:

- **Predicted winner** with a confidence badge (High / Medium / Toss-up)
- **Projected spread** — e.g. "Lakers by 4.5" or "Pick'em" (from the Ridge regression model)
- Side-by-side probability bars for Logistic Regression, Random Forest, and the Ensemble
- **Actual result** when back-testing past dates: score, win/loss correctness, and whether the spread was beaten (📈 more / 📉 less / 🔄 upset)

---

## Notes

- All predictions are for **regular season games only** (game IDs starting with `002`).
- The NBA API has rate limiting — `fetch_data.py` adds a 1.5-second sleep between API calls.
- Injury data is sourced from ESPN's public JSON API at prediction time and is **not** stored or re-used between runs.
- Model accuracy is typically in the **60–65% range**, consistent with published NBA prediction literature.
- This is a statistical model for informational/research purposes only.
