# 🏀 NBA Game Winner Probability Predictor

Predicts the winner of every NBA regular-season game using a **Logistic Regression + Random Forest ensemble** trained on 6 seasons of data (2019-20 → 2024-25), with recent seasons weighted more heavily.

---

## Features used

| Feature | Description |
|---|---|
| `home_advantage` | 1 = home team (always 1 at prediction time) |
| `month` | Calendar month of the game (captures fatigue/pace across the season) |
| `home_win_pct` | Home team's season win% **before** this game |
| `away_win_pct` | Away team's season win% before this game |
| `win_pct_diff` | home − away win% |
| `home_form5` | Home team's win rate in their last 5 games |
| `away_form5` | Away team's win rate in their last 5 games |
| `form5_diff` | home − away form5 |
| `home_h2h` | Home team's historical win rate vs this specific opponent |
| `away_h2h` | Away team's historical win rate vs this specific opponent |

**Recency weights**: season 2019-20 → weight 1 … 2024-25 → weight 6.

---

## Project structure

```
nba_predictor/
├── fetch_data.py     # pulls data from nba_api, caches to data/games_raw.csv
├── features.py       # feature engineering pipeline
├── train.py          # trains LR + RF, saves models to models/
├── predict.py        # fetches today's schedule, predicts, writes HTML
├── daily_run.py      # ← RUN THIS daily (cron entry point)
├── requirements.txt
├── data/             # auto-created — cached game CSV
├── models/           # auto-created — saved .joblib model files
├── outputs/          # auto-created — HTML dashboards
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

This pulls ~6 seasons of game logs from the NBA API (~1–2 minutes due to rate limiting):

```bash
python fetch_data.py          # downloads and caches data/games_raw.csv
python train.py               # trains both models, prints CV accuracy
```

### 4. Run today's predictions

```bash
python predict.py
# → opens outputs/predictions_YYYY-MM-DD.html
```

---

## Daily automation (cron)

Add this line to your crontab (`crontab -e`) to run every morning at 9 AM:

```
0 9 * * * /full/path/to/venv/bin/python /full/path/to/nba_predictor/daily_run.py
```

The script automatically:
- Skips runs during the off-season (May–September)
- Only retrains if new game results have been added
- Writes a fresh `outputs/predictions_YYYY-MM-DD.html` each day

**Windows Task Scheduler** equivalent:
- Program: `C:\path\to\venv\Scripts\python.exe`
- Arguments: `C:\path\to\nba_predictor\daily_run.py`
- Trigger: Daily at 9:00 AM

---

## Manual usage

```bash
# Force a full re-fetch of all 6 seasons
python fetch_data.py --refresh

# Force retrain even if no new games
python daily_run.py --force-train

# Run for a specific date (useful for back-testing)
python predict.py --date 2025-01-20
python daily_run.py --date 2025-01-20
```

---

## Output — HTML Dashboard

Each run produces `outputs/predictions_YYYY-MM-DD.html`.  Open it in any browser.

For each game you'll see:
- **Predicted winner** with a confidence badge (High / Medium / Toss-up)
- Side-by-side probability bars for Logistic Regression, Random Forest, and the Ensemble

---

## Notes

- All predictions are for **regular season games only** (game IDs starting with `002`).
- The NBA API has rate limiting — `fetch_data.py` adds a short sleep between calls.
- Model accuracy is typically in the **60–65% range**, consistent with published NBA prediction literature.
- This is a statistical model for informational/research purposes only.
