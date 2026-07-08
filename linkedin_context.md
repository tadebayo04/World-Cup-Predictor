# Context Document — World Cup Predictor Project
# (For writing a LinkedIn post)

## What to write

A LinkedIn post that's professional but conversational. It should showcase the end-to-end ML project I built to predict 2026 FIFA World Cup match outcomes. Highlight the data engineering, feature development, and model results. The tone should reflect someone who is passionate about football and data science. Keep it engaging — not a dry technical report. Use emojis sparingly but effectively. Aim for around 300–400 words.

---

## Full Project Context

### What the project does
Built a machine learning pipeline from scratch that predicts the outcome of every 2026 FIFA World Cup group-stage match (72 fixtures) as: **Home Win**, **Away Win**, or **Draw**. Two models were trained and compared: **Random Forest** and **Logistic Regression**.

---

### Data Sources (7 raw datasets)

| Dataset | Rows | What it covers |
|---|---|---|
| `results.csv` | 49,412 | Every international football match ever played, 1872–2026 |
| `fifa_rankings.csv` | 70,195 | FIFA world rankings snapshots, 1992–2026 |
| `players.csv` | 47,702 | Player profiles: position, caps, market value, citizenship |
| `player_valuations.csv` | 507,816 | Time-series of each player's market value in EUR, 2000–2026 |
| `national_teams.csv` | 119 | National team metadata |
| `games.csv` | 88,808 | Club-level international game data |
| `game_lineups.csv` | 3,172,455 | Match-by-match player lineup data |

---

### Data Cleaning Highlights

- Standardised **team name inconsistencies** across sources (e.g. "South Korea" → "Korea Republic", "Turkey" → "Türkiye", "USA" vs "United States", "DR Congo" vs "Congo DR" — 13+ mappings in total)
- Separated 49,339 **historical completed matches** from 72 **upcoming 2026 WC fixtures**
- Filtered to post-1992 data only (FIFA ranking data availability)
- Resolved country name mismatches between Transfermarkt player data and FIFA results data to correctly merge squad valuations
- Fixed a `"--"` garbage date row in the processed dataset
- Imputed pre-2004 squad values (before Transfermarkt coverage began) with the global median to avoid false zero signals

---

### Features Engineered (18 model features)

**FIFA Ranking Features** — merged using `merge_asof` so each match gets the ranking snapshot closest to (but before) match date:
- `home_rank`, `away_rank` — each team's FIFA ranking at time of match
- `home_points`, `away_points` — FIFA rating points
- `points_diff` — points difference (home minus away)

**Recent Form Features** — rolling averages over last 5 and 10 matches (Win=3, Draw=1, Loss=0), computed across both home and away appearances:
- `home_form_5`, `home_form_10`
- `away_form_5`, `away_form_10`

**Goal Difference Features** — rolling average goal difference (goals scored minus conceded):
- `home_goal_diff_5`, `away_goal_diff_5`

**Head-to-Head History** — cumulative historical record between each team pair, computed with `shift(1)` to avoid data leakage:
- `h2h_home_win_rate` — fraction of past meetings won by the home team
- `h2h_total_games` — total past meetings (weights how meaningful the rate is)

**Squad Valuation Features** — total EUR market value of each squad, merged as a time-series using `merge_asof`:
- `home_squad_value_log`, `away_squad_value_log` — log-scaled (right-skewed distribution, max ~€3.4B vs median ~€1.5M)
- `squad_value_diff` — absolute difference in squad wealth

**Match Context Features:**
- `neutral` — whether match is at a neutral venue (binary). For 2026 WC: USA, Canada, and Mexico have home advantage for their own fixtures; all other 63 matches are neutral
- `tournament_weight` — importance tier: Friendly=1, Qualifier=2, Major Tournament=3, World Cup=4

---

### Modelling Approach

**Train/test split:** time-based (not random) — trained on 1993–2021 (~23,000 matches), tested on 2022–2026 (~4,000 matches including the 2022 Qatar World Cup). Random shuffling was deliberately avoided because form and H2H features encode temporal state.

**Cross-validation:** `TimeSeriesSplit(n_splits=5)` inside `GridSearchCV` to prevent future leakage during hyperparameter tuning.

**Preprocessing:** `StandardScaler` fit on training data only and applied to test/future data (for Logistic Regression). Squad value features log-transformed before scaling.

**Class imbalance:** `class_weight='balanced'` used in both models — the dataset is ~47% Home Win, ~29% Away Win, ~24% Draw.

---

### Model Results

**Naive baseline accuracy** (always predict Home Win): ~47%

| Model | Accuracy | F1 Macro | F1 Home Win | F1 Away Win | F1 Draw |
|---|---|---|---|---|---|
| **Random Forest** | **68.8%** | **0.66** | 0.78 | 0.72 | 0.49 |
| Logistic Regression | 66.7% | 0.63 | 0.77 | 0.71 | 0.41 |

Random Forest best params: 300 trees, no max depth, min_samples_leaf=5
Logistic Regression best C: 0.1

Draw is the hardest class to predict (F1=0.49) — as expected in football.

---

### Predictions for 2026 World Cup

All 72 group-stage fixtures predicted with win/draw probabilities from both models. Notable findings:
- Both models agree on 57 of 72 fixtures
- USA are predicted to lose all 3 of their home group games despite the home advantage
- The models correctly give USA, Canada, and Mexico a home advantage boost (9 fixtures marked non-neutral)
- 15 fixtures have model disagreement, concentrated on the tightest matchups (e.g. England vs Croatia, Argentina vs Austria, Norway vs France)

---

### Tech Stack
Python · Pandas · NumPy · Scikit-learn · Jupyter Notebooks · Git/GitHub

---

## Group Stage Evaluation (Post-Tournament)

The 2026 FIFA World Cup group stage ran June 11–27, 2026. After the 72 matches concluded, the pre-tournament predictions were evaluated against actual results.

### Actual group stage outcome distribution
| Outcome | Count |
|---|---|
| Home Win | 33 (46%) |
| Draw | 20 (28%) |
| Away Win | 19 (26%) |

### Group Stage Accuracy

| Model | Accuracy | F1-macro | F1 Home Win | F1 Away Win | F1 Draw |
|---|---|---|---|---|---|
| **Random Forest** | **43.1%** (31/72) | 0.420 | 0.500 | 0.400 | 0.359 |
| Logistic Regression | 45.8% (33/72) | 0.455 | 0.475 | 0.468 | 0.421 |

**Context on why accuracy dropped vs the 68.8% test-set baseline:**
- The group stage had a higher draw rate (28%) and more upsets than the historical mix used for training
- 48-team format brought in more mismatched teams and more defensive, cautious football
- Draws are the hardest class to predict — lowest F1 in both training evaluation and group stage

### Model Agreement
- Models agreed on 59/72 fixtures
- When both agreed and were correct: 27 matches
- Both wrong on 35 of 72 matches

### Notable Upsets the Models Missed (RF ≥60% confident, wrong)
The models were highly confident on 16 matches that went the other way:
- **Congo DR 3–1 Uzbekistan** — model predicted Uzbekistan win (83% confidence)
- **Ecuador 2–1 Germany** — model predicted Germany win (77%)
- **USA 2–0 Australia** — model predicted Australia win (71%)
- **Sweden 5–1 Tunisia** — model predicted Tunisia win (70%)
- **England 0–0 Ghana** — model predicted England win (76%)

These upsets highlight where FIFA ranking and squad value don't capture on-the-day factors: injuries, tactical setups, tournament momentum, and group-stage motivation dynamics.

---

## Knockout Round Predictions (Round of 32) — v2 Models

Both models were retrained on the full dataset including the 72 group stage results (v2 models), then used to predict the 13 confirmed Round of 32 fixtures.

**v2 Model Test Accuracy (2022–2026 holdout):** RF 56.1%, LR 55.1%

### Round of 32 Predictions vs Actuals

| Date | Home | Away | Score | Actual | RF | LR |
|---|---|---|---|---|---|---|
| 30 Jun | Côte d'Ivoire | Norway | 1–2 | Away Win | ✓ Away Win | ✓ Away Win |
| 30 Jun | France | Sweden | 3–0 | Home Win | ✓ Home Win | ✓ Home Win |
| 30 Jun | Mexico | Ecuador | 2–0 | Home Win | ✗ Draw | ✓ Home Win |
| 1 Jul | England | Congo DR | 2–1 | Home Win | ✓ Home Win | ✓ Home Win |
| 1 Jul | Belgium | Senegal | 3–2 | Home Win | ✗ Away Win | ✗ Away Win |
| 1 Jul | USA | Bosnia and Herzegovina | 2–0 | Home Win | ✓ Home Win | ✓ Home Win |
| 2 Jul | Spain | Austria | 3–0 | Home Win | ✓ Home Win | ✓ Home Win |
| 2 Jul | Portugal | Croatia | 2–1 | Home Win | ✓ Home Win | ✓ Home Win |
| 2 Jul | Switzerland | Algeria | 2–0 | Home Win | ✗ Draw | ✗ Draw |
| 3 Jul | Australia | Egypt | 1–1 | Draw | ✓ Draw | ✗ Away Win |
| 3 Jul | Argentina | Cabo Verde | 3–2 | Home Win | ✓ Home Win | ✓ Home Win |
| 3 Jul | Colombia | Ghana | 1–0 | Home Win | ✓ Home Win | ✓ Home Win |
| 4 Jul | Canada | Morocco | 0–3 | Away Win | ✗ Draw | ✓ Away Win |

**R32 Accuracy: RF 69.2% (9/13) · LR 76.9% (10/13)**

Notable: Belgium beat Senegal 3–2 despite both models backing Senegal — the biggest upset of the round. Morocco also thrashed Canada 3–0 where RF only predicted a draw.

---

## Round of 16 Results

The Round of 16 played out July 4–7. These results were used to retrain the v3 models.

| Date | Home | Away | Score | Result |
|---|---|---|---|---|
| 4 Jul | Paraguay | France | 0–1 | France advances |
| 5 Jul | Brazil | Norway | 1–2 | Norway advances (upset) |
| 5 Jul | Mexico | England | 2–3 | England advances |
| 6 Jul | Portugal | Spain | 0–1 | Spain advances |
| 6 Jul | USA | Belgium | 1–4 | Belgium advances (dominant) |
| 7 Jul | Argentina | Egypt | 3–2 | Argentina advances |
| 7 Jul | Switzerland | Colombia | 0–0 | TBD via extra time/penalties |

**Key storylines:**
- Norway continued their surprise run, beating Brazil 2–1
- Belgium dismantled USA 4–1, making a strong statement as QF contenders
- Spain edged Portugal 1–0 in a tight Iberian derby

---

## Quarterfinal Predictions — v3 Models

Models retrained on all 96 scored 2026 WC matches (group stage + R32 + R16).

**v3 Model Test Accuracy (2022–2026 holdout):** RF 56.0%, LR 55.3%

All four QF matches: **both models agree on every fixture.**

| Date | Home | Away | RF | LR | RF Probs (HW / D / AW) |
|---|---|---|---|---|---|
| 9 Jul | France | Morocco | Home Win | Home Win | 61.9% / 22.7% / 15.4% |
| 10 Jul | Spain | Belgium | Home Win | Home Win | 43.4% / 33.5% / 23.2% |
| 11 Jul | Norway | England | Away Win | Away Win | 14.9% / 31.7% / 53.4% |
| 11 Jul | Argentina | Switzerland | Home Win | Home Win | 59.5% / 20.7% / 19.8% |

**Predicted QF winners:** France, Spain, England, Argentina

---

## Semifinal Predictions

Using predicted QF winners to construct the bracket:

| Date | Home | Away | Prediction | RF Probs (HW / D / AW) |
|---|---|---|---|---|
| 14 Jul | France | Spain | **Spain** | 15.6% / 43.6% / 40.9% |
| 15 Jul | England | Argentina | **Argentina** | 18.9% / 27.2% / 54.0% |

Both models agree on both semifinals. Spain predicted to edge France (as slight away favourites), Argentina predicted to beat England.

---

## Final Prediction — 2026 World Cup

| Date | Home | Away | RF | LR | RF Probs (HW / D / AW) |
|---|---|---|---|---|---|
| 19 Jul | Spain | Argentina | **Argentina** | **Argentina** | 33.4% / 29.8% / 36.7% |

**🏆 Predicted 2026 World Cup Winner: Argentina**

The final is the tightest call in the entire bracket — RF gives Argentina a 36.7% win probability vs Spain's 33.4%, only 3.3 percentage points separating them. LR is more decisive at 39.7% vs 26.7% for Argentina. Both models agree Argentina lifts the trophy.

Argentina's edge comes from: superior squad market value, better recent form (surviving a scare vs Cabo Verde 3–2 and Egypt 3–2), and a strong historical H2H record against European opponents in World Cup knockouts.

---

## What's Next
- Retrain again with QF results as they come in for updated SF/Final predictions
- Tournament bracket simulation (probability-weighted path to champion)
- Adding lineup-based features from the 3.2M row game_lineups dataset
