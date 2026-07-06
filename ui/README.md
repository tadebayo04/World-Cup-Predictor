# World Cup 2026 Prediction UI

A static, dependency-free front end for browsing the model's match predictions.

## Viewing it

Open `ui/index.html` directly in a browser — no server or build step needed.
(Or serve the folder: `python -m http.server -d ui` and visit http://localhost:8000.)

## What it shows

- **All 85 predicted fixtures** (72 group stage + 13 round of 32), grouped by date,
  filterable by stage or team, in card or table view.
- **The predicted result** per match — the average of the Random Forest and
  Logistic Regression win/draw/loss probabilities, with a win-probability bar and a
  per-model breakdown on hover. Cards flag whether the two models agree or split.
- **Why that team is favored** — an expandable panel per match with plain-English
  reasons plus the underlying feature comparison: FIFA ranking and ranking points,
  points-per-game form and goal difference over the last 5 matches, squad market
  value, head-to-head record, and home advantage.

## Regenerating the data

`ui/data.js` is generated from the prediction CSVs and the feature pipeline:

```bash
python src/export_ui_data.py
```

Re-run it after retraining models or refreshing predictions
(`python src/train.py`). Note: displayed squad market values are recomputed for
readability (sum of each country's top-26 players' latest valuations); the model
itself consumes the log-scaled snapshot features from the training pipeline.
