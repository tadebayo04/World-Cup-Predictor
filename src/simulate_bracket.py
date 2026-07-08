"""
Simulate the 2026 WC knockout bracket from the quarterfinals to the final.

Usage:
    python src/simulate_bracket.py

Requires v3 models and a current future_fixtures_features.csv (QFs only).
Outputs data/processed/bracket_predictions.csv.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    FEATURE_COLS, _PROCESSED, _DATA_DIR,
    load_source_data, build_squad_value_features,
    apply_features_to_future_fixtures, add_derived_features,
    build_feature_matrix,
)

_MODELS_DIR = os.path.join(_DATA_DIR, 'models')

BRACKET = {
    # QF winner-pair → SF fixture (home_team, away_team, date)
    # SF1 draws from QF1/QF2, SF2 draws from QF3/QF4
    'QF1': ('France',    'Morocco',      '2026-07-09'),
    'QF2': ('Spain',     'Belgium',      '2026-07-10'),
    'QF3': ('Norway',    'England',      '2026-07-11'),
    'QF4': ('Argentina', 'Switzerland',  '2026-07-11'),
    'SF1': ('QF1', 'QF2', '2026-07-14'),   # winner of QF1 vs winner of QF2
    'SF2': ('QF3', 'QF4', '2026-07-15'),
    'FIN': ('SF1', 'SF2', '2026-07-19'),
}

SF_DATES  = {'SF1': '2026-07-14', 'SF2': '2026-07-15'}
FIN_DATE  = '2026-07-19'


def load_models(suffix='_v3'):
    rf     = joblib.load(os.path.join(_MODELS_DIR, f'random_forest{suffix}.pkl'))
    lr     = joblib.load(os.path.join(_MODELS_DIR, f'logistic_regression{suffix}.pkl'))
    scaler = joblib.load(os.path.join(_MODELS_DIR, f'scaler{suffix}.pkl'))
    return rf, lr, scaler


def predict_match(rf, lr, scaler, fixture_row):
    """Return (rf_result, lr_result, rf_probs, lr_probs) for a single fixture row."""
    X = pd.DataFrame([fixture_row[FEATURE_COLS]])
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)

    classes = list(rf.classes_)
    hw = classes.index('Home Win')
    aw = classes.index('Away Win')
    dr = classes.index('Draw')

    rf_proba = rf.predict_proba(X)[0]
    lr_proba = lr.predict_proba(X_scaled)[0]

    # In a knockout match one team must advance — pick team with higher win prob
    rf_winner = 'home' if rf_proba[hw] >= rf_proba[aw] else 'away'
    lr_winner = 'home' if lr_proba[hw] >= lr_proba[aw] else 'away'

    rf_probs = {'Home Win': rf_proba[hw], 'Draw': rf_proba[dr], 'Away Win': rf_proba[aw]}
    lr_probs = {'Home Win': lr_proba[hw], 'Draw': lr_proba[dr], 'Away Win': lr_proba[aw]}

    return rf_winner, lr_winner, rf_probs, lr_probs


def build_fixture_df(home_team, away_team, date):
    """Create a minimal future_fixtures DataFrame row for feature enrichment."""
    return pd.DataFrame([{
        'date':             pd.Timestamp(date),
        'home_team':        home_team,
        'away_team':        away_team,
        'home_score':       float('nan'),
        'away_score':       float('nan'),
        'tournament':       'FIFA World Cup',
        'neutral':          1,
        'tournament_weight': 4,
    }])


def enrich_fixture(fixture_df, hist, rankings, squad_values):
    """Apply features to a fixture DataFrame and add derived features."""
    enriched = apply_features_to_future_fixtures(fixture_df, hist, rankings, squad_values)
    enriched = add_derived_features(enriched)
    return enriched


def format_prob(p):
    return f"{p*100:.1f}%"


def main():
    print("Loading data and v3 models...")
    hist, _, valuations, players, rankings = load_source_data(use_prebuilt_features=True)
    squad_values = build_squad_value_features(valuations, players)
    from features import add_derived_features as _add
    hist = _add(hist)

    rf, lr, scaler = load_models('_v3')

    rows = []          # collected results for CSV
    winners = {}       # round_key → {'rf': team, 'lr': team, 'consensus': team}

    def run_match(round_key, home_team, away_team, date, stage):
        fixture_df = build_fixture_df(home_team, away_team, date)
        enriched   = enrich_fixture(fixture_df, hist, rankings, squad_values)
        row        = enriched.iloc[0]

        rf_side, lr_side, rf_probs, lr_probs = predict_match(rf, lr, scaler, row)

        rf_winner  = home_team if rf_side == 'home' else away_team
        lr_winner  = home_team if lr_side == 'home' else away_team
        consensus  = rf_winner if rf_winner == lr_winner else (
            rf_winner if rf_probs['Home Win'] + rf_probs['Away Win'] >
                         lr_probs['Home Win'] + lr_probs['Away Win']
            else lr_winner
        )
        # Break ties by whichever team has higher combined model win prob
        rf_hw = rf_probs['Home Win']; rf_aw = rf_probs['Away Win']
        lr_hw = lr_probs['Home Win']; lr_aw = lr_probs['Away Win']
        avg_home = (rf_hw + lr_hw) / 2
        avg_away = (rf_aw + lr_aw) / 2
        consensus = home_team if avg_home >= avg_away else away_team

        agree = '✓' if rf_winner == lr_winner else '✗'

        print(f"\n{'─'*60}")
        print(f"  {stage}: {home_team} vs {away_team}  ({date})")
        print(f"  RF  → {rf_winner}  "
              f"[HW {format_prob(rf_hw)} / D {format_prob(rf_probs['Draw'])} / AW {format_prob(rf_aw)}]")
        print(f"  LR  → {lr_winner}  "
              f"[HW {format_prob(lr_hw)} / D {format_prob(lr_probs['Draw'])} / AW {format_prob(lr_aw)}]")
        print(f"  Consensus → {consensus}  {agree}")

        rows.append({
            'stage': stage, 'date': date,
            'home_team': home_team, 'away_team': away_team,
            'rf_prediction': rf_winner, 'lr_prediction': lr_winner,
            'consensus_prediction': consensus, 'models_agree': rf_winner == lr_winner,
            'rf_home_win_pct': round(rf_hw * 100, 1),
            'rf_draw_pct':     round(rf_probs['Draw'] * 100, 1),
            'rf_away_win_pct': round(rf_aw * 100, 1),
            'lr_home_win_pct': round(lr_hw * 100, 1),
            'lr_draw_pct':     round(lr_probs['Draw'] * 100, 1),
            'lr_away_win_pct': round(lr_aw * 100, 1),
        })
        winners[round_key] = {'rf': rf_winner, 'lr': lr_winner, 'consensus': consensus}
        return consensus

    # ── Quarterfinals ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  QUARTERFINALS")
    qf1_winner = run_match('QF1', 'France',    'Morocco',      '2026-07-09', 'QF1')
    qf2_winner = run_match('QF2', 'Spain',     'Belgium',      '2026-07-10', 'QF2')
    qf3_winner = run_match('QF3', 'Norway',    'England',      '2026-07-11', 'QF3')
    qf4_winner = run_match('QF4', 'Argentina', 'Switzerland',  '2026-07-11', 'QF4')

    # ── Semifinals ─────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  SEMIFINALS")
    sf1_winner = run_match('SF1', qf1_winner, qf2_winner, '2026-07-14', 'SF1')
    sf2_winner = run_match('SF2', qf3_winner, qf4_winner, '2026-07-15', 'SF2')

    # ── Final ──────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL")
    champion = run_match('FIN', sf1_winner, sf2_winner, '2026-07-19', 'Final')

    print("\n" + "═"*60)
    print(f"  🏆  PREDICTED 2026 WORLD CUP WINNER: {champion}")
    print("═"*60)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(_PROCESSED, 'bracket_predictions.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
