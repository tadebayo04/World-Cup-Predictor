"""Export match predictions + explanatory features as JSON for the front-end UI.

Combines the group-stage predictions (predictions.csv) and knockout
predictions (knockout_predictions.csv) with the per-match feature set the
models actually saw (FIFA rank, form, goal difference, head-to-head, squad
value, venue), then writes ui/data.js — a single self-contained payload the
static UI loads with no server or build step.

Usage:
    python src/export_ui_data.py
"""
import json
import os

import numpy as np
import pandas as pd

from features import (
    SQUAD_VALUE_NAME_MAP,
    apply_features_to_future_fixtures,
    add_derived_features,
    build_squad_value_features,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_PROCESSED = os.path.join(_DATA_DIR, 'processed')
_UI_DIR = os.path.join(os.path.dirname(__file__), '..', 'ui')

# Group-stage fixtures where a host nation plays on home soil (from
# predictions_clean.csv "Venue" column). Everything else is neutral.
_HOSTS = {'USA', 'Mexico', 'Canada'}

FLAGS = {
    'Algeria': '🇩🇿', 'Argentina': '🇦🇷', 'Australia': '🇦🇺', 'Austria': '🇦🇹',
    'Belgium': '🇧🇪', 'Bosnia and Herzegovina': '🇧🇦', 'Brazil': '🇧🇷',
    'Cabo Verde': '🇨🇻', 'Canada': '🇨🇦', 'Colombia': '🇨🇴', 'Congo DR': '🇨🇩',
    "Côte d'Ivoire": '🇨🇮', 'Croatia': '🇭🇷', 'Curaçao': '🇨🇼', 'Czechia': '🇨🇿',
    'Ecuador': '🇪🇨', 'Egypt': '🇪🇬', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'France': '🇫🇷',
    'Germany': '🇩🇪', 'Ghana': '🇬🇭', 'Haiti': '🇭🇹', 'IR Iran': '🇮🇷', 'Iraq': '🇮🇶',
    'Italy': '🇮🇹', 'Japan': '🇯🇵', 'Jordan': '🇯🇴', 'Korea Republic': '🇰🇷',
    'Mexico': '🇲🇽', 'Morocco': '🇲🇦', 'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿',
    'Norway': '🇳🇴', 'Panama': '🇵🇦', 'Paraguay': '🇵🇾', 'Portugal': '🇵🇹',
    'Qatar': '🇶🇦', 'Saudi Arabia': '🇸🇦', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Senegal': '🇸🇳',
    'South Africa': '🇿🇦', 'Spain': '🇪🇸', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭',
    'Tunisia': '🇹🇳', 'Türkiye': '🇹🇷', 'USA': '🇺🇸', 'Uruguay': '🇺🇾',
    'Uzbekistan': '🇺🇿',
}


def load_predictions():
    """Load both prediction files, tagged with tournament stage."""
    group = pd.read_csv(os.path.join(_PROCESSED, 'predictions.csv'))
    group['stage'] = 'Group Stage'

    knockout = pd.read_csv(os.path.join(_PROCESSED, 'knockout_predictions.csv'))
    knockout['stage'] = 'Round of 32'

    return pd.concat([group, knockout], ignore_index=True)


def build_fixture_features(preds):
    """Rebuild the model feature set for every predicted fixture.

    knockout fixtures still have their features on disk, but the group-stage
    feature file was overwritten by the knockout run — so features for all
    fixtures are recomputed here with the same pipeline used at training time.
    """
    hist = pd.read_csv(os.path.join(_PROCESSED, 'historical_results_features.csv'))
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    rankings = pd.read_csv(os.path.join(_PROCESSED, 'fifa_rankings_clean.csv'))
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    rankings = rankings.sort_values('rank_date')

    valuations = pd.read_csv(os.path.join(_PROCESSED, 'player_valuations_clean.csv'))
    valuations['date'] = pd.to_datetime(valuations['date'])
    players = pd.read_csv(os.path.join(_PROCESSED, 'players_clean.csv'))
    squad_values = build_squad_value_features(valuations, players)

    fixtures = preds[['date', 'home_team', 'away_team', 'stage']].copy()
    fixtures['date'] = pd.to_datetime(fixtures['date'])
    # Group stage: hosts at home get the advantage. Knockout: reuse the exact
    # flags from future_fixtures.csv so features match what the model saw.
    ko = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures.csv'))
    ko['date'] = pd.to_datetime(ko['date'])
    ko['ko_neutral'] = ko['neutral'].astype(bool).astype(int)
    fixtures = fixtures.merge(
        ko[['date', 'home_team', 'away_team', 'ko_neutral']],
        on=['date', 'home_team', 'away_team'], how='left',
    )
    group_neutral = (~fixtures['home_team'].isin(_HOSTS)).astype(int)
    fixtures['neutral'] = fixtures['ko_neutral'].fillna(group_neutral).astype(int)
    fixtures = fixtures.drop(columns=['ko_neutral', 'stage'])

    feats = apply_features_to_future_fixtures(fixtures, hist, rankings, squad_values)
    feats = add_derived_features(feats)
    return feats


def build_display_squad_values():
    """Realistic squad market values for display in the UI.

    The model's squad_value feature sums only the valuations updated on a
    single snapshot date, which is fine as a log-scaled model input but reads
    as noise when shown raw (e.g. France at €350K). For display, sum each
    country's top-26 players by latest market valuation, restricted to
    players still being valued (2025 onwards).
    """
    valuations = pd.read_csv(
        os.path.join(_PROCESSED, 'player_valuations_clean.csv'), parse_dates=['date'])
    players = pd.read_csv(os.path.join(_PROCESSED, 'players_clean.csv'))

    latest = valuations.sort_values('date').groupby('player_id').last().reset_index()
    active = latest[latest['date'] >= '2025-01-01']
    merged = active.merge(players[['player_id', 'country_of_citizenship']], on='player_id')
    merged['country_of_citizenship'] = merged['country_of_citizenship'].replace(SQUAD_VALUE_NAME_MAP)

    top26 = (merged.sort_values('market_value_in_eur', ascending=False)
                   .groupby('country_of_citizenship').head(26))
    return top26.groupby('country_of_citizenship')['market_value_in_eur'].sum().to_dict()


def fmt_value(eur):
    """Compact euro amount: €1.2B / €340M / €5.1M."""
    if eur <= 0:
        return '—'
    if eur >= 1e9:
        return f"€{eur / 1e9:.1f}B"
    if eur >= 1e6:
        return f"€{eur / 1e6:.0f}M"
    return f"€{eur / 1e3:.0f}K"


def build_reasons(row, winner):
    """Human-readable reasons the predicted winner is favored (or, for a
    predicted draw, why the teams are evenly matched)."""
    home, away = row['home_team'], row['away_team']
    reasons = []

    if winner == 'Draw':
        if abs(row['home_rank'] - row['away_rank']) <= 15:
            reasons.append(
                f"Closely matched on FIFA ranking: #{int(row['home_rank'])} vs #{int(row['away_rank'])}"
            )
        if abs(row['home_form_5']) > 0 and abs(row['home_form_5'] - row['away_form_5']) <= 0.5:
            reasons.append(
                f"Similar recent form: {row['home_form_5']:.1f} vs {row['away_form_5']:.1f} points per game over the last 5"
            )
        if not reasons:
            reasons.append("Neither side has a decisive edge across ranking, form, or squad value")
        return reasons

    w, l = ('home', 'away') if winner == home else ('away', 'home')
    loser = away if winner == home else home

    rank_gap = row[f'{l}_rank'] - row[f'{w}_rank']
    if rank_gap > 0:
        reasons.append(
            f"Ranked {int(rank_gap)} places higher on the FIFA rankings "
            f"(#{int(row[f'{w}_rank'])} vs #{int(row[f'{l}_rank'])})"
        )

    form_gap = row[f'{w}_form_5'] - row[f'{l}_form_5']
    if form_gap >= 0.3:
        reasons.append(
            f"Better recent form: {row[f'{w}_form_5']:.1f} vs {row[f'{l}_form_5']:.1f} "
            f"points per game over the last 5 matches"
        )

    gd_gap = row[f'{w}_goal_diff_5'] - row[f'{l}_goal_diff_5']
    if gd_gap >= 0.5:
        reasons.append(
            f"Scoring more freely: {row[f'{w}_goal_diff_5']:+.1f} vs {row[f'{l}_goal_diff_5']:+.1f} "
            f"average goal difference over the last 5"
        )

    wv, lv = row[f'{w}_display_value'], row[f'{l}_display_value']
    if lv > 0 and wv / lv >= 1.5:
        ratio = wv / lv
        reasons.append(
            f"Squad worth {ratio:.0f}× more on the transfer market "
            f"({fmt_value(wv)} vs {fmt_value(lv)})" if ratio >= 2 else
            f"More valuable squad on the transfer market ({fmt_value(wv)} vs {fmt_value(lv)})"
        )

    total = row['h2h_total_games']
    if total >= 3:
        hwr = row['h2h_home_win_rate']
        wr = hwr if winner == home else max(0.0, 1 - hwr - (row['h2h_draw'] / total if total else 0))
        w_wins = row['h2h_team1_wins'] if winner == row['team1'] else row['h2h_team2_wins']
        l_wins = row['h2h_team2_wins'] if winner == row['team1'] else row['h2h_team1_wins']
        if w_wins > l_wins:
            d = int(row['h2h_draw'])
            reasons.append(
                f"Leads the head-to-head record {int(w_wins)}–{int(l_wins)} "
                f"({d} draw{'s' if d != 1 else ''}) against {loser}"
            )

    if winner == home and not row['neutral']:
        reasons.append("Playing on home soil as a tournament host")

    if not reasons:
        reasons.append("Marginal edges across ranking, form and squad value add up in the model")
    return reasons


def main():
    preds = load_predictions()
    feats = build_fixture_features(preds)

    preds['date'] = pd.to_datetime(preds['date'])
    merged = preds.merge(
        feats,
        on=['date', 'home_team', 'away_team'],
        how='left',
        validate='one_to_one',
    )
    assert merged['home_rank'].notna().all(), "feature merge left gaps"

    display_sv = build_display_squad_values()
    merged['home_display_value'] = merged['home_team'].map(display_sv).fillna(0.0)
    merged['away_display_value'] = merged['away_team'].map(display_sv).fillna(0.0)

    matches = []
    for _, r in merged.sort_values(['date', 'home_team']).iterrows():
        # Blend the two models' probabilities for the headline call.
        p_home = (r['rf_home_win_prob'] + r['lr_home_win_prob']) / 2
        p_away = (r['rf_away_win_prob'] + r['lr_away_win_prob']) / 2
        p_draw = (r['rf_draw_prob'] + r['lr_draw_prob']) / 2

        outcomes = {'Home Win': p_home, 'Away Win': p_away, 'Draw': p_draw}
        call = max(outcomes, key=outcomes.get)
        winner = {'Home Win': r['home_team'], 'Away Win': r['away_team'], 'Draw': 'Draw'}[call]

        matches.append({
            'date': r['date'].strftime('%Y-%m-%d'),
            'stage': r['stage'],
            'home': r['home_team'],
            'away': r['away_team'],
            'homeFlag': FLAGS.get(r['home_team'], '🏳️'),
            'awayFlag': FLAGS.get(r['away_team'], '🏳️'),
            'neutral': bool(r['neutral']),
            'winner': winner,
            'call': call,
            'confidence': round(float(outcomes[call]), 4),
            'probs': {
                'home': round(float(p_home), 4),
                'draw': round(float(p_draw), 4),
                'away': round(float(p_away), 4),
            },
            'models': {
                'rf': {
                    'prediction': r['rf_prediction'],
                    'home': round(float(r['rf_home_win_prob']), 4),
                    'draw': round(float(r['rf_draw_prob']), 4),
                    'away': round(float(r['rf_away_win_prob']), 4),
                },
                'lr': {
                    'prediction': r['lr_prediction'],
                    'home': round(float(r['lr_home_win_prob']), 4),
                    'draw': round(float(r['lr_draw_prob']), 4),
                    'away': round(float(r['lr_away_win_prob']), 4),
                },
                'agree': r['rf_prediction'] == r['lr_prediction'],
            },
            'features': {
                'rank': [int(r['home_rank']), int(r['away_rank'])],
                'points': [round(float(r['home_points'])), round(float(r['away_points']))],
                'form5': [round(float(r['home_form_5']), 2), round(float(r['away_form_5']), 2)],
                'goalDiff5': [round(float(r['home_goal_diff_5']), 2), round(float(r['away_goal_diff_5']), 2)],
                'squadValue': [float(r['home_display_value']), float(r['away_display_value'])],
                'squadValueLabel': [fmt_value(r['home_display_value']), fmt_value(r['away_display_value'])],
                'h2h': {
                    'homeWins': int(r['h2h_team1_wins'] if r['home_team'] == r['team1'] else r['h2h_team2_wins']),
                    'awayWins': int(r['h2h_team2_wins'] if r['home_team'] == r['team1'] else r['h2h_team1_wins']),
                    'draws': int(r['h2h_draw']),
                },
            },
            'reasons': build_reasons(r, winner),
        })

    payload = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'tournament': 'FIFA World Cup 2026',
        'matches': matches,
    }

    os.makedirs(_UI_DIR, exist_ok=True)
    out = os.path.join(_UI_DIR, 'data.js')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('// Generated by src/export_ui_data.py — do not edit by hand.\n')
        f.write('const WC_DATA = ')
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write(';\n')
    print(f"Wrote {out}  ({len(matches)} matches)")


if __name__ == '__main__':
    main()
