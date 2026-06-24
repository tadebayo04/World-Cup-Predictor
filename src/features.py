import os
import pandas as pd
import numpy as np

SQUAD_VALUE_NAME_MAP = {
    'Bosnia-Herzegovina':   'Bosnia and Herzegovina',
    "Cote d'Ivoire":        "Côte d'Ivoire",
    'Korea, South':         'Korea Republic',
    'Korea, North':         'Korea DPR',
    'United States':        'USA',
    'DR Congo':             'Congo DR',
    'Turkey':               'Türkiye',
    'Czech Republic':       'Czechia',
    'Iran':                 'IR Iran',
    'Cape Verde':           'Cabo Verde',
    'Ireland':              'Republic of Ireland',
    'China':                'China PR',
    'Curacao':              'Curaçao',
}

FEATURE_COLS = [
    'home_rank', 'away_rank', 'home_points', 'away_points', 'points_diff',
    'home_form_5', 'away_form_5', 'home_form_10', 'away_form_10',
    'home_goal_diff_5', 'away_goal_diff_5',
    'h2h_home_win_rate', 'h2h_total_games',
    'home_squad_value_log', 'away_squad_value_log', 'squad_value_diff',
    'neutral', 'tournament_weight',
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_PROCESSED = os.path.join(_DATA_DIR, 'processed')
_RAW = os.path.join(_DATA_DIR, 'raw')


def load_processed_data():
    """Load processed CSVs needed to build the model-ready feature matrix.

    Returns:
        hist: historical_results_features.csv with squad value columns already merged
        future: future_fixtures.csv (72 upcoming 2026 WC matches)
        valuations: player_valuations_clean.csv
        players: players_clean.csv
        rankings: fifa_rankings_clean.csv
    """
    hist = pd.read_csv(os.path.join(_PROCESSED, 'historical_results_features.csv'))
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    hist['neutral'] = hist['neutral'].map({True: 1, False: 0, 'True': 1, 'False': 0}).astype(int)

    future = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures.csv'))
    future['date'] = pd.to_datetime(future['date'])
    future['neutral'] = future['neutral'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    valuations = pd.read_csv(os.path.join(_PROCESSED, 'player_valuations_clean.csv'))
    valuations['date'] = pd.to_datetime(valuations['date'])

    players = pd.read_csv(os.path.join(_PROCESSED, 'players_clean.csv'))

    rankings = pd.read_csv(os.path.join(_PROCESSED, 'fifa_rankings_clean.csv'))
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    rankings = rankings.sort_values('rank_date')

    return hist, future, valuations, players, rankings


def build_squad_value_features(valuations, players):
    """Build squad value timeseries from player valuations and citizenship data.

    Returns DataFrame with columns: [country_of_citizenship, date, squad_value_eur]
    """
    merged = pd.merge(
        valuations[['player_id', 'date', 'market_value_in_eur']],
        players[['player_id', 'country_of_citizenship']],
        on='player_id',
    )
    merged['country_of_citizenship'] = merged['country_of_citizenship'].replace(SQUAD_VALUE_NAME_MAP)
    squad = (
        merged
        .groupby(['country_of_citizenship', 'date'])['market_value_in_eur']
        .sum()
        .reset_index()
        .rename(columns={'market_value_in_eur': 'squad_value_eur'})
        .sort_values('date')
    )
    return squad


def merge_squad_values(df, squad_values):
    """Merge squad value timeseries onto match dataframe via merge_asof.

    NaN rows (pre-2004 matches with no valuation data) are filled with the
    global median so the model sees a neutral signal, not a false zero.
    """
    global_median = squad_values['squad_value_eur'].median()

    home_sv = squad_values.rename(columns={
        'country_of_citizenship': 'home_team',
        'squad_value_eur': 'home_squad_value',
    })
    away_sv = squad_values.rename(columns={
        'country_of_citizenship': 'away_team',
        'squad_value_eur': 'away_squad_value',
    })

    df = df.sort_values('date')
    df = pd.merge_asof(df, home_sv, on='date', by='home_team', direction='backward')
    df = pd.merge_asof(df, away_sv, on='date', by='away_team', direction='backward')

    df['home_squad_value'] = df['home_squad_value'].fillna(global_median)
    df['away_squad_value'] = df['away_squad_value'].fillna(global_median)
    df['squad_value_diff'] = df['home_squad_value'] - df['away_squad_value']
    return df


def add_derived_features(df):
    """Add h2h ratio features and log-scaled squad value columns."""
    # H2H win rate for the home team.
    # team1/team2 use canonical alphabetical ordering (team1 = min(home, away)).
    # Getting this direction wrong would reverse every H2H signal.
    home_wins = np.where(
        df['home_team'] == df['team1'],
        df['h2h_team1_wins'],
        df['h2h_team2_wins'],
    )
    h2h_total = df['h2h_team1_wins'] + df['h2h_team2_wins'] + df['h2h_draw']
    df['h2h_home_win_rate'] = np.where(h2h_total > 0, home_wins / h2h_total, 0.33)
    df['h2h_total_games'] = h2h_total

    df['home_squad_value_log'] = np.log1p(df['home_squad_value'])
    df['away_squad_value_log'] = np.log1p(df['away_squad_value'])
    return df


def _build_team_latest_state(historical):
    """Return a per-team lookup of the most recent form and goal diff values."""
    home_rows = historical[['date', 'home_team', 'home_form_5', 'home_form_10',
                             'home_goal_diff_5', 'home_goal_diff_10']].copy()
    home_rows.columns = ['date', 'team', 'form_5', 'form_10', 'goal_diff_5', 'goal_diff_10']

    away_rows = historical[['date', 'away_team', 'away_form_5', 'away_form_10',
                             'away_goal_diff_5', 'away_goal_diff_10']].copy()
    away_rows.columns = ['date', 'team', 'form_5', 'form_10', 'goal_diff_5', 'goal_diff_10']

    combined = pd.concat([home_rows, away_rows], ignore_index=True).sort_values('date')
    return combined.groupby('team').last().reset_index()


def apply_features_to_future_fixtures(future, historical, rankings, squad_values):
    """Build the full feature set for the 72 upcoming 2026 WC fixtures.

    Uses the historical record as of the most recent match per team for form
    and H2H features; merge_asof for rankings and squad values.
    """
    future = future.copy().sort_values('date')

    # --- FIFA Rankings ---
    home_rank = rankings[['country_full', 'rank_date', 'rank', 'total_points']].rename(
        columns={'country_full': 'home_team', 'rank': 'home_rank', 'total_points': 'home_points'})
    away_rank = rankings[['country_full', 'rank_date', 'rank', 'total_points']].rename(
        columns={'country_full': 'away_team', 'rank': 'away_rank', 'total_points': 'away_points'})

    future = pd.merge_asof(future, home_rank, left_on='date', right_on='rank_date',
                           by='home_team', direction='backward')
    future = pd.merge_asof(future, away_rank, left_on='date', right_on='rank_date',
                           by='away_team', direction='backward')
    future['points_diff'] = future['home_points'] - future['away_points']

    # --- Form and goal diff: most recent appearance per team ---
    team_state = _build_team_latest_state(historical)

    home_state = team_state.rename(columns={
        'team': 'home_team', 'form_5': 'home_form_5', 'form_10': 'home_form_10',
        'goal_diff_5': 'home_goal_diff_5', 'goal_diff_10': 'home_goal_diff_10',
    })
    away_state = team_state.rename(columns={
        'team': 'away_team', 'form_5': 'away_form_5', 'form_10': 'away_form_10',
        'goal_diff_5': 'away_goal_diff_5', 'goal_diff_10': 'away_goal_diff_10',
    })

    future = future.merge(home_state[['home_team', 'home_form_5', 'home_form_10',
                                       'home_goal_diff_5', 'home_goal_diff_10']],
                          on='home_team', how='left')
    future = future.merge(away_state[['away_team', 'away_form_5', 'away_form_10',
                                       'away_goal_diff_5', 'away_goal_diff_10']],
                          on='away_team', how='left')

    # --- H2H: latest cumulative counts per canonical team pair ---
    future['team1'] = future[['home_team', 'away_team']].min(axis=1)
    future['team2'] = future[['home_team', 'away_team']].max(axis=1)

    h2h_current = (historical.sort_values('date')
                   .groupby(['team1', 'team2'])
                   .last()[['h2h_team1_wins', 'h2h_team2_wins', 'h2h_draw']]
                   .reset_index())
    future = future.merge(h2h_current, on=['team1', 'team2'], how='left')
    future[['h2h_team1_wins', 'h2h_team2_wins', 'h2h_draw']] = (
        future[['h2h_team1_wins', 'h2h_team2_wins', 'h2h_draw']].fillna(0)
    )

    # --- Squad values ---
    future = merge_squad_values(future, squad_values)

    # All group-stage WC fixtures share the same tournament weight
    future['tournament_weight'] = 4

    return future


def build_feature_matrix(df, include_target=True):
    """Select the final 18 feature columns (+ optional outcome target)."""
    cols = FEATURE_COLS.copy()
    if include_target:
        cols.append('outcome')
    return df[cols].copy()


def save_model_ready(train_df, future_df):
    os.makedirs(_PROCESSED, exist_ok=True)
    train_df.to_csv(os.path.join(_PROCESSED, 'model_ready.csv'), index=False)
    future_df.to_csv(os.path.join(_PROCESSED, 'future_fixtures_features.csv'), index=False)
    print(f"model_ready.csv:              {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"future_fixtures_features.csv: {future_df.shape[0]} rows × {future_df.shape[1]} cols")


if __name__ == '__main__':
    hist, future, valuations, players, rankings = load_processed_data()
    squad_values = build_squad_value_features(valuations, players)
    hist = add_derived_features(hist)
    future_enriched = apply_features_to_future_fixtures(future, hist, rankings, squad_values)
    future_enriched = add_derived_features(future_enriched)
    train_matrix = build_feature_matrix(hist, include_target=True)
    future_matrix = build_feature_matrix(future_enriched, include_target=False)
    save_model_ready(train_matrix, future_matrix)
