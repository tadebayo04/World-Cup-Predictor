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

_NEUTRAL_MAP = {True: 1, False: 0, 'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_source_data(use_prebuilt_features=False):
    """Load source CSVs for the feature-building pipeline.

    use_prebuilt_features=True  → fast path: loads historical_results_features.csv
    use_prebuilt_features=False → full rebuild path: loads historical_results.csv
    """
    hist_file = 'historical_results_features.csv' if use_prebuilt_features else 'historical_results.csv'
    hist = pd.read_csv(os.path.join(_PROCESSED, hist_file))
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    hist['neutral'] = hist['neutral'].map(_NEUTRAL_MAP).fillna(0).astype(int)

    future = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures.csv'))
    future['date'] = pd.to_datetime(future['date'])
    future['neutral'] = future['neutral'].map(_NEUTRAL_MAP).fillna(0).astype(int)

    valuations = pd.read_csv(os.path.join(_PROCESSED, 'player_valuations_clean.csv'))
    valuations['date'] = pd.to_datetime(valuations['date'])

    players = pd.read_csv(os.path.join(_PROCESSED, 'players_clean.csv'))

    rankings = pd.read_csv(os.path.join(_PROCESSED, 'fifa_rankings_clean.csv'))
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    rankings = rankings.sort_values('rank_date')

    return hist, future, valuations, players, rankings


def load_processed_data():
    """Load processed CSVs (legacy entry point — keeps notebook callers working).

    Returns:
        hist: historical_results_features.csv with squad value columns already merged
        future: future_fixtures.csv
        valuations: player_valuations_clean.csv
        players: players_clean.csv
        rankings: fifa_rankings_clean.csv
    """
    hist = pd.read_csv(os.path.join(_PROCESSED, 'historical_results_features.csv'))
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    hist['neutral'] = hist['neutral'].map(_NEUTRAL_MAP).fillna(0).astype(int)

    future = pd.read_csv(os.path.join(_PROCESSED, 'future_fixtures.csv'))
    future['date'] = pd.to_datetime(future['date'])
    future['neutral'] = future['neutral'].map(_NEUTRAL_MAP).fillna(0).astype(int)

    valuations = pd.read_csv(os.path.join(_PROCESSED, 'player_valuations_clean.csv'))
    valuations['date'] = pd.to_datetime(valuations['date'])

    players = pd.read_csv(os.path.join(_PROCESSED, 'players_clean.csv'))

    rankings = pd.read_csv(os.path.join(_PROCESSED, 'fifa_rankings_clean.csv'))
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    rankings = rankings.sort_values('rank_date')

    return hist, future, valuations, players, rankings


# ──────────────────────────────────────────────────────────────────────────────
# From-scratch feature builders
# ──────────────────────────────────────────────────────────────────────────────

def build_ranking_features(hist, rankings):
    """Merge the closest-prior FIFA ranking snapshot onto each match row."""
    ranks = rankings[['country_full', 'rank_date', 'rank', 'total_points']]

    home_rank = ranks.rename(columns={
        'country_full': 'home_team', 'rank_date': 'date',
        'rank': 'home_rank', 'total_points': 'home_points',
    })
    away_rank = ranks.rename(columns={
        'country_full': 'away_team', 'rank_date': 'date',
        'rank': 'away_rank', 'total_points': 'away_points',
    })

    hist = hist.sort_values('date')
    hist = pd.merge_asof(hist, home_rank, on='date', by='home_team', direction='backward')
    hist = pd.merge_asof(hist, away_rank, on='date', by='away_team', direction='backward')
    hist['points_diff'] = hist['home_points'] - hist['away_points']
    return hist


def build_team_form_timeseries(hist):
    """Build per-team rolling form from all historical results.

    Combines home and away appearances into a single long-format series per team.
    shift(1) before rolling ensures the current match's result is never included.
    Returns one row per (team, match) with pre-match rolling form values.
    """
    home = hist[['date', 'home_team', 'home_score', 'away_score']].copy()
    home.columns = ['date', 'team', 'goals_for', 'goals_against']

    away = hist[['date', 'away_team', 'away_score', 'home_score']].copy()
    away.columns = ['date', 'team', 'goals_for', 'goals_against']

    combined = pd.concat([home, away], ignore_index=True)
    combined['goal_diff'] = combined['goals_for'] - combined['goals_against']
    combined['form_pts'] = np.select(
        [combined['goals_for'] > combined['goals_against'],
         combined['goals_for'] == combined['goals_against']],
        [3, 1], default=0,
    )
    combined = combined.sort_values(['team', 'date']).reset_index(drop=True)

    grp = combined.groupby('team', group_keys=False)
    combined['form_5']       = grp['form_pts'].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
    combined['form_10']      = grp['form_pts'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    combined['goal_diff_5']  = grp['goal_diff'].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
    combined['goal_diff_10'] = grp['goal_diff'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    return combined[['date', 'team', 'form_5', 'form_10', 'goal_diff_5', 'goal_diff_10']]


def merge_form_features(hist, form_ts):
    """Merge pre-match form timeseries onto hist for home and away teams.

    Deduplicates form_ts by (date, team) before merging to avoid row explosion
    from rare same-day appearances.  After merge, deduplicates hist rows that
    arise from British Home Championship fixtures in old data.
    """
    form_deduped = form_ts.drop_duplicates(subset=['date', 'team'])

    home_form = form_deduped.rename(columns={
        'team': 'home_team',
        'form_5': 'home_form_5', 'form_10': 'home_form_10',
        'goal_diff_5': 'home_goal_diff_5', 'goal_diff_10': 'home_goal_diff_10',
    })
    away_form = form_deduped.rename(columns={
        'team': 'away_team',
        'form_5': 'away_form_5', 'form_10': 'away_form_10',
        'goal_diff_5': 'away_goal_diff_5', 'goal_diff_10': 'away_goal_diff_10',
    })

    hist = hist.merge(
        home_form[['date', 'home_team', 'home_form_5', 'home_form_10',
                   'home_goal_diff_5', 'home_goal_diff_10']],
        on=['date', 'home_team'], how='left',
    )
    hist = hist.merge(
        away_form[['date', 'away_team', 'away_form_5', 'away_form_10',
                   'away_goal_diff_5', 'away_goal_diff_10']],
        on=['date', 'away_team'], how='left',
    )
    hist = hist.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
    return hist


def build_h2h_features(hist):
    """Add cumulative H2H counts using canonical (alphabetical) team ordering.

    shift(1) before expanding sum prevents the current match's result leaking
    into its own H2H features.
    """
    hist = hist.copy()
    hist['team1'] = hist[['home_team', 'away_team']].min(axis=1)
    hist['team2'] = hist[['home_team', 'away_team']].max(axis=1)

    # team1 = alphabetically first; team1_won=1 whether team1 was home or away
    hist['_t1_won'] = np.where(
        hist['home_team'] == hist['team1'],
        (hist['home_score'] > hist['away_score']).astype(int),
        (hist['away_score'] > hist['home_score']).astype(int),
    )
    hist['_t2_won'] = np.where(
        hist['away_team'] == hist['team2'],
        (hist['away_score'] > hist['home_score']).astype(int),
        (hist['home_score'] > hist['away_score']).astype(int),
    )
    hist['_draw'] = (hist['home_score'] == hist['away_score']).astype(int)

    hist = hist.sort_values(['team1', 'team2', 'date']).reset_index(drop=True)

    for src, dst in [('_t1_won', 'h2h_team1_wins'),
                     ('_t2_won', 'h2h_team2_wins'),
                     ('_draw',   'h2h_draw')]:
        hist[dst] = (hist
                     .groupby(['team1', 'team2'])[src]
                     .transform(lambda x: x.shift(1).expanding().sum().fillna(0)))

    hist = hist.drop(columns=['_t1_won', '_t2_won', '_draw'])
    hist = hist.sort_values('date').reset_index(drop=True)
    return hist


def build_all_features_from_scratch(hist_raw, rankings, squad_values):
    """Build the full feature matrix from cleaned (but un-featured) match data.

    Form and H2H are computed on ALL historical data before filtering to post-1992
    so that early 1993 matches correctly reflect pre-1992 rolling history.
    """
    hist = hist_raw.copy()
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    hist['neutral'] = hist['neutral'].map(_NEUTRAL_MAP).fillna(0).astype(int)

    # Form and H2H on the full dataset (preserves pre-1992 context for early post-92 matches)
    form_ts = build_team_form_timeseries(hist)
    hist = merge_form_features(hist, form_ts)
    hist = build_h2h_features(hist)

    # Rankings: filter to post-1992 + matches with valid ranking data
    hist = build_ranking_features(hist, rankings)
    hist = hist[(hist['date'] >= '1993-01-01') & hist['home_rank'].notna() & hist['away_rank'].notna()]
    hist = hist.reset_index(drop=True)

    # Squad values via merge_asof
    hist = merge_squad_values(hist, squad_values)

    return hist


# ──────────────────────────────────────────────────────────────────────────────
# Squad value features (used in both fast and full-rebuild paths)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Derived features (applied last, in both paths)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Future fixtures
# ──────────────────────────────────────────────────────────────────────────────

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
    """Build the full feature set for upcoming fixtures.

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

    future['tournament_weight'] = 4

    return future


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df, include_target=True, include_date=False):
    """Select the final 18 feature columns (+ optional outcome target and date)."""
    cols = []
    if include_date:
        cols.append('date')
    cols.extend(FEATURE_COLS)
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
    import argparse
    parser = argparse.ArgumentParser(description='Build model-ready feature matrix')
    parser.add_argument('--use-prebuilt', action='store_true',
                        help='Skip full rebuild and load existing historical_results_features.csv')
    args = parser.parse_args()

    hist, future, valuations, players, rankings = load_source_data(
        use_prebuilt_features=args.use_prebuilt
    )
    squad_values = build_squad_value_features(valuations, players)

    if not args.use_prebuilt:
        hist = build_all_features_from_scratch(hist, rankings, squad_values)
        hist.to_csv(os.path.join(_PROCESSED, 'historical_results_features.csv'), index=False)
        print(f"Saved historical_results_features.csv ({len(hist):,} rows)")

    hist = add_derived_features(hist)
    future_enriched = apply_features_to_future_fixtures(future, hist, rankings, squad_values)
    future_enriched = add_derived_features(future_enriched)

    save_model_ready(
        build_feature_matrix(hist, include_target=True, include_date=True),
        build_feature_matrix(future_enriched, include_target=False),
    )
