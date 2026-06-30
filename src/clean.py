import os
import pandas as pd
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), '..')

def load_data():
    # Match results — na_values=['NA'] ensures knockout rows with "NA,NA" scores
    # are parsed as NaN rather than the string "NA", so they land in future_fixtures
    results = pd.read_csv(os.path.join(_ROOT, 'data/raw/results.csv'), na_values=['NA'])

    # FIFA rankings
    fifa_rankings = pd.read_csv(os.path.join(_ROOT, 'data/raw/fifa_rankings.csv'))

    # Transfermarkt files
    players = pd.read_csv(os.path.join(_ROOT, 'data/raw/transfermarkt/players.csv'))
    player_valuations = pd.read_csv(
        os.path.join(_ROOT, 'data/raw/transfermarkt/player_valuations.csv'))
    national_teams = pd.read_csv(os.path.join(_ROOT, 'data/raw/transfermarkt/national_teams.csv'))
    games = pd.read_csv(os.path.join(_ROOT, 'data/raw/transfermarkt/games.csv'))
    lineups_path = os.path.join(_ROOT, 'data/raw/transfermarkt/game_lineups.csv')
    game_lineups = (pd.read_csv(lineups_path, low_memory=False)
                    if os.path.exists(lineups_path) else pd.DataFrame())
    
    return results, fifa_rankings, players, player_valuations, national_teams, games, game_lineups

def clean_results(results):
    results['date'] = pd.to_datetime(results['date'])

    results = results.replace({
    'South Korea': 'Korea Republic',
    'North Korea': 'Korea DPR',
    'Ivory Coast': "Côte d'Ivoire",
    'Turkey': 'Türkiye',
    'Iran': 'IR Iran',
    'United States': 'USA',
    'DR Congo': 'Congo DR',
    'Czech Republic': 'Czechia',
    'Cape Verde': 'Cabo Verde'
        }
    ) 

    historical_results = results[results['home_score'].notna()]

    future_fixtures = results[results['home_score'].isnull()]

    historical_results.loc[:, 'home_score'] = historical_results['home_score'].astype(int)
    historical_results.loc[:, 'away_score'] = historical_results['away_score'].astype(int)



    conditions = [
        historical_results['home_score'] > historical_results['away_score'],
        historical_results['home_score'] < historical_results['away_score'],
        historical_results['home_score'] == historical_results['away_score']
    ]

    choices = [
        'Home Win',
        'Away Win',
        'Draw'
    ]

    historical_results.loc[:, 'outcome'] = np.select(conditions, choices, default='Unknown')

    tournament_weights = {
    # Tier 4 - World Cup
    'FIFA World Cup': 4,
    
    # Tier 3 - Major tournaments
    'UEFA Euro': 3,
    'Copa América': 3,   
    'African Cup of Nations': 3,
    'AFC Asian Cup': 3,
    'Gold Cup': 3,

    
    # Tier 2 - Qualification
    'FIFA World Cup qualification': 2,
    'UEFA Euro qualification': 2,
    'African Cup of Nations qualification': 2,
    'AFC Asian Cup qualification': 2,
    'Gold Cup qualification': 2,
    
    # Tier 1 - Friendly
    'Friendly': 1,
    'Olympic Games': 1
    }

    historical_results.loc[:, 'tournament_weight'] = historical_results['tournament'].map(tournament_weights).fillna(1).astype(int)

    return historical_results, future_fixtures

def clean_rankings(fifa_rankings):
    fifa_rankings['rank_date'] = pd.to_datetime(fifa_rankings['rank_date'])
    fifa_rankings = fifa_rankings.drop('Unnamed: 0', axis=1)
    fifa_rankings = fifa_rankings[fifa_rankings['rank'].notna()]

    return fifa_rankings

def clean_players(players):
    players = players[['player_id', 'name', 'country_of_citizenship', 'position', 'international_caps', 'market_value_in_eur']]
    players = players[players['country_of_citizenship'].notna()]

    return players

def clean_valuations(player_valuations):
    player_valuations = player_valuations.drop('player_club_domestic_competition_id', axis = 1)
    player_valuations['date'] = pd.to_datetime(player_valuations['date'])

    return player_valuations

def clean_national_teams(national_teams):
    national_teams = national_teams[['national_team_id', 'name', 'country_name', 'confederation']]

    return national_teams

def save_data(historical_results, future_fixtures, fifa_rankings, players, player_valuations, national_teams):
    processed = os.path.join(_ROOT, 'data/processed')
    os.makedirs(processed, exist_ok=True)
    historical_results.to_csv(os.path.join(processed, 'historical_results.csv'), index=False)
    future_fixtures.to_csv(os.path.join(processed, 'future_fixtures.csv'), index=False)
    fifa_rankings.to_csv(os.path.join(processed, 'fifa_rankings_clean.csv'), index=False)
    players.to_csv(os.path.join(processed, 'players_clean.csv'), index=False)
    player_valuations.to_csv(os.path.join(processed, 'player_valuations_clean.csv'), index=False)
    national_teams.to_csv(os.path.join(processed, 'national_teams_clean.csv'), index=False)
    print('All files saved to data/processed/')

if __name__ == '__main__':
    results, fifa_rankings, players, player_valuations, national_teams, games, game_lineups = load_data()

    fifa_rankings = clean_rankings(fifa_rankings)
    players = clean_players(players)
    player_valuations = clean_valuations(player_valuations)
    national_teams = clean_national_teams(national_teams)
    historical_results, future_fixtures = clean_results(results)
    save_data(historical_results, future_fixtures, fifa_rankings, players, player_valuations, national_teams)