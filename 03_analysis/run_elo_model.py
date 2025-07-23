# nhl_stats/03_analysis/run_elo_model.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# --- Configuration & Path Handling ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

# --- File Paths ---
PLAYER_GAME_SUMMARIES_FILE = PROJECT_ROOT / "data/processed/player_game_summaries.csv"
TEAMMATE_MATCHUPS_FILE = PROJECT_ROOT / "data/processed/teammate_matchups.csv"
OPPONENT_MATCHUPS_FILE = PROJECT_ROOT / "data/processed/opponent_matchups.csv"
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
GAME_TYPES_FILE = PROJECT_ROOT / "data/processed/game_types.csv"
FINAL_ELO_OUTPUT_FILE = PROJECT_ROOT / config.FINAL_ELO_OUTPUT_FILE

# --- Elo Calculation Functions ---
def calculate_expected_score(player_elo, avg_teammate_elo, avg_opponent_elo, is_home):
    home_ice_advantage = config.HOME_ICE_ADVANTAGE_ELO if is_home else 0
    player_unit_strength = (0.2 * player_elo) + (0.8 * avg_teammate_elo)
    elo_diff = player_unit_strength - avg_opponent_elo + home_ice_advantage
    return 1 / (1 + 10**(-elo_diff / 400))

def update_elo(k, actual_score, expected_score):
    return k * (actual_score - expected_score)

def main():
    """
    Main function to run the "Positionally Normalized Game-Weight" Elo model
    for the REGULAR SEASON only, without a hard cap on performance.
    """
    print("--- Starting Final Elo Model Calculation (Your Design) ---")

    try:
        player_summaries = pd.read_csv(PLAYER_GAME_SUMMARIES_FILE)
        teammate_matchups = pd.read_csv(TEAMMATE_MATCHUPS_FILE)
        opponent_matchups = pd.read_csv(OPPONENT_MATCHUPS_FILE)
        player_db = pd.read_csv(PLAYER_DATABASE_FILE)
        game_types = pd.read_csv(GAME_TYPES_FILE)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required summary file was not found: {e}")
        sys.exit(1)

    print("Filtering for regular season games only...")
    regular_season_games = game_types[game_types['game_type'] == 'regular']['game_id']
    player_summaries = player_summaries[player_summaries['game_id'].isin(regular_season_games)]
    teammate_matchups = teammate_matchups[teammate_matchups['game_id'].isin(regular_season_games)]
    opponent_matchups = opponent_matchups[opponent_matchups['game_id'].isin(regular_season_games)]
    
    toi_per_team = teammate_matchups.groupby(['player_id', 'team', 'game_id'])['shared_toi'].sum().reset_index()
    toi_per_team['total_toi_mins'] = (toi_per_team['shared_toi'] / 60) / 4
    
    player_summaries = pd.merge(player_summaries, toi_per_team[['player_id', 'game_id', 'total_toi_mins']], on=['player_id', 'game_id'], how='left')
    player_summaries = pd.merge(player_summaries, player_db[['player_id', 'position']], on='player_id', how='left')
    skater_summaries = player_summaries[player_summaries['position'] != 'G'].copy()
    skater_summaries['total_toi_mins'] = skater_summaries['total_toi_mins'].fillna(0)
    
    all_skaters = skater_summaries['player_id'].unique()
    elo_ratings = {player_id: config.INITIAL_ELO for player_id in all_skaters}
    
    game_groups = skater_summaries.groupby('game_id')
    
    print("\nCalculating Elo ratings game by game...")
    for game_id, game_df in tqdm(game_groups, total=len(game_groups)):
        elo_changes = {}
        game_teammate_matchups = teammate_matchups[teammate_matchups['game_id'] == game_id]
        game_opponent_matchups = opponent_matchups[opponent_matchups['game_id'] == game_id]

        for _, player_row in game_df.iterrows():
            player_id, is_home = player_row['player_id'], player_row['team'] == 'home'
            
            teammates = game_teammate_matchups[game_teammate_matchups['player_id'] == player_id]
            opponents = game_opponent_matchups[game_opponent_matchups['player_id'] == player_id]
            if teammates.empty or opponents.empty or player_row['total_toi_mins'] <= 0: continue

            avg_teammate_elo = np.average(teammates['teammate_id'].map(elo_ratings), weights=teammates['shared_toi'])
            avg_opponent_elo = np.average(opponents['opponent_id'].map(elo_ratings), weights=opponents['shared_toi'])
            
            expected_score = calculate_expected_score(elo_ratings[player_id], avg_teammate_elo, avg_opponent_elo, is_home)
            
            # --- THIS IS THE FIX FOR THE PERFORMANCE CAP ---
            # We no longer use np.clip(), allowing outlier performances to be fully rewarded.
            normalized_diff = player_row['game_xg_diff'] / config.GAME_XG_NORMALIZATION_FACTOR
            actual_score = 0.5 * normalized_diff + 0.5
            
            if player_row['position'] == 'D':
                normalization_toi = config.DEFENSEMAN_NORMALIZATION_TOI
            else:
                normalization_toi = config.FORWARD_NORMALIZATION_TOI
            
            num_games_played = player_row['total_toi_mins'] / normalization_toi
            # We scale the K-factor by the number of "games" played.
            elo_change = update_elo(config.K_FACTOR * num_games_played, actual_score, expected_score)
            
            elo_changes[player_id] = elo_change

        for pid, change in elo_changes.items():
            elo_ratings[pid] += change
    
    print("\nFinalizing player ratings and performance metrics...")
    final_ratings_df = pd.DataFrame(elo_ratings.items(), columns=['player_id', 'elo_rating'])
    
    per_team_stats = skater_summaries.groupby(['player_id', 'team']).agg(
        games_played=('game_id', 'nunique'),
        total_toi_mins=('total_toi_mins', 'sum'),
        season_xg_for=('game_xg_for', 'sum'),
        season_xg_against=('game_xg_against', 'sum')
    ).reset_index()

    total_stats = per_team_stats.groupby('player_id').agg(
        games_played=('games_played', 'sum'),
        total_toi_mins=('total_toi_mins', 'sum'),
        season_xg_for=('season_xg_for', 'sum'),
        season_xg_against=('season_xg_against', 'sum')
    ).reset_index()
    total_stats['team'] = 'TOT'

    traded_player_ids = per_team_stats.groupby('player_id').filter(lambda x: len(x) > 1)['player_id'].unique()
    final_summary = pd.concat([per_team_stats, total_stats[total_stats['player_id'].isin(traded_player_ids)]])
    
    # --- THIS IS THE FIX FOR THE KEYERROR ---
    player_db_to_merge = player_db.drop(columns=['games_played'])
    final_output_df = pd.merge(final_summary, player_db_to_merge, on='player_id', how='left')
    final_output_df = pd.merge(final_output_df, final_ratings_df, on='player_id', how='left')
    
    final_output_df['xg_percentage'] = (final_output_df['season_xg_for'] / (final_output_df['season_xg_for'] + final_output_df['season_xg_against'])).fillna(0) * 100

    final_output_df.sort_values(by='elo_rating', ascending=False, inplace=True)
    final_output_df.insert(0, 'rank', range(1, 1 + len(final_output_df)))
    
    final_output_df = final_output_df[[
        'rank', 'full_name', 'position', 'team', 'elo_rating', 'games_played', 'total_toi_mins',
        'xg_percentage', 'season_xg_for', 'season_xg_against', 'player_id'
    ]].round(2)
    
    FINAL_ELO_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(FINAL_ELO_OUTPUT_FILE, index=False)

    print(f"\n--- ELO MODEL COMPLETE ---")
    print(f"Final regular season skater rankings saved to: {FINAL_ELO_OUTPUT_FILE}")
    
    leaderboard_tot = final_output_df[final_output_df['team'] == 'TOT']
    leaderboard_single = final_output_df[~final_output_df['player_id'].isin(traded_player_ids)]
    leaderboard = pd.concat([leaderboard_tot, leaderboard_single]).sort_values('elo_rating', ascending=False)
    leaderboard_filtered = leaderboard[leaderboard['games_played'] >= 20]

    print("\nTop 20 Skaters (min. 20 GP, Total Season) by Game-Weighted 5v5 xG Elo Rating:")
    print(leaderboard_filtered.head(20).to_string(index=False))

if __name__ == "__main__":
    main()