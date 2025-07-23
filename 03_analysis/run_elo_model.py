# nhl_stats/03_analysis/run_elo_model.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# --- Configuration & Path Handling ---
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

# --- File Paths from Config ---
PLAYER_GAME_SUMMARIES_FILE = PROJECT_ROOT / config.PLAYER_GAME_SUMMARIES_FILE
TEAMMATE_MATCHUPS_FILE = PROJECT_ROOT / config.TEAMMATE_MATCHUPS_FILE
OPPONENT_MATCHUPS_FILE = PROJECT_ROOT / config.OPPONENT_MATCHUPS_FILE
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
MONEYPUCK_PBP_FILE = PROJECT_ROOT / config.MONEYPUCK_PBP_FILE
FINAL_ELO_OUTPUT_FILE = PROJECT_ROOT / config.FINAL_ELO_OUTPUT_FILE

# --- Elo Calculation Functions (Unchanged) ---
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
    for the REGULAR SEASON only.
    """
    print("--- Starting Final Elo Model Calculation (Uncapped Performance) ---")

    try:
        player_summaries = pd.read_csv(PLAYER_GAME_SUMMARIES_FILE)
        teammate_matchups = pd.read_csv(TEAMMATE_MATCHUPS_FILE)
        opponent_matchups = pd.read_csv(OPPONENT_MATCHUPS_FILE)
        player_db = pd.read_csv(PLAYER_DATABASE_FILE)
        moneypuck_df = pd.read_csv(MONEYPUCK_PBP_FILE, low_memory=False)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required summary file was not found: {e}")
        sys.exit(1)

    # --- Data Preparation ---
    print("Preparing data and filtering for regular season games...")
    
    moneypuck_df['game_id_full'] = (config.SEASON_TO_PROCESS * 1_000_000) + moneypuck_df['game_id']
    home_team_map = moneypuck_df.drop_duplicates('game_id_full').set_index('game_id_full')['homeTeamCode'].to_dict()

    all_game_ids = player_summaries['game_id'].unique()
    regular_season_ids = [gid for gid in all_game_ids if str(gid)[4:6] == '02']
    
    player_summaries = player_summaries[player_summaries['game_id'].isin(regular_season_ids)].copy()
    teammate_matchups = teammate_matchups[teammate_matchups['game_id'].isin(regular_season_ids)].copy()
    opponent_matchups = opponent_matchups[opponent_matchups['game_id'].isin(regular_season_ids)].copy()

    player_summaries['toi_5v5_mins'] = player_summaries['toi_5v5'] / 60
    player_summaries['game_xg_diff'] = player_summaries['xg_for_5v5'] - player_summaries['xg_against_5v5']
    
    skater_summaries = pd.merge(player_summaries, player_db[['player_id', 'position']], on='player_id', how='left')
    skater_summaries = skater_summaries[skater_summaries['position'] != 'G'].copy()
    
    all_skaters = skater_summaries['player_id'].unique()
    elo_ratings = {player_id: config.INITIAL_ELO for player_id in all_skaters}
    
    # --- NEW: Dictionary to store avg opponent Elo ---
    avg_opp_elo_log = {}

    game_groups = skater_summaries.sort_values(by='game_id').groupby('game_id')
    
    print(f"\nCalculating Elo ratings for {len(game_groups)} regular season games...")
    for game_id, game_df in tqdm(game_groups, total=len(game_groups)):
        elo_changes = {}
        game_teammate_matchups = teammate_matchups[teammate_matchups['game_id'] == game_id]
        game_opponent_matchups = opponent_matchups[opponent_matchups['game_id'] == game_id]
        home_team = home_team_map.get(game_id)

        for _, player_row in game_df.iterrows():
            player_id = player_row['player_id']
            is_home = (player_row['team'] == home_team)
            
            teammates = game_teammate_matchups[game_teammate_matchups['player_id'] == player_id]
            opponents = game_opponent_matchups[game_opponent_matchups['player_id'] == player_id]
            
            if teammates.empty or opponents.empty or player_row['toi_5v5_mins'] <= 0: continue

            avg_teammate_elo = np.average(teammates['teammate_id'].map(elo_ratings), weights=teammates['shared_toi'])
            avg_opponent_elo = np.average(opponents['opponent_id'].map(elo_ratings), weights=opponents['shared_toi'])
            
            # --- NEW: Log the avg opponent elo for this player's game ---
            avg_opp_elo_log[(game_id, player_id)] = avg_opponent_elo

            expected_score = calculate_expected_score(elo_ratings[player_id], avg_teammate_elo, avg_opponent_elo, is_home)
            
            # --- NEW: Uncapped Actual Score ---
            # This makes the model much more sensitive to large xG differentials.
            # A player's raw xG differential for the game is now the primary driver of performance.
            actual_score = 0.5 + (player_row['game_xg_diff'] / 2.0)

            normalization_toi = config.DEFENSEMAN_NORMALIZATION_TOI if player_row['position'] == 'D' else config.FORWARD_NORMALIZATION_TOI
            
            num_games_played = player_row['toi_5v5_mins'] / normalization_toi
            elo_change = update_elo(config.K_FACTOR * num_games_played, actual_score, expected_score)
            
            elo_changes[player_id] = elo_changes.get(player_id, 0) + elo_change

        for pid, change in elo_changes.items():
            elo_ratings[pid] += change
    
    print("\nFinalizing player ratings and performance metrics...")
    final_ratings_df = pd.DataFrame(elo_ratings.items(), columns=['player_id', 'elo_rating'])
    
    per_team_stats = skater_summaries.groupby(['player_id', 'team']).agg(
        games_played=('game_id', 'nunique'),
        total_toi_mins=('toi_5v5_mins', 'sum'),
        season_xg_for=('xg_for_5v5', 'sum'),
        season_xg_against=('xg_against_5v5', 'sum')
    ).reset_index()

    # --- NEW: Calculate season-long average opponent Elo ---
    avg_opp_elo_df = pd.DataFrame(avg_opp_elo_log.items(), columns=['game_player_id', 'avg_opp_elo'])
    avg_opp_elo_df[['game_id', 'player_id']] = pd.DataFrame(avg_opp_elo_df['game_player_id'].tolist(), index=avg_opp_elo_df.index)
    season_avg_opp_elo = avg_opp_elo_df.groupby('player_id')['avg_opp_elo'].mean().reset_index()

    traded_player_ids = per_team_stats[per_team_stats.duplicated(subset='player_id', keep=False)]['player_id'].unique()
    single_team_stats = per_team_stats[~per_team_stats['player_id'].isin(traded_player_ids)]
    total_stats = per_team_stats[per_team_stats['player_id'].isin(traded_player_ids)].groupby('player_id').agg(
        games_played=('games_played', 'sum'),
        total_toi_mins=('total_toi_mins', 'sum'),
        season_xg_for=('season_xg_for', 'sum'),
        season_xg_against=('season_xg_against', 'sum')
    ).reset_index()
    total_stats['team'] = 'TOT'
    final_summary = pd.concat([single_team_stats, total_stats])

    player_db_to_merge = player_db.drop(columns=['games_played'], errors='ignore')
    final_output_df = pd.merge(final_summary, player_db_to_merge, on='player_id', how='left')
    final_output_df = pd.merge(final_output_df, final_ratings_df, on='player_id', how='left')
    # --- NEW: Merge in the opponent Elo data ---
    final_output_df = pd.merge(final_output_df, season_avg_opp_elo, on='player_id', how='left')
    
    final_output_df['xg_percentage'] = (final_output_df['season_xg_for'] / (final_output_df['season_xg_for'] + final_output_df['season_xg_against'])).fillna(0.5) * 100

    final_output_df.sort_values(by='elo_rating', ascending=False, inplace=True)
    final_output_df.insert(0, 'rank', range(1, 1 + len(final_output_df)))
    
    # --- NEW: Added 'avg_opp_elo' to the final output ---
    final_output_df = final_output_df[[
        'rank', 'full_name', 'position', 'team', 'elo_rating', 'avg_opp_elo', 'games_played', 'total_toi_mins',
        'xg_percentage', 'season_xg_for', 'season_xg_against', 'player_id'
    ]].round(2)
    
    FINAL_ELO_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(FINAL_ELO_OUTPUT_FILE, index=False)

    print(f"\n--- ELO MODEL COMPLETE ---")
    print(f"Final regular season skater rankings saved to: {FINAL_ELO_OUTPUT_FILE}")
    
    leaderboard_filtered = final_output_df[final_output_df['games_played'] >= 20]

    print("\nTop 20 Skaters (min. 20 GP) by Game-Weighted 5v5 xG Elo Rating:")
    print(leaderboard_filtered.head(20).to_string(index=False))

if __name__ == "__main__":
    main()