# nhl_stats/03_analysis/plot_player_elo_history.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration & Path Handling ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

# --- !!! CHOOSE THE PLAYER TO TRACK HERE !!! ---
PLAYER_NAME_TO_TRACK = "Connor McDavid"
# Other great examples: "Evan Bouchard", "Nathan MacKinnon", "Adam Lowry"

# --- File Paths ---
PLAYER_GAME_SUMMARIES_FILE = PROJECT_ROOT / "data/processed/player_game_summaries.csv"
TEAMMATE_MATCHUPS_FILE = PROJECT_ROOT / "data/processed/teammate_matchups.csv"
OPPONENT_MATCHUPS_FILE = PROJECT_ROOT / "data/processed/opponent_matchups.csv"
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE

# --- Elo Calculation Functions (identical to the main script) ---
def calculate_expected_score(player_elo, avg_teammate_elo, avg_opponent_elo, is_home):
    home_ice_advantage = config.HOME_ICE_ADVANTAGE_ELO if is_home else 0
    player_unit_strength = (0.2 * player_elo) + (0.8 * avg_teammate_elo)
    elo_diff = player_unit_strength - avg_opponent_elo + home_ice_advantage
    return 1 / (1 + 10**(-elo_diff / 400))

def update_elo(k, actual_score, expected_score):
    game_k = k * 2.5 
    return game_k * (actual_score - expected_score)

def main():
    """
    Runs the Elo model and plots the rating history for a specific player
    over the course of the season.
    """
    print(f"--- Plotting Elo History for: {PLAYER_NAME_TO_TRACK} ---")

    try:
        player_summaries = pd.read_csv(PLAYER_GAME_SUMMARIES_FILE)
        teammate_matchups = pd.read_csv(TEAMMATE_MATCHUPS_FILE)
        opponent_matchups = pd.read_csv(OPPONENT_MATCHUPS_FILE)
        player_db = pd.read_csv(PLAYER_DATABASE_FILE)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required summary file was not found: {e}")
        sys.exit(1)

    # Find the player ID for the name we're tracking
    try:
        target_player_id = player_db[player_db['full_name'] == PLAYER_NAME_TO_TRACK]['player_id'].iloc[0]
    except IndexError:
        print(f"FATAL ERROR: Could not find player named '{PLAYER_NAME_TO_TRACK}' in the database.")
        sys.exit(1)

    # --- Prepare Data ---
    player_summaries = pd.merge(player_summaries, player_db[['player_id', 'position']], on='player_id', how='left')
    skater_summaries = player_summaries[player_summaries['position'] != 'G'].copy()
    
    all_skaters = skater_summaries['player_id'].unique()
    elo_ratings = {player_id: config.INITIAL_ELO for player_id in all_skaters}

    # This will store our player's history
    elo_history = [{'game_num': 0, 'elo': config.INITIAL_ELO}]
    game_counter = 0

    game_groups = skater_summaries.groupby('game_id')
    
    print("\nCalculating Elo ratings and tracking player history...")
    for game_id, game_df in tqdm(game_groups, total=len(game_groups)):
        elo_changes = {}
        player_was_in_game = target_player_id in game_df['player_id'].values

        if player_was_in_game:
            game_counter += 1

        game_teammate_matchups = teammate_matchups[teammate_matchups['game_id'] == game_id]
        game_opponent_matchups = opponent_matchups[opponent_matchups['game_id'] == game_id]

        for _, player_row in game_df.iterrows():
            player_id, is_home = player_row['player_id'], player_row['team'] == 'home'
            
            teammates = game_teammate_matchups[game_teammate_matchups['player_id'] == player_id]
            opponents = game_opponent_matchups[game_opponent_matchups['player_id'] == player_id]
            if teammates.empty or opponents.empty: continue

            avg_teammate_elo = np.average(teammates['teammate_id'].map(elo_ratings), weights=teammates['shared_toi'])
            avg_opponent_elo = np.average(opponents['opponent_id'].map(elo_ratings), weights=opponents['shared_toi'])
            
            expected_score = calculate_expected_score(elo_ratings[player_id], avg_teammate_elo, avg_opponent_elo, is_home)
            normalized_diff = player_row['game_xg_diff'] / config.GAME_XG_NORMALIZATION_FACTOR
            actual_score = np.clip(0.5 * normalized_diff + 0.5, 0, 1)
            elo_changes[player_id] = update_elo(config.K_FACTOR, actual_score, expected_score)

        for pid, change in elo_changes.items():
            elo_ratings[pid] += change
            
        # If our target player was in this game, record their new rating
        if player_was_in_game:
            elo_history.append({'game_num': game_counter, 'elo': elo_ratings[target_player_id]})
    
    # --- Plotting ---
    print("\nGenerating plot...")
    history_df = pd.DataFrame(elo_history)
    
    plt.style.use('seaborn-v0_8-grid')
    plt.figure(figsize=(12, 7))
    plt.plot(history_df['game_num'], history_df['elo'], marker='o', linestyle='-', markersize=4)
    
    plt.title(f"Elo Rating History for {PLAYER_NAME_TO_TRACK} (2023-24 Season)", fontsize=16)
    plt.xlabel("Game Number", fontsize=12)
    plt.ylabel("5v5 xG Elo Rating", fontsize=12)
    plt.axhline(y=config.INITIAL_ELO, color='r', linestyle='--', label=f'Starting Elo ({config.INITIAL_ELO})')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()