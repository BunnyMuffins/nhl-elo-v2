# nhl_stats/02_processing/build_final_summaries.py

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
from itertools import combinations, product
import numpy as np

# --- Configuration & Path Handling ---
PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

# --- File Paths from Config ---
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / config.PROCESSED_SHIFTS_DIR
MONEYPUCK_PBP_FILE = PROJECT_ROOT / config.MONEYPUCK_PBP_FILE
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
FINAL_OUTPUT_DIR = PROJECT_ROOT / config.FINAL_OUTPUT_DIR

PLAYER_GAME_SUMMARIES_OUTPUT_FILE = PROJECT_ROOT / config.PLAYER_GAME_SUMMARIES_FILE
TEAMMATE_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / config.TEAMMATE_MATCHUPS_FILE
OPPONENT_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / config.OPPONENT_MATCHUPS_FILE

def process_game_file(args):
    file_path, moneypuck_df, player_db_df = args
    try:
        game_id = int(file_path.stem)
        game_shifts_df = pd.read_csv(file_path)
        if game_shifts_df.empty: return None
        game_roster = game_shifts_df[['player_id', 'team']].drop_duplicates()
        game_shifts_df = pd.merge(game_shifts_df, player_db_df[['player_id', 'position']], on='player_id', how='left')
        teams = game_shifts_df['team'].unique()
        if len(teams) < 2: return None
        home_team, away_team = teams[0], teams[1]
    except Exception: return None

    skater_shifts = game_shifts_df[game_shifts_df['position'] != 'G'].copy()
    if skater_shifts.empty: return None
    time_points = np.unique(np.concatenate([skater_shifts['absolute_start_seconds'].values, skater_shifts['absolute_end_seconds'].values]))
    player_toi, teammate_toi, opponent_toi = defaultdict(float), defaultdict(float), defaultdict(float)
    for i in range(len(time_points) - 1):
        start_sec, end_sec = time_points[i], time_points[i+1]
        duration = end_sec - start_sec
        if duration < 0.01: continue
        mid_point = start_sec + (duration / 2)
        on_ice_mask = (skater_shifts['absolute_start_seconds'] <= mid_point) & (skater_shifts['absolute_end_seconds'] > mid_point)
        on_ice_players = skater_shifts[on_ice_mask]
        home_on_ice = on_ice_players[on_ice_players['team'] == home_team]
        away_on_ice = on_ice_players[on_ice_players['team'] == away_team]
        if len(home_on_ice) == 5 and len(away_on_ice) == 5:
            home_ids, away_ids = sorted(home_on_ice['player_id'].tolist()), sorted(away_on_ice['player_id'].tolist())
            for pid in home_ids + away_ids: player_toi[pid] += duration
            for p1, p2 in combinations(home_ids, 2): teammate_toi[(p1, p2)] += duration
            for p1, p2 in combinations(away_ids, 2): teammate_toi[(p1, p2)] += duration
            for p1, p2 in product(home_ids, away_ids): opponent_toi[tuple(sorted((p1, p2)))] += duration
    
    game_shot_events = moneypuck_df[moneypuck_df['game_id'] == game_id]
    player_xg = defaultdict(lambda: {'xg_for': 0.0, 'xg_against': 0.0})
    if not game_shot_events.empty:
        for _, event in game_shot_events.iterrows():
            time_sec, xg, shooting_team = event['absolute_time_seconds'], event['xGoal'], event['teamCode']
            on_ice_mask = (game_shifts_df['absolute_start_seconds'] < time_sec) & (game_shifts_df['absolute_end_seconds'] >= time_sec)
            event_on_ice_players = game_shifts_df[on_ice_mask]
            for _, player in event_on_ice_players.iterrows():
                if player['team'] == shooting_team: player_xg[player['player_id']]['xg_for'] += xg
                else: player_xg[player['player_id']]['xg_against'] += xg
    
    player_summaries = [{'game_id': game_id, 'player_id': pid, 'team': team, 'toi_5v5': player_toi[pid], 'xg_for_5v5': player_xg[pid]['xg_for'], 'xg_against_5v5': player_xg[pid]['xg_against']} for pid, team in game_roster.values]
    teammate_matchups = [{'game_id': game_id, 'player1_id': p1, 'player2_id': p2, 'toi_5v5': toi} for (p1, p2), toi in teammate_toi.items()]
    opponent_matchups = [{'game_id': game_id, 'player1_id': p1, 'player2_id': p2, 'toi_5v5': toi} for (p1, p2), toi in opponent_toi.items()]
    return (player_summaries, teammate_matchups, opponent_matchups)

def main():
    print("--- Building All Final Summary Files (Corrected Teammate Columns) ---")
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        player_db_df = pd.read_csv(PLAYER_DATABASE_FILE)
        moneypuck_df = pd.read_csv(MONEYPUCK_PBP_FILE, low_memory=False)
        processed_files = list(PROCESSED_SHIFTS_DIR.glob('*.csv'))
        if not processed_files: print(f"FATAL ERROR: No files in {PROCESSED_SHIFTS_DIR}."); sys.exit(1)
    except FileNotFoundError as e: print(f"FATAL ERROR: File not found: {e}"); sys.exit(1)
    
    moneypuck_df['game_id'] = (config.SEASON_TO_PROCESS * 1_000_000) + moneypuck_df['game_id']
    shot_events = moneypuck_df[(moneypuck_df['homeSkatersOnIce'] == 5) & (moneypuck_df['awaySkatersOnIce'] == 5) & (moneypuck_df['event'].isin(['SHOT','GOAL','MISS','BLOCK']))].copy()
    shot_events['absolute_time_seconds'] = shot_events['time'].fillna(0).astype(int)
    
    tasks = [(f, shot_events, player_db_df) for f in processed_files]
    all_player_summaries, all_teammate_matchups, all_opponent_matchups = [], [], []
    pool_func = process_game_file
    if config.USE_MULTIPROCESSING:
        num_cores = config.NUM_CPU_CORES_TO_USE or multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(pool_func, tasks), total=len(tasks), desc="Processing Games"))
    else:
        results = [pool_func(task) for task in tqdm(tasks, desc="Processing Games")]
    for res in filter(None, results):
        all_player_summaries.extend(res[0]); all_teammate_matchups.extend(res[1]); all_opponent_matchups.extend(res[2])
    
    print("\nFinalizing and saving summary files...")
    player_summary_df = pd.DataFrame(all_player_summaries)
    player_summary_df.to_csv(PLAYER_GAME_SUMMARIES_OUTPUT_FILE, index=False)
    print(f"✔ Player game summaries saved to: {PLAYER_GAME_SUMMARIES_OUTPUT_FILE}")
    
    # --- CORRECTED Teammate Matchups Finalization ---
    teammate_matchups_df = pd.DataFrame(all_teammate_matchups)
    player_team_map = player_summary_df[['player_id', 'team']].drop_duplicates()
    # First, rename columns to their final names
    temp_teammates = teammate_matchups_df.rename(columns={
        'player1_id': 'player_id', 'player2_id': 'teammate_id', 'toi_5v5': 'shared_toi'
    })
    # Now, merge using the clean 'player_id' key
    final_teammates = pd.merge(temp_teammates, player_team_map, on='player_id', how='left')
    # Select and order the final columns
    final_teammates = final_teammates[['game_id', 'player_id', 'team', 'teammate_id', 'shared_toi']]
    final_teammates.to_csv(TEAMMATE_MATCHUPS_OUTPUT_FILE, index=False)
    print(f"✔ Teammate matchups saved to: {TEAMMATE_MATCHUPS_OUTPUT_FILE}")

    # --- Opponent Matchups Finalization (already correct) ---
    opponent_matchups_df = pd.DataFrame(all_opponent_matchups)
    final_opponents = pd.merge(opponent_matchups_df, player_team_map, left_on='player1_id', right_on='player_id', how='left')
    final_opponents = pd.merge(final_opponents, player_team_map, left_on='player2_id', right_on='player_id', how='left', suffixes=('_p1', '_p2'))
    final_opponents = final_opponents.rename(columns={
        'player1_id': 'player_id', 'team_p1': 'team',
        'player2_id': 'opponent_id', 'team_p2': 'opponent_team',
        'toi_5v5': 'shared_toi'
    })
    final_opponents = final_opponents[['game_id', 'player_id', 'team', 'opponent_id', 'opponent_team', 'shared_toi']]
    final_opponents.to_csv(OPPONENT_MATCHUPS_OUTPUT_FILE, index=False)
    print(f"✔ Opponent matchups saved to: {OPPONENT_MATCHUPS_OUTPUT_FILE}")
    
    print(f"\n--- SUCCESS! ---")

if __name__ == "__main__":
    main()