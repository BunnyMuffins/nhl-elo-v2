# nhl_stats/02_processing/build_game_summaries.py

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

# --- Configuration & Path Handling ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

# --- File Paths ---
SEASON_YEAR = str(config.SEASON_TO_PROCESS)
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "shift_charts" / SEASON_YEAR / "01"
MONEYPUCK_PBP_FILE = PROJECT_ROOT / config.MONEYPUCK_PBP_FILE
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
PLAYER_GAME_SUMMARIES_OUTPUT_FILE = PROJECT_ROOT / "data/processed/player_game_summaries.csv"

# --- Main Processing Function for a Single Game ---
def process_game_file(args):
    """
    Processes a single game file to generate two components:
    1. A summary of each player's total 5v5 TOI.
    2. A summary of each player's total 5v5 on-ice xG.
    """
    file_path, moneypuck_df, player_db_df = args
    try:
        game_id = int(file_path.stem.split('_')[0])
        game_shifts_df = pd.read_csv(file_path)
        game_shifts_df.rename(columns={'playerId': 'player_id', 'gameId': 'game_id', 'teamAbbrev': 'team'}, inplace=True)
    except Exception:
        return (pd.DataFrame(), pd.DataFrame())

    # Part 1: Correctly Calculate 5v5 TOI using Time-Slicing
    game_shifts_df = pd.merge(game_shifts_df, player_db_df[['player_id', 'position']], on='player_id', how='left')
    skater_shifts = game_shifts_df[game_shifts_df['position'] != 'G'].copy()
    if skater_shifts.empty: return (pd.DataFrame(), pd.DataFrame())

    all_times = pd.concat([
        skater_shifts[['period', 'start_seconds_period']],
        skater_shifts[['period', 'end_seconds_period']].rename(columns={'end_seconds_period': 'start_seconds_period'})
    ]).drop_duplicates().sort_values(['period', 'start_seconds_period']).reset_index(drop=True)
    
    home_team_abbrev = skater_shifts['team'].value_counts().idxmax()
    
    on_ice_toi_records = []
    for i in range(len(all_times) - 1):
        start_row, end_row = all_times.iloc[i], all_times.iloc[i+1]
        period, start_sec, end_sec = start_row['period'], start_row['start_seconds_period'], end_row['start_seconds_period']
        duration = end_sec - start_sec
        if period != end_row['period'] or duration <= 0: continue
            
        mid_point = start_sec + (duration / 2)
        on_ice_players = skater_shifts[(skater_shifts['period'] == period) & (skater_shifts['start_seconds_period'] <= mid_point) & (skater_shifts['end_seconds_period'] > mid_point)]
        
        if len(on_ice_players[on_ice_players['team'] == home_team_abbrev]) == 5 and len(on_ice_players[on_ice_players['team'] != home_team_abbrev]) == 5:
            for _, player in on_ice_players.iterrows():
                on_ice_toi_records.append({'game_id': game_id, 'player_id': player['player_id'], 'duration_seconds': duration})

    game_toi_summary = pd.DataFrame(on_ice_toi_records).groupby(['game_id', 'player_id'])['duration_seconds'].sum().reset_index()

    # Part 2: Correctly Calculate 5v5 xG using Event Attribution
    game_shot_events = moneypuck_df[moneypuck_df['game_id_pbp'] == game_id]
    player_game_xg_stats = defaultdict(lambda: {'xg_for': 0.0, 'xg_against': 0.0})
    if not game_shot_events.empty:
        for _, event in game_shot_events.iterrows():
            time_sec, xg, shooting_team = event['absolute_time_seconds'], event['xGoal'], event['teamCode']
            event_on_ice = game_shifts_df[(game_shifts_df['absolute_start_seconds'] < time_sec) & (game_shifts_df['absolute_end_seconds'] >= time_sec)]
            for _, player in event_on_ice.iterrows():
                player_key = (game_id, player['player_id'])
                if player['team'] == shooting_team:
                    player_game_xg_stats[player_key]['xg_for'] += xg
                else:
                    player_game_xg_stats[player_key]['xg_against'] += xg
    
    xg_records = [{'game_id': k[0], 'player_id': k[1], 'game_xg_for': v['xg_for'], 'game_xg_against': v['xg_against']} for k, v in player_game_xg_stats.items()]
    game_xg_summary = pd.DataFrame(xg_records)

    return (game_toi_summary, game_xg_summary)


def main():
    """Builds a complete player game summary file with correct xG and TOI."""
    print("--- Building Complete Player Game Summaries ---")
    
    try:
        player_db_df = pd.read_csv(PLAYER_DATABASE_FILE)
        moneypuck_df = pd.read_csv(MONEYPUCK_PBP_FILE, low_memory=False)
        # --- THIS IS THE FIX ---
        processed_files = list(SOURCE_DIR.glob('*_shifts_01.csv'))
        # --- END OF FIX ---
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required data file was not found: {e}"); sys.exit(1)

    moneypuck_df['game_id_pbp'] = pd.to_numeric(moneypuck_df['season'].astype(str) + '0' + moneypuck_df['game_id'].astype(str))
    shot_events = moneypuck_df[(moneypuck_df['homeSkatersOnIce'] == 5) & (moneypuck_df['awaySkatersOnIce'] == 5)].copy()
    shot_events = shot_events[shot_events['event'].isin(['SHOT', 'GOAL', 'MISS', 'BLOCK'])]
    shot_events['absolute_time_seconds'] = shot_events['time'].fillna(0).astype(int)

    tasks = [(f, shot_events, player_db_df) for f in processed_files]
    all_toi_summaries, all_xg_summaries = [], []

    if config.USE_MULTIPROCESSING:
        num_cores = config.NUM_CPU_CORES_TO_USE if config.NUM_CPU_CORES_TO_USE else multiprocessing.cpu_count()
        print(f"Processing {len(tasks)} games using {num_cores} CPU cores...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            for toi_summary, xg_summary in tqdm(pool.imap_unordered(process_game_file, tasks), total=len(tasks)):
                all_toi_summaries.append(toi_summary); all_xg_summaries.append(xg_summary)
    else:
        for task in tqdm(tasks):
            toi_summary, xg_summary = process_game_file(task)
            all_toi_summaries.append(toi_summary); all_xg_summaries.append(xg_summary)

    print("\nFinalizing summary...")
    full_toi_summary = pd.concat(all_toi_summaries, ignore_index=True)
    full_xg_summary = pd.concat(all_xg_summaries, ignore_index=True)

    final_summary = pd.merge(full_toi_summary, full_xg_summary, on=['game_id', 'player_id'], how='left')
    final_summary.fillna(0, inplace=True)
    final_summary['game_xg_diff'] = final_summary['game_xg_for'] - final_summary['game_xg_against']

    master_roster_files = list(SOURCE_DIR.glob('*_shifts_01.csv'))
    master_roster = pd.concat((pd.read_csv(f, usecols=['gameId', 'playerId', 'teamAbbrev']) for f in master_roster_files)).drop_duplicates()
    master_roster.rename(columns={'gameId':'game_id', 'playerId':'player_id', 'teamAbbrev':'team'}, inplace=True)
    
    final_summary = pd.merge(final_summary, master_roster, on=['game_id', 'player_id'], how='left')

    final_summary.to_csv(PLAYER_GAME_SUMMARIES_OUTPUT_FILE, index=False)
    
    print(f"\nComplete player game summary saved to:\n{PLAYER_GAME_SUMMARIES_OUTPUT_FILE}")
    print("\n--- SANITY CHECK ---")
    ror_check = final_summary[(final_summary['game_id'] == 2023020001) & (final_summary['player_id'] == 8475158)]
    sissons_check = final_summary[(final_summary['game_id'] == 2023020001) & (final_summary['player_id'] == 8476925)]
    
    print("Ryan O'Reilly's calculated 5v5 stats for Game 2023020001:")
    print(ror_check.to_string(index=False))
    print("\nColton Sissons' calculated 5v5 stats for Game 2023020001:")
    print(sissons_check.to_string(index=False))

if __name__ == "__main__":
    main()