# nhl_stats/02_processing/build_matchup_summaries.py

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing

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
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
TEAMMATE_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / "data/processed/teammate_matchups.csv"
OPPONENT_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / "data/processed/opponent_matchups.csv"

def process_game_for_matchups(args):
    """
    Uses the robust time-slicing method to generate TOI matchup records for a single game.
    """
    file_path, player_db_df = args
    try:
        game_id = int(file_path.stem.split('_')[0])
        game_shifts_df = pd.read_csv(file_path)
        game_shifts_df.rename(columns={'playerId': 'player_id', 'teamAbbrev': 'team'}, inplace=True)
    except Exception:
        return []

    game_shifts_df = pd.merge(game_shifts_df, player_db_df[['player_id', 'position']], on='player_id', how='left')
    skater_shifts = game_shifts_df[game_shifts_df['position'] != 'G'].copy()
    if skater_shifts.empty: return []

    all_times = pd.concat([
        skater_shifts[['period', 'start_seconds_period']],
        skater_shifts[['period', 'end_seconds_period']].rename(columns={'end_seconds_period': 'start_seconds_period'})
    ]).drop_duplicates().sort_values(['period', 'start_seconds_period']).reset_index(drop=True)
    
    if skater_shifts['team'].nunique() < 2: return []
    home_team_abbrev = skater_shifts['team'].value_counts().idxmax()
    matchup_records = []

    for i in range(len(all_times) - 1):
        start_row, end_row = all_times.iloc[i], all_times.iloc[i+1]
        period, start_sec, end_sec = start_row['period'], start_row['start_seconds_period'], end_row['start_seconds_period']
        duration = end_sec - start_sec
        if period != end_row['period'] or duration <= 0: continue
            
        mid_point = start_sec + (duration / 2)
        on_ice_players = skater_shifts[(skater_shifts['period'] == period) & (skater_shifts['start_seconds_period'] <= mid_point) & (skater_shifts['end_seconds_period'] > mid_point)]
        
        home_players = on_ice_players[on_ice_players['team'] == home_team_abbrev]
        away_players = on_ice_players[on_ice_players['team'] != home_team_abbrev]
        if len(home_players) != 5 or len(away_players) != 5: continue
            
        home_player_data = home_players[['player_id', 'team']].to_records(index=False)
        away_player_data = away_players[['player_id', 'team']].to_records(index=False)
        
        for p1_id, p1_team in home_player_data:
            for p2_id, p2_team in away_player_data:
                matchup_records.append({'game_id': game_id, 'player_id': p1_id, 'team': p1_team, 'other_id': p2_id, 'duration': duration, 'type': 'opponent'})
                matchup_records.append({'game_id': game_id, 'player_id': p2_id, 'team': p2_team, 'other_id': p1_id, 'duration': duration, 'type': 'opponent'})
            for p2_id, p2_team in home_player_data:
                if p1_id < p2_id:
                    matchup_records.append({'game_id': game_id, 'player_id': p1_id, 'team': p1_team, 'other_id': p2_id, 'duration': duration, 'type': 'teammate'})
                    matchup_records.append({'game_id': game_id, 'player_id': p2_id, 'team': p2_team, 'other_id': p1_id, 'duration': duration, 'type': 'teammate'})

    return matchup_records

def main():
    """
    Main function to build the final matchup summaries using the trusted time-slicing logic.
    """
    print("--- Building Final Matchup Summaries (Time-Slicing Method) ---")
    try:
        player_db_df = pd.read_csv(PLAYER_DATABASE_FILE)
        processed_files = list(SOURCE_DIR.glob('*_01.csv'))
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required data file was not found: {e}"); sys.exit(1)

    tasks = [(f, player_db_df) for f in processed_files]
    all_matchup_records = []

    if config.USE_MULTIPROCESSING:
        num_cores = config.NUM_CPU_CORES_TO_USE if config.NUM_CPU_CORES_TO_USE else multiprocessing.cpu_count()
        print(f"Processing {len(tasks)} games for TOI matchups using {num_cores} CPU cores...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            for matchup_records in tqdm(pool.imap_unordered(process_game_for_matchups, tasks), total=len(tasks)):
                all_matchup_records.extend(matchup_records)
    else:
        for task in tqdm(tasks): all_matchup_records.extend(process_game_for_matchups(task))

    print("\nFinalizing Matchup Summaries...")
    matchups_df = pd.DataFrame(all_matchup_records)
    
    teammates_df = matchups_df[matchups_df['type'] == 'teammate']
    opponents_df = matchups_df[matchups_df['type'] == 'opponent']
    
    teammate_summary = teammates_df.groupby(['game_id', 'player_id', 'team', 'other_id'])['duration'].sum().reset_index().rename(columns={'other_id': 'teammate_id', 'duration': 'shared_toi'})
    opponent_summary = opponents_df.groupby(['game_id', 'player_id', 'team', 'other_id'])['duration'].sum().reset_index().rename(columns={'other_id': 'opponent_id', 'duration': 'shared_toi'})

    teammate_summary.to_csv(TEAMMATE_MATCHUPS_OUTPUT_FILE, index=False)
    opponent_summary.to_csv(OPPONENT_MATCHUPS_OUTPUT_FILE, index=False)
    print(f"Teammate matchup summaries saved to: {TEAMMATE_MATCHUPS_OUTPUT_FILE}")
    print(f"Opponent matchup summaries saved to: {OPPONENT_MATCHUPS_OUTPUT_FILE}")
    print(f"\nSanity check: The new teammate summary file contains data for {teammate_summary['game_id'].nunique()} games.")

if __name__ == "__main__":
    main()