# nhl_stats/02_processing/build_final_summaries.py

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
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / "data" / "processed" / "shift_charts" / str(config.SEASON_TO_PROCESS) / "01"
MONEYPUCK_PBP_FILE = PROJECT_ROOT / config.MONEYPUCK_PBP_FILE
PLAYER_DATABASE_FILE = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
PLAYER_GAME_SUMMARIES_OUTPUT_FILE = PROJECT_ROOT / "data/processed/player_game_summaries.csv"
TEAMMATE_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / "data/processed/teammate_matchups.csv"
OPPONENT_MATCHUPS_OUTPUT_FILE = PROJECT_ROOT / "data/processed/opponent_matchups.csv"

def process_game_file(args):
    """
    Processes a single game file to generate all three summary components:
    1. Player xG totals.
    2. Player TOI totals (which are part of the main summary).
    3. Teammate/Opponent TOI matchups.
    """
    file_path, moneypuck_df, player_db_df = args
    try:
        game_id = int(file_path.stem.split('_')[0])
        game_shifts_df = pd.read_csv(file_path)
        game_shifts_df.rename(columns={'playerId': 'player_id', 'gameId': 'game_id', 'teamAbbrev': 'team'}, inplace=True)
    except Exception:
        return (pd.DataFrame(), pd.DataFrame())

    game_shifts_df = pd.merge(game_shifts_df, player_db_df[['player_id', 'position']], on='player_id', how='left')
    skater_shifts = game_shifts_df[game_shifts_df['position'] != 'G'].copy()
    if skater_shifts.empty: return (pd.DataFrame(), pd.DataFrame())

    # --- Time-Slicing Logic ---
    all_times = pd.concat([
        skater_shifts[['period', 'start_seconds_period']],
        skater_shifts[['period', 'end_seconds_period']].rename(columns={'end_seconds_period': 'start_seconds_period'})
    ]).drop_duplicates().sort_values(['period', 'start_seconds_period']).reset_index(drop=True)
    
    home_team_abbrev = skater_shifts['team'].value_counts().idxmax()
    
    player_toi_records, matchup_records = [], []

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
        
        for _, player in on_ice_players.iterrows():
            player_toi_records.append({'game_id': game_id, 'player_id': player['player_id'], 'duration_seconds': duration})

        home_player_data = list(home_players[['player_id', 'team']].itertuples(index=False, name=None))
        away_player_data = list(away_players[['player_id', 'team']].itertuples(index=False, name=None))
        for p1_id, p1_team in home_player_data:
            for p2_id, p2_team in away_player_data: matchup_records.append({'game_id': game_id, 'player_id': p1_id, 'team': p1_team, 'other_id': p2_id, 'other_team': p2_team, 'duration': duration, 'type': 'opponent'})
            for p2_id, p2_team in home_player_data:
                if p1_id < p2_id: matchup_records.append({'game_id': game_id, 'player_id': p1_id, 'team': p1_team, 'other_id': p2_id, 'other_team': p2_team, 'duration': duration, 'type': 'teammate'})

    game_toi_summary = pd.DataFrame(player_toi_records).groupby(['game_id', 'player_id'])['duration_seconds'].sum().reset_index()

    # --- xG Attribution Logic ---
    game_shot_events = moneypuck_df[moneypuck_df['game_id_pbp'] == game_id]
    player_game_xg_stats = defaultdict(lambda: {'xg_for': 0.0, 'xg_against': 0.0})
    if not game_shot_events.empty:
        for _, event in game_shot_events.iterrows():
            time_sec, xg, shooting_team = event['absolute_time_seconds'], event['xGoal'], event['teamCode']
            event_on_ice = game_shifts_df[(game_shifts_df['absolute_start_seconds'] < time_sec) & (game_shifts_df['absolute_end_seconds'] >= time_sec)]
            for _, player in event_on_ice.iterrows():
                player_key = (game_id, player['player_id'])
                if player['team'] == shooting_team: player_game_xg_stats[player_key]['xg_for'] += xg
                else: player_game_xg_stats[player_key]['xg_against'] += xg
    
    xg_records = [{'game_id': k[0], 'player_id': k[1], 'game_xg_for': v['xg_for'], 'game_xg_against': v['xg_against']} for k, v in player_game_xg_stats.items()]
    game_xg_summary = pd.DataFrame(xg_records)
    
    # Combine TOI and xG for the main summary file
    game_summary = pd.merge(game_toi_summary, game_xg_summary, on=['game_id', 'player_id'], how='left').fillna(0)

    return (game_summary, matchup_records)

def main():
    """Builds all three final summary files in one go, with all necessary columns."""
    print("--- Building All Final Summary Files ---")
    try:
        player_db_df = pd.read_csv(PLAYER_DATABASE_FILE)
        moneypuck_df = pd.read_csv(MONEYPUCK_PBP_FILE, low_memory=False)
        processed_files = list(PROCESSED_SHIFTS_DIR.glob('*_01.csv'))
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required data file was not found: {e}"); sys.exit(1)

    moneypuck_df['game_id_pbp'] = pd.to_numeric(moneypuck_df['season'].astype(str) + '0' + moneypuck_df['game_id'].astype(str))
    shot_events = moneypuck_df[(moneypuck_df['homeSkatersOnIce'] == 5) & (moneypuck_df['awaySkatersOnIce'] == 5) & (moneypuck_df['event'].isin(['SHOT','GOAL','MISS','BLOCK']))].copy()
    shot_events['absolute_time_seconds'] = shot_events['time'].fillna(0).astype(int)

    tasks = [(f, shot_events, player_db_df) for f in processed_files]
    all_summaries, all_matchups = [], []

    if config.USE_MULTIPROCESSING:
        num_cores = config.NUM_CPU_CORES_TO_USE if config.NUM_CPU_CORES_TO_USE else multiprocessing.cpu_count()
        print(f"Processing {len(tasks)} games using {num_cores} CPU cores...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            for summary, matchups in tqdm(pool.imap_unordered(process_game_file, tasks), total=len(tasks)):
                all_summaries.append(summary); all_matchups.extend(matchups)
    else:
        for task in tqdm(tasks):
            summary, matchups = process_game_file(task)
            all_summaries.append(summary); all_matchups.extend(matchups)

    print("\nFinalizing Player Game Summaries...")
    player_summary_df = pd.concat(all_summaries, ignore_index=True)
    master_roster = pd.concat((pd.read_csv(f, usecols=['gameId', 'playerId', 'teamAbbrev']) for f in processed_files)).drop_duplicates()
    master_roster.rename(columns={'gameId':'game_id', 'playerId':'player_id', 'teamAbbrev':'team'}, inplace=True)
    
    final_player_summary = pd.merge(master_roster, player_summary_df, on=['game_id', 'player_id'], how='left').fillna(0)
    final_player_summary['game_xg_diff'] = final_player_summary['game_xg_for'] - final_player_summary['game_xg_against']
    final_player_summary.to_csv(PLAYER_GAME_SUMMARIES_OUTPUT_FILE, index=False)
    print(f"Player game summaries saved to: {PLAYER_GAME_SUMMARIES_OUTPUT_FILE}")

    print("Finalizing Matchup Summaries...")
    matchups_df = pd.DataFrame(all_matchups)
    opponents_df = matchups_df[matchups_df['type'] == 'opponent']
    reciprocal_opps = opponents_df.rename(columns={'player_id': 'other_id', 'other_team': 'team', 'other_id': 'player_id', 'team': 'other_team'})
    full_opponents = pd.concat([opponents_df, reciprocal_opps])
    opponent_summary = full_opponents.groupby(['game_id', 'player_id', 'team', 'other_id', 'other_team'])['duration'].sum().reset_index().rename(columns={'other_id': 'opponent_id', 'other_team': 'opponent_team', 'duration': 'shared_toi'})

    teammates_df = matchups_df[matchups_df['type'] == 'teammate']
    reciprocal_teammates = teammates_df.rename(columns={'player_id': 'other_id', 'other_id': 'player_id'})
    full_teammates = pd.concat([teammates_df, reciprocal_teammates])
    teammate_summary = full_teammates.groupby(['game_id', 'player_id', 'team', 'other_id'])['duration'].sum().reset_index().rename(columns={'other_id': 'teammate_id', 'duration': 'shared_toi'})
    
    teammate_summary.to_csv(TEAMMATE_MATCHUPS_OUTPUT_FILE, index=False)
    opponent_summary.to_csv(OPPONENT_MATCHUPS_OUTPUT_FILE, index=False)
    print(f"Teammate/Opponent summaries saved.")
    print(f"\nSanity check: Game summary file contains data for {final_player_summary['game_id'].nunique()} games.")

if __name__ == "__main__":
    main()