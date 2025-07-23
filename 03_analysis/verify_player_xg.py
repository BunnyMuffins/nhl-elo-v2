# nhl_stats/03_analysis/verify_player_xg.py

import pandas as pd
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
MONEYPUCK_PBP_FILE = PROJECT_ROOT / config.MONEYPUCK_PBP_FILE
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / "data" / "processed" / f"shift_charts_{config.SEASON_TO_PROCESS}_{config.SEASON_TO_PROCESS+1}_processed"

def main(game_id_to_inspect: int, player_id_to_inspect: int):
    """
    A reusable tool to verify the on-ice xG calculation for any player in any game.
    """
    print(f"--- Verifying xG for Player {player_id_to_inspect} in Game {game_id_to_inspect} ---")

    # 1. Load the specific, necessary source files
    try:
        pbp_df = pd.read_csv(MONEYPUCK_PBP_FILE, low_memory=False)
        shift_file_path = PROCESSED_SHIFTS_DIR / f"{game_id_to_inspect}_shifts.csv"
        game_shifts_df = pd.read_csv(shift_file_path)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find processed shift file for game {game_id_to_inspect}.")
        sys.exit(1)

    # 2. Prepare the data
    pbp_df['game_id_pbp'] = pd.to_numeric(pbp_df['season'].astype(str) + '0' + pbp_df['game_id'].astype(str))
    game_pbp = pbp_df[pbp_df['game_id_pbp'] == game_id_to_inspect].copy()
    
    player_shifts = game_shifts_df[game_shifts_df['playerId'] == player_id_to_inspect].copy()
    if player_shifts.empty:
        print(f"\nPlayer {player_id_to_inspect} did not have any shifts recorded in this game.")
        return
        
    player_team_abbrev = player_shifts['teamAbbrev'].iloc[0]

    shot_events = game_pbp[
        (game_pbp['homeSkatersOnIce'] == 5) & (game_pbp['awaySkatersOnIce'] == 5) &
        (game_pbp['event'].isin(['SHOT', 'GOAL', 'MISS', 'BLOCK']))
    ].copy()
    shot_events['absolute_time_seconds'] = shot_events['time'].fillna(0).astype(int)
    
    # 3. Create the detailed event log
    event_log = []
    for _, event in shot_events.iterrows():
        time_sec = event['absolute_time_seconds']
        
        on_ice_check = not player_shifts[
            (player_shifts['absolute_start_seconds'] < time_sec) &
            (player_shifts['absolute_end_seconds'] >= time_sec)
        ].empty

        if on_ice_check:
            shooting_team = event['teamCode']
            xg_for = event['xGoal'] if shooting_team == player_team_abbrev else 0.0
            xg_against = event['xGoal'] if shooting_team != player_team_abbrev else 0.0
            event_log.append({'time': time_sec, 'shooter': event['shooterName'], 'xGoal': event['xGoal'], 'xg_for': xg_for, 'xg_against': xg_against})

    # 4. Print the "Receipt"
    log_df = pd.DataFrame(event_log)
    print(f"\n--- Found {len(log_df)} On-Ice 5v5 Shot Events for {player_shifts['firstName'].iloc[0]} {player_shifts['lastName'].iloc[0]} ---")
    if not log_df.empty:
        print(log_df.to_string(index=False))

    # 5. Print the final summary
    final_xg_for = log_df['xg_for'].sum()
    final_xg_against = log_df['xg_against'].sum()
    
    print("\n--- FINAL SUMMARY ---")
    print(f"Calculated 5v5 stats:")
    print(f"  On-Ice xG For:      {final_xg_for:.4f}")
    print(f"  On-Ice xG Against:  {final_xg_against:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_player_xg.py <game_id> <player_id>")
        # Provide a default example for ease of use
        print("\nRunning default example: Cale Makar in Game 2023020005")
        main(game_id_to_inspect=2023020005, player_id_to_inspect=8480069)
    else:
        try:
            game_id = int(sys.argv[1])
            player_id = int(sys.argv[2])
            main(game_id_to_inspect=game_id, player_id_to_inspect=player_id)
        except ValueError:
            print("Error: Game ID and Player ID must be integers.")