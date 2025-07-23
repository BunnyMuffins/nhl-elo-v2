# nhl_stats/02_processing/01_process_raw_shifts.py

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# --- Configuration & Path Handling ---
# CORRECTED: The Project Root is the current working directory,
# since the script is run from 'nhl_stats'.
PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# --- File Paths from Config ---
RAW_SHIFTS_DIR = PROJECT_ROOT / config.RAW_SHIFTS_DATA_FOLDER
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / config.PROCESSED_SHIFTS_DIR

# --- Helper Function ---
def parse_time_to_seconds(time_str):
    try:
        if isinstance(time_str, str):
            m, s = map(int, time_str.split(':'))
            return m * 60 + s
    except (ValueError, AttributeError):
        return None

def main():
    """
    Reads all raw shift charts, filters for ONLY regular shifts (typeCode 517),
    converts times to seconds, and saves them to the new structured processed directory.
    """
    print("--- Starting Foundational Step: Processing All Raw Shift Charts ---")
    
    PROCESSED_SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processed files will be saved in:\n{PROCESSED_SHIFTS_DIR}\n")

    raw_files_to_process = list(RAW_SHIFTS_DIR.glob('*.csv'))

    if not raw_files_to_process:
        print(f"FATAL ERROR: No raw shift files found in {RAW_SHIFTS_DIR}.")
        sys.exit(1)

    for raw_file_path in tqdm(raw_files_to_process, desc="Processing Raw Shift Files"):
        try:
            game_id_str = raw_file_path.stem.split('_')[0]
            game_df = pd.read_csv(raw_file_path)
            
            regular_shifts_df = game_df[game_df['typeCode'] == 517].copy()
            if regular_shifts_df.empty:
                continue

            regular_shifts_df.rename(columns={
                'gameId': 'game_id', 'playerId': 'player_id', 'teamAbbrev': 'team'
            }, inplace=True)

            regular_shifts_df['duration_seconds'] = regular_shifts_df['duration'].apply(parse_time_to_seconds)
            regular_shifts_df['start_seconds_period'] = regular_shifts_df['startTime'].apply(parse_time_to_seconds)
            
            regular_shifts_df.dropna(subset=['duration_seconds', 'start_seconds_period'], inplace=True)
            
            regular_shifts_df['end_seconds_period'] = regular_shifts_df['start_seconds_period'] + regular_shifts_df['duration_seconds']

            regular_shifts_df['absolute_start_seconds'] = (regular_shifts_df['period'] - 1) * 1200 + regular_shifts_df['start_seconds_period']
            regular_shifts_df['absolute_end_seconds'] = regular_shifts_df['absolute_start_seconds'] + regular_shifts_df['duration_seconds']
            
            final_columns = [
                'game_id', 'player_id', 'team', 'firstName', 'lastName', 'period',
                'absolute_start_seconds', 'absolute_end_seconds', 'duration_seconds'
            ]
            processed_df = regular_shifts_df[[col for col in final_columns if col in regular_shifts_df.columns]].copy()
            
            output_path = PROCESSED_SHIFTS_DIR / f"{game_id_str}.csv"
            processed_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Warning: Could not process file {raw_file_path.name}. Error: {e}")
            continue

    print("\n--- SUCCESS! ---")
    print(f"All raw shift charts have been processed and saved to {PROCESSED_SHIFTS_DIR}")

if __name__ == "__main__":
    main()