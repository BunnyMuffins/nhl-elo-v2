# nhl_stats/02_processing/01_process_raw_shifts.py

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
# Use the new, clean raw directory from our successful download
RAW_SHIFTS_DIR = PROJECT_ROOT / "data" / "raw" / f"shift_charts_{config.SEASON_TO_PROCESS}_{config.SEASON_TO_PROCESS+1}"
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / "data" / "processed" / f"shift_charts_{config.SEASON_TO_PROCESS}_{config.SEASON_TO_PROCESS+1}_processed"


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
    converts times to seconds, and saves them to a new processed directory.
    This is the definitive foundation for the project.
    """
    print("--- Starting Foundational Step: Processing All Raw Shift Charts ---")
    
    PROCESSED_SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processed files will be saved in:\n{PROCESSED_SHIFTS_DIR}\n")

    raw_files_to_process = list(RAW_SHIFTS_DIR.glob('*.csv'))

    if not raw_files_to_process:
        print(f"FATAL ERROR: No raw shift files found in {RAW_SHIFTS_DIR}.")
        print("Please ensure the 'fetch_season_data.py' script has been run successfully.")
        sys.exit(1)

    for raw_file_path in tqdm(raw_files_to_process, desc="Processing Raw Shift Files"):
        try:
            game_df = pd.read_csv(raw_file_path)

            # --- THIS IS THE CRITICAL FIX BASED ON YOUR DIAGNOSTIC ---
            # We only want regular on-ice shifts, which have the real duration.
            regular_shifts_df = game_df[game_df['typeCode'] == 517].copy()
            # --- END OF FIX ---

            if regular_shifts_df.empty:
                continue

            # Perform all time conversions on this clean data
            regular_shifts_df['playerId'] = regular_shifts_df['playerId'].astype(int)
            regular_shifts_df['period'] = regular_shifts_df['period'].astype(int)
            
            # Use the reliable 'duration' column from the 517 shifts
            regular_shifts_df['duration_seconds'] = regular_shifts_df['duration'].apply(parse_time_to_seconds)
            regular_shifts_df['start_seconds_period'] = regular_shifts_df['startTime'].apply(parse_time_to_seconds)
            regular_shifts_df['end_seconds_period'] = regular_shifts_df['endTime'].apply(parse_time_to_seconds)
            
            regular_shifts_df.dropna(subset=['duration_seconds', 'start_seconds_period', 'end_seconds_period'], inplace=True)

            regular_shifts_df['absolute_start_seconds'] = (regular_shifts_df['period'] - 1) * 1200 + regular_shifts_df['start_seconds_period']
            regular_shifts_df['absolute_end_seconds'] = (regular_shifts_df['period'] - 1) * 1200 + regular_shifts_df['end_seconds_period']
            
            # Define the columns for our new, clean files
            final_columns = [
                'gameId', 'shiftNumber', 'period', 'playerId', 'teamAbbrev', 'firstName', 'lastName',
                'startTime', 'endTime', 'duration', 'start_seconds_period', 'end_seconds_period',
                'absolute_start_seconds', 'absolute_end_seconds', 'duration_seconds'
            ]
            final_columns = [col for col in final_columns if col in regular_shifts_df.columns]
            processed_df = regular_shifts_df[final_columns]
            
            output_path = PROCESSED_SHIFTS_DIR / raw_file_path.name
            processed_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Warning: Could not process file {raw_file_path.name}. Error: {e}")
            continue

    print("\n--- SUCCESS! ---")
    print("All raw shift charts have been processed using the correct logic.")
    print("The new processed files contain only valid shifts with reliable durations.")

if __name__ == "__main__":
    main()