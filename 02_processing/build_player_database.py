# nhl_stats/02_processing/build_player_database.py

import pandas as pd
import requests
import time
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
# --- THIS IS THE KEY CHANGE ---
# We now read from the FAST, pre-processed directory
PROCESSED_SHIFTS_DIR = PROJECT_ROOT / "data" / "processed" / f"shift_charts_{config.SEASON_TO_PROCESS}_{config.SEASON_TO_PROCESS+1}_processed"
API_BASE_URL = "https://api-web.nhle.com/v1/player/"

def get_ids_and_games_from_local(processed_dir: Path) -> tuple[list[int], dict[int, int]]:
    """
    Scans all pre-processed shift chart CSVs to find unique player IDs and count
    the number of games they appear in.
    """
    print(f"Building player list from processed files in: {processed_dir}")

    try:
        game_files = list(processed_dir.glob('*.csv'))
    except FileNotFoundError:
        print(f"FATAL ERROR: The directory was not found: '{processed_dir}'")
        sys.exit(1)

    if not game_files:
        print(f"FATAL ERROR: No processed CSV files found in '{processed_dir}'")
        sys.exit(1)

    player_games = {} # Using a dict of sets for efficiency

    for file_path in tqdm(game_files, desc="Scanning Processed Shift Files"):
        try:
            game_id = int(file_path.stem.split('_')[0])
            # The column is now 'playerId' in the processed files from the API
            df = pd.read_csv(file_path, usecols=['playerId'])
            
            for player_id in df['playerId'].unique():
                if player_id not in player_games:
                    player_games[player_id] = set()
                player_games[player_id].add(game_id)
        except Exception as e:
            print(f"\nWarning: Could not process file {file_path.name}. Error: {e}. Skipping.")
            continue

    unique_player_ids = sorted(list(player_games.keys()))
    games_played_count = {pid: len(games) for pid, games in player_games.items()}

    print(f"Found {len(unique_player_ids)} unique players across {len(game_files)} games.")
    return unique_player_ids, games_played_count

def fetch_player_details_from_api(player_ids: list[int]) -> list[dict]:
    """Fetches detailed player information from the NHLe API."""
    print("Fetching authoritative player details from NHLe API...")
    player_data_list = []
    
    with requests.Session() as http_session:
        for player_id in tqdm(player_ids, desc="Querying NHLe API"):
            url = f"{API_BASE_URL}{player_id}/landing"
            try:
                response = http_session.get(url, timeout=10)
                if response.status_code == 404: continue # Player not in API, skip
                response.raise_for_status()
                data = response.json()

                player_info = {
                    'player_id': player_id,
                    'full_name': f"{data.get('firstName', {}).get('default', '')} {data.get('lastName', {}).get('default', '')}".strip(),
                    'position': data.get('position', 'UNK')
                }
                player_data_list.append(player_info)
            except requests.exceptions.RequestException:
                continue # Silently skip players that fail
            time.sleep(0.05)

    return player_data_list

if __name__ == "__main__":
    player_ids, games_played_map = get_ids_and_games_from_local(PROCESSED_SHIFTS_DIR)
    
    if not player_ids:
        print("FATAL ERROR: No players were found. Check processed data folder.")
        sys.exit(1)

    player_details_list = fetch_player_details_from_api(player_ids)

    if not player_details_list:
        print("FATAL ERROR: No data could be fetched from API.")
        sys.exit(1)

    print("\nCombining local and API data into final database...")
    details_df = pd.DataFrame(player_details_list)
    games_df = pd.DataFrame(games_played_map.items(), columns=['player_id', 'games_played'])

    player_database = pd.merge(details_df, games_df, on='player_id', how='left')
    player_database['games_played'] = player_database['games_played'].fillna(0).astype(int)
    
    output_path = PROJECT_ROOT / config.PLAYER_DATABASE_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_database.to_csv(output_path, index=False)

    print(f"\n--- SUCCESS! ---")
    print(f"Player database created successfully at:\n{output_path}")
    print("\nSample of the database:")
    print(player_database.head())