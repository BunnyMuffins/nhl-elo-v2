# nhl_stats/01_scraping/fetch_season_data.py

import pandas as pd
import requests
import sys
import time
from pathlib import Path
from tqdm import tqdm
from datetime import date, timedelta

# --- Configuration & Path Handling ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# --- API Endpoints ---
# Use the web API to get a reliable list of game IDs
SCHEDULE_API_BASE_URL = "https://api-web.nhle.com/v1/schedule/"
# Use the older, stable stats API for the actual shift data (from Zmalski docs)
SHIFTS_API_BASE_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId="

# --- File Paths ---
# We will save to a clean, new raw directory.
RAW_SHIFTS_DIR = PROJECT_ROOT / "data" / "raw" / f"shift_charts_{config.SEASON_TO_PROCESS}_{config.SEASON_TO_PROCESS+1}"


def get_all_game_ids_for_season(year: int) -> list[int]:
    """
    Gets a complete list of all regular season and playoff game IDs for a given season.
    """
    print(f"Fetching all game IDs for the {year}-{year+1} season...")
    start_date = date(year, 10, 1)  # Season starts in October
    end_date = date(year + 1, 7, 1) # Playoffs end by July
    
    all_game_ids = set()
    session = requests.Session()
    
    current_date = start_date
    with tqdm(total=(end_date - start_date).days, desc="Scanning Season Schedule") as pbar:
        while current_date < end_date:
            url = f"{SCHEDULE_API_BASE_URL}{current_date.strftime('%Y-%m-%d')}"
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 404: # No games on this day
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue

                response.raise_for_status()
                schedule_data = response.json()
                
                for day in schedule_data.get('gameWeek', []):
                    for game in day.get('games', []):
                        # We only want regular season (2) and playoffs (3)
                        if game.get('gameType') in [2, 3]:
                            all_game_ids.add(game['id'])
            except requests.RequestException:
                pass # Ignore days with no schedule or failed requests
            
            current_date += timedelta(days=1)
            pbar.update(1)

    print(f"Found {len(all_game_ids)} total regular season & playoff games.")
    return sorted(list(all_game_ids))

def get_already_downloaded_ids(raw_dir: Path) -> set[int]:
    """Scans the raw data directory to see which game files already exist."""
    if not raw_dir.exists():
        return set()
    return {int(f.stem.split('_')[0]) for f in raw_dir.glob('*_shifts.csv')}

def download_and_save_shifts(game_id: int, raw_dir: Path, session: requests.Session):
    """Downloads shift data for a single game and saves it."""
    try:
        response = session.get(f"{SHIFTS_API_BASE_URL}{game_id}", timeout=20)
        response.raise_for_status()
        data = response.json().get('data')
        
        if data:
            df = pd.DataFrame(data)
            output_path = raw_dir / f"{game_id}_shifts.csv"
            df.to_csv(output_path, index=False)
            return True
    except requests.exceptions.RequestException:
        # Silently ignore 404s for games that might not have data
        return False
    return False

def main():
    """
    Downloads a complete, clean set of shift charts for the specified season.
    """
    print("--- Starting Fresh Download of All Shift Charts ---")
    
    # Ensure the target directory exists and is clean
    RAW_SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"All new raw data will be saved in:\n{RAW_SHIFTS_DIR}\n")

    # 1. Get a complete list of games that should exist.
    all_season_ids = get_all_game_ids_for_season(config.SEASON_TO_PROCESS)

    # 2. See which games we already have.
    downloaded_ids = get_already_downloaded_ids(RAW_SHIFTS_DIR)
    print(f"Found {len(downloaded_ids)} games already downloaded.")

    # 3. Determine which games are missing and need to be downloaded.
    missing_ids = [gid for gid in all_season_ids if gid not in downloaded_ids]

    if not missing_ids:
        print("\nSUCCESS: Your dataset is already complete! No new games to download.")
    else:
        print(f"\nFound {len(missing_ids)} missing games. Starting download...")
        with requests.Session() as http_session:
            for game_id in tqdm(missing_ids, desc="Downloading Missing Games"):
                download_and_save_shifts(game_id, RAW_SHIFTS_DIR, http_session)
                time.sleep(0.1) # Be a good citizen to the API
        print("\n--- Download Complete! ---")

    # 4. Final verification
    print("\nVerifying final file count...")
    final_ids = get_already_downloaded_ids(RAW_SHIFTS_DIR)
    print(f"Total files in directory: {len(final_ids)}")
    if len(final_ids) == len(all_season_ids):
        print("Verification successful. All game files are present.")
    else:
        # It's common for a few playoff games to not have data, so this is a warning.
        print(f"Warning: Expected {len(all_season_ids)} games, but found {len(final_ids)}. This may be normal.")

if __name__ == "__main__":
    main()