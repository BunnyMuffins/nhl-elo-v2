# config.py

# ==============================================================================
# SCRIPT EXECUTION SETTINGS
# ==============================================================================
USE_MULTIPROCESSING = True
NUM_CPU_CORES_TO_USE = 24

# ==============================================================================
# ELO MODEL PARAMETERS
# ==============================================================================
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ICE_ADVANTAGE_ELO = 35
REGRESSION_FACTOR = 0.75
FORWARD_NORMALIZATION_TOI = 15.0
DEFENSEMAN_NORMALIZATION_TOI = 20.0
GAME_XG_NORMALIZATION_FACTOR = 2.0

# ==============================================================================
# DATA & FILE PATHS
# ==============================================================================
SEASON_TO_PROCESS = 2023

# --- Raw Data Input Paths ---
RAW_SHIFTS_DATA_FOLDER = f"data/raw/shift_charts_{SEASON_TO_PROCESS}_{SEASON_TO_PROCESS+1}"
MONEYPUCK_PBP_FILE = f"data/external/shots_{SEASON_TO_PROCESS}.csv"

# --- Processed Data Paths ---
# This is the new, structured path for our clean shift files
PROCESSED_SHIFTS_DIR = f"data/processed/shift_charts/{SEASON_TO_PROCESS}/01"
PLAYER_DATABASE_FILE = f"data/processed/player_database.csv"

# --- Final Summary Output Paths ---
FINAL_OUTPUT_DIR = "data/final"
PLAYER_GAME_SUMMARIES_FILE = f"{FINAL_OUTPUT_DIR}/player_game_summaries_{SEASON_TO_PROCESS}.csv"
TEAMMATE_MATCHUPS_FILE = f"{FINAL_OUTPUT_DIR}/teammate_matchups_{SEASON_TO_PROCESS}.csv"
OPPONENT_MATCHUPS_FILE = f"{FINAL_OUTPUT_DIR}/opponent_matchups_{SEASON_TO_PROCESS}.csv"


# --- Final Elo Model Output Path ---
# This is the missing variable
FINAL_ELO_OUTPUT_FILE = f"data/output/player_elo_ratings_{SEASON_TO_PROCESS}.csv"