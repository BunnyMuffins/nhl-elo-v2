# config.py
# This is the central configuration file for the entire NHL Elo project.

# ==============================================================================
# SCRIPT EXECUTION SETTINGS
# ==============================================================================

# Set to True to use multiprocessing for faster data processing.
# Set to False to run on a single core (slower, but easier to debug).
USE_MULTIPROCESSING = True

# If using multiprocessing, limit the number of cores used.
# Set to None to use all available cores.
NUM_CPU_CORES_TO_USE = 24


# ==============================================================================
# ELO MODEL PARAMETERS
# ==============================================================================

# The starting Elo rating for any new player.
INITIAL_ELO = 1500

# The K-factor determines how much Elo ratings change.
K_FACTOR = 20

# Home ice advantage, expressed in Elo points.
HOME_ICE_ADVANTAGE_ELO = 35

# Elo regression factor for the start of a new season.
REGRESSION_FACTOR = 0.75


# ==============================================================================
# DATA & FILE PATHS
# ==============================================================================
# The season you want to process (used to find the correct folders/files).
SEASON_TO_PROCESS = 2023

# --- Input Paths ---
SHIFTS_DATA_FOLDER = f"data/raw/shift_charts_{SEASON_TO_PROCESS}_{SEASON_TO_PROCESS+1}"
MONEYPUCK_PBP_FILE = f"data/external/shots_{SEASON_TO_PROCESS}.csv"
PROCESSED_DATA_FILE = f"data/processed/master_shifts_with_xg_{SEASON_TO_PROCESS}.csv"

# --- Output Paths ---
# This is the main output from the processing script
OUTPUT_FILE = f"data/processed/master_shifts_with_xg_{SEASON_TO_PROCESS}.csv"

# This will be the main output from the analysis script
FINAL_ELO_OUTPUT_FILE = f"data/output/player_elo_ratings_{SEASON_TO_PROCESS}.csv"

# --- THIS IS THE LINE TO ADD ---
# This will be the output from the player database build script
PLAYER_DATABASE_FILE = f"data/processed/player_database.csv"


# In the ELO MODEL PARAMETERS section

# --- Positional TOI Normalization ---
# The expected 5v5 Time on Ice for a top-tier player in a single game.
# Used to normalize the number of "games" each player plays per night.
FORWARD_NORMALIZATION_TOI = 15.0  # minutes
DEFENSEMAN_NORMALIZATION_TOI = 20.0 # minutes


# --- Per-Minute Game Elo Parameters ---
# This normalizes the xG differential PER MINUTE. A value of 0.15 means a
# player with a +0.15 xG differential per minute is considered to have had a
# "perfect" 1-minute performance (actual_score of 1.0).
GAME_XG_NORMALIZATION_FACTOR = 2.0