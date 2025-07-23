# nhl_stats/03_analysis/create_player_quadrant_plot.py

import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Configuration & Path Handling ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# --- File Paths ---
FINAL_ELO_OUTPUT_FILE = PROJECT_ROOT / config.FINAL_ELO_OUTPUT_FILE

def main():
    """
    Creates a quadrant plot to visualize player performance (xG %) vs.
    the quality of their competition (Average Opponent Elo).
    """
    print("--- Creating Player Performance Quadrant Plot ---")

    try:
        ratings_df = pd.read_csv(FINAL_ELO_OUTPUT_FILE)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: The final ratings file was not found: {e}")
        sys.exit(1)

    # For this visualization, let's focus on players with significant ice time
    plot_df = ratings_df[ratings_df['games_played'] >= 40].copy()
    
    # --- Define the Quadrants ---
    # The dividing lines will be the league average for our two metrics
    avg_xg_pct = plot_df['xg_percentage'].mean()
    avg_opp_elo = plot_df['avg_opponent_elo'].mean()
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create the scatter plot
    # We can color-code by position for extra insight
    colors = {'C': 'blue', 'L': 'green', 'R': 'red', 'D': 'purple'}
    plot_df['color'] = plot_df['position'].map(colors).fillna('gray')
    ax.scatter(plot_df['avg_opponent_elo'], plot_df['xg_percentage'], c=plot_df['color'], alpha=0.6)

    # --- Add Quadrant Lines and Labels ---
    ax.axhline(y=avg_xg_pct, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=avg_opp_elo, color='k', linestyle='--', linewidth=1)

    fig.text(0.95, 0.95, 'Superstars', fontsize=14, ha='right', va='top', color='green', alpha=0.7)
    fig.text(0.05, 0.95, 'Vultures', fontsize=14, ha='left', va='top', color='orange', alpha=0.7)
    fig.text(0.05, 0.05, 'Overmatched', fontsize=14, ha='left', va='bottom', color='red', alpha=0.7)
    fig.text(0.95, 0.05, 'Survivors', fontsize=14, ha='right', va='bottom', color='blue', alpha=0.7)

    # --- Annotate Key Players for Context ---
    # Let's highlight the top 10 Elo players and some other interesting names
    players_to_annotate = plot_df.head(10)['full_name'].tolist()
    
    for player_name in players_to_annotate:
        player_data = plot_df[plot_df['full_name'] == player_name].iloc[0]
        ax.text(player_data['avg_opponent_elo'] + 0.1, 
                player_data['xg_percentage'], 
                player_name, 
                fontsize=9,
                ha='left')

    # --- Final Touches ---
    ax.set_title('5v5 Performance vs. Quality of Competition (2023-24)', fontsize=18, pad=20)
    ax.set_xlabel('Average Opponent Elo (Tougher Competition ->)', fontsize=12)
    ax.set_ylabel('On-Ice xG Percentage (Better Performance ->)', fontsize=12)
    
    # Create a legend for the position colors
    legend_patches = [mpatches.Patch(color=color, label=pos) for pos, color in colors.items()]
    ax.legend(handles=legend_patches, title='Position', loc='lower right')
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95]) # Adjust layout to make space for text
    plt.show()


if __name__ == "__main__":
    main()