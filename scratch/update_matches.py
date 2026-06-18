import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path to import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import normalize_team_name

def main():
    data_dir = PROJECT_ROOT / 'data'
    
    # Load 2026 actual results
    wc26_matches = pd.read_csv(data_dir / '2026_world_cup_matches.csv')
    print(f"Loaded {len(wc26_matches)} actual matches from 2026_world_cup_matches.csv")
    
    # Load international matches
    intl_matches = pd.read_csv(data_dir / 'international_matches.csv')
    print(f"Loaded {len(intl_matches)} matches from international_matches.csv")
    
    updated_count = 0
    
    for idx, row in wc26_matches.iterrows():
        date = row['Date']
        h_team = normalize_team_name(row['Home Team'])
        a_team = normalize_team_name(row['Away Team'])
        h_goals = row['Home Score']
        a_goals = row['Away Score']
        
        if pd.isna(h_goals) or pd.isna(a_goals):
            continue
            
        # Find match in international_matches.csv
        # We need to normalize team names in intl_matches to check for a match
        # To make it fast, we filter by Date first
        mask_date = intl_matches['Date'] == date
        matches_on_date = intl_matches[mask_date]
        
        found = False
        for intl_idx, intl_row in matches_on_date.iterrows():
            intl_h = normalize_team_name(intl_row['Home Team'])
            intl_a = normalize_team_name(intl_row['Away Team'])
            
            # Check for match (either home/away or away/home)
            if (intl_h == h_team and intl_a == a_team):
                intl_matches.at[intl_idx, 'Home Goals'] = float(h_goals)
                intl_matches.at[intl_idx, 'Away Goals'] = float(a_goals)
                found = True
                updated_count += 1
                print(f"Updated: {intl_row['Home Team']} {int(h_goals)} - {int(a_goals)} {intl_row['Away Team']} ({date})")
                break
            elif (intl_h == a_team and intl_a == h_team):
                intl_matches.at[intl_idx, 'Home Goals'] = float(a_goals)
                intl_matches.at[intl_idx, 'Away Goals'] = float(h_goals)
                found = True
                updated_count += 1
                print(f"Updated (reversed): {intl_row['Home Team']} {int(a_goals)} - {int(h_goals)} {intl_row['Away Team']} ({date})")
                break
                
        if not found:
            print(f"Warning: Match not found on {date} between {row['Home Team']} and {row['Away Team']}")
            
    # Save back to disk
    intl_matches.to_csv(data_dir / 'international_matches.csv', index=False)
    print(f"Successfully updated {updated_count} matches in international_matches.csv")

if __name__ == '__main__':
    main()
