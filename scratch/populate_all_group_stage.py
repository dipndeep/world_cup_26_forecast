import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path to import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import normalize_team_name

# Complete list of 72 group stage matches
matches_data = [
    # Matchday 1
    {"ID": 1, "Year": 2026, "Date": "2026-06-11", "Stage": "Group stage", "Home Team": "Mexico", "Away Team": "South Africa", "Host Team": True, "Home Score": 2, "Away Score": 0},
    {"ID": 2, "Year": 2026, "Date": "2026-06-11", "Stage": "Group stage", "Home Team": "Korea Republic", "Away Team": "Czechia", "Host Team": False, "Home Score": 2, "Away Score": 1},
    {"ID": 3, "Year": 2026, "Date": "2026-06-12", "Stage": "Group stage", "Home Team": "Canada", "Away Team": "Bosnia and Herzegovina", "Host Team": True, "Home Score": 1, "Away Score": 1},
    {"ID": 4, "Year": 2026, "Date": "2026-06-12", "Stage": "Group stage", "Home Team": "USA", "Away Team": "Paraguay", "Host Team": True, "Home Score": 4, "Away Score": 1},
    {"ID": 5, "Year": 2026, "Date": "2026-06-13", "Stage": "Group stage", "Home Team": "Qatar", "Away Team": "Switzerland", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 6, "Year": 2026, "Date": "2026-06-13", "Stage": "Group stage", "Home Team": "Brazil", "Away Team": "Morocco", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 7, "Year": 2026, "Date": "2026-06-13", "Stage": "Group stage", "Home Team": "Scotland", "Away Team": "Haiti", "Host Team": False, "Home Score": 1, "Away Score": 0},
    {"ID": 8, "Year": 2026, "Date": "2026-06-13", "Stage": "Group stage", "Home Team": "Australia", "Away Team": "Türkiye", "Host Team": False, "Home Score": 2, "Away Score": 0},
    {"ID": 9, "Year": 2026, "Date": "2026-06-14", "Stage": "Group stage", "Home Team": "Germany", "Away Team": "Curaçao", "Host Team": False, "Home Score": 7, "Away Score": 1},
    {"ID": 10, "Year": 2026, "Date": "2026-06-14", "Stage": "Group stage", "Home Team": "Côte d'Ivoire", "Away Team": "Ecuador", "Host Team": False, "Home Score": 1, "Away Score": 0},
    {"ID": 11, "Year": 2026, "Date": "2026-06-14", "Stage": "Group stage", "Home Team": "Netherlands", "Away Team": "Japan", "Host Team": False, "Home Score": 2, "Away Score": 2},
    {"ID": 12, "Year": 2026, "Date": "2026-06-14", "Stage": "Group stage", "Home Team": "Sweden", "Away Team": "Tunisia", "Host Team": False, "Home Score": 5, "Away Score": 1},
    {"ID": 13, "Year": 2026, "Date": "2026-06-15", "Stage": "Group stage", "Home Team": "Spain", "Away Team": "Cabo Verde", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 14, "Year": 2026, "Date": "2026-06-15", "Stage": "Group stage", "Home Team": "Saudi Arabia", "Away Team": "Uruguay", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 15, "Year": 2026, "Date": "2026-06-15", "Stage": "Group stage", "Home Team": "Belgium", "Away Team": "Egypt", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 16, "Year": 2026, "Date": "2026-06-15", "Stage": "Group stage", "Home Team": "IR Iran", "Away Team": "New Zealand", "Host Team": False, "Home Score": 2, "Away Score": 2},
    {"ID": 17, "Year": 2026, "Date": "2026-06-16", "Stage": "Group stage", "Home Team": "France", "Away Team": "Senegal", "Host Team": False, "Home Score": 3, "Away Score": 1},
    {"ID": 18, "Year": 2026, "Date": "2026-06-16", "Stage": "Group stage", "Home Team": "Norway", "Away Team": "Iraq", "Host Team": False, "Home Score": 4, "Away Score": 1},
    {"ID": 19, "Year": 2026, "Date": "2026-06-16", "Stage": "Group stage", "Home Team": "Argentina", "Away Team": "Algeria", "Host Team": False, "Home Score": 3, "Away Score": 0},
    {"ID": 20, "Year": 2026, "Date": "2026-06-16", "Stage": "Group stage", "Home Team": "Austria", "Away Team": "Jordan", "Host Team": False, "Home Score": 3, "Away Score": 1},
    {"ID": 21, "Year": 2026, "Date": "2026-06-17", "Stage": "Group stage", "Home Team": "Portugal", "Away Team": "Congo DR", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 22, "Year": 2026, "Date": "2026-06-17", "Stage": "Group stage", "Home Team": "Colombia", "Away Team": "Uzbekistan", "Host Team": False, "Home Score": 3, "Away Score": 1},
    {"ID": 23, "Year": 2026, "Date": "2026-06-17", "Stage": "Group stage", "Home Team": "England", "Away Team": "Croatia", "Host Team": False, "Home Score": 4, "Away Score": 2},
    {"ID": 24, "Year": 2026, "Date": "2026-06-17", "Stage": "Group stage", "Home Team": "Ghana", "Away Team": "Panama", "Host Team": False, "Home Score": 1, "Away Score": 0},
    
    # Matchday 2
    {"ID": 25, "Year": 2026, "Date": "2026-06-18", "Stage": "Group stage", "Home Team": "Czechia", "Away Team": "South Africa", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 26, "Year": 2026, "Date": "2026-06-18", "Stage": "Group stage", "Home Team": "Switzerland", "Away Team": "Bosnia and Herzegovina", "Host Team": False, "Home Score": 2, "Away Score": 0},
    {"ID": 27, "Year": 2026, "Date": "2026-06-18", "Stage": "Group stage", "Home Team": "Canada", "Away Team": "Qatar", "Host Team": True, "Home Score": 3, "Away Score": 0},
    {"ID": 28, "Year": 2026, "Date": "2026-06-18", "Stage": "Group stage", "Home Team": "Mexico", "Away Team": "Korea Republic", "Host Team": True, "Home Score": 1, "Away Score": 0},
    {"ID": 29, "Year": 2026, "Date": "2026-06-19", "Stage": "Group stage", "Home Team": "Scotland", "Away Team": "Morocco", "Host Team": False, "Home Score": 0, "Away Score": 1},
    {"ID": 30, "Year": 2026, "Date": "2026-06-19", "Stage": "Group stage", "Home Team": "Brazil", "Away Team": "Haiti", "Host Team": False, "Home Score": 4, "Away Score": 1},
    {"ID": 31, "Year": 2026, "Date": "2026-06-19", "Stage": "Group stage", "Home Team": "USA", "Away Team": "Australia", "Host Team": True, "Home Score": 2, "Away Score": 0},
    {"ID": 32, "Year": 2026, "Date": "2026-06-19", "Stage": "Group stage", "Home Team": "Türkiye", "Away Team": "Paraguay", "Host Team": False, "Home Score": 0, "Away Score": 1},
    {"ID": 33, "Year": 2026, "Date": "2026-06-20", "Stage": "Group stage", "Home Team": "Germany", "Away Team": "Côte d'Ivoire", "Host Team": False, "Home Score": 2, "Away Score": 1},
    {"ID": 34, "Year": 2026, "Date": "2026-06-20", "Stage": "Group stage", "Home Team": "Ecuador", "Away Team": "Curaçao", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 35, "Year": 2026, "Date": "2026-06-20", "Stage": "Group stage", "Home Team": "Netherlands", "Away Team": "Sweden", "Host Team": False, "Home Score": 5, "Away Score": 1},
    {"ID": 36, "Year": 2026, "Date": "2026-06-20", "Stage": "Group stage", "Home Team": "Tunisia", "Away Team": "Japan", "Host Team": False, "Home Score": 0, "Away Score": 4},
    {"ID": 37, "Year": 2026, "Date": "2026-06-21", "Stage": "Group stage", "Home Team": "Belgium", "Away Team": "IR Iran", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 38, "Year": 2026, "Date": "2026-06-21", "Stage": "Group stage", "Home Team": "New Zealand", "Away Team": "Egypt", "Host Team": False, "Home Score": 1, "Away Score": 3},
    {"ID": 39, "Year": 2026, "Date": "2026-06-21", "Stage": "Group stage", "Home Team": "Spain", "Away Team": "Saudi Arabia", "Host Team": False, "Home Score": 4, "Away Score": 0},
    {"ID": 40, "Year": 2026, "Date": "2026-06-21", "Stage": "Group stage", "Home Team": "Uruguay", "Away Team": "Cabo Verde", "Host Team": False, "Home Score": 2, "Away Score": 2},
    {"ID": 41, "Year": 2026, "Date": "2026-06-22", "Stage": "Group stage", "Home Team": "France", "Away Team": "Iraq", "Host Team": False, "Home Score": 3, "Away Score": 0},
    {"ID": 42, "Year": 2026, "Date": "2026-06-22", "Stage": "Group stage", "Home Team": "Norway", "Away Team": "Senegal", "Host Team": False, "Home Score": 3, "Away Score": 2},
    {"ID": 43, "Year": 2026, "Date": "2026-06-22", "Stage": "Group stage", "Home Team": "Argentina", "Away Team": "Austria", "Host Team": False, "Home Score": 2, "Away Score": 0},
    {"ID": 44, "Year": 2026, "Date": "2026-06-22", "Stage": "Group stage", "Home Team": "Jordan", "Away Team": "Algeria", "Host Team": False, "Home Score": 1, "Away Score": 2},
    {"ID": 45, "Year": 2026, "Date": "2026-06-23", "Stage": "Group stage", "Home Team": "Portugal", "Away Team": "Uzbekistan", "Host Team": False, "Home Score": 5, "Away Score": 0},
    {"ID": 46, "Year": 2026, "Date": "2026-06-23", "Stage": "Group stage", "Home Team": "Colombia", "Away Team": "Congo DR", "Host Team": False, "Home Score": 1, "Away Score": 0},
    {"ID": 47, "Year": 2026, "Date": "2026-06-23", "Stage": "Group stage", "Home Team": "England", "Away Team": "Ghana", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 48, "Year": 2026, "Date": "2026-06-23", "Stage": "Group stage", "Home Team": "Panama", "Away Team": "Croatia", "Host Team": False, "Home Score": 0, "Away Score": 1},
    
    # Matchday 3
    {"ID": 49, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "Mexico", "Away Team": "Czechia", "Host Team": True, "Home Score": 3, "Away Score": 0},
    {"ID": 50, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "South Africa", "Away Team": "Korea Republic", "Host Team": False, "Home Score": 1, "Away Score": 0},
    {"ID": 51, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "Canada", "Away Team": "Switzerland", "Host Team": True, "Home Score": 1, "Away Score": 2},
    {"ID": 52, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "Bosnia and Herzegovina", "Away Team": "Qatar", "Host Team": False, "Home Score": 3, "Away Score": 1},
    {"ID": 53, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "Scotland", "Away Team": "Brazil", "Host Team": False, "Home Score": 0, "Away Score": 3},
    {"ID": 54, "Year": 2026, "Date": "2026-06-24", "Stage": "Group stage", "Home Team": "Morocco", "Away Team": "Haiti", "Host Team": False, "Home Score": 4, "Away Score": 2},
    {"ID": 55, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "USA", "Away Team": "Türkiye", "Host Team": True, "Home Score": 2, "Away Score": 3},
    {"ID": 56, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "Paraguay", "Away Team": "Australia", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 57, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "Curaçao", "Away Team": "Côte d'Ivoire", "Host Team": False, "Home Score": 0, "Away Score": 2},
    {"ID": 58, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "Ecuador", "Away Team": "Germany", "Host Team": False, "Home Score": 2, "Away Score": 1},
    {"ID": 59, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "Japan", "Away Team": "Sweden", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 60, "Year": 2026, "Date": "2026-06-25", "Stage": "Group stage", "Home Team": "Tunisia", "Away Team": "Netherlands", "Host Team": False, "Home Score": 1, "Away Score": 3},
    {"ID": 61, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "Egypt", "Away Team": "IR Iran", "Host Team": False, "Home Score": 1, "Away Score": 1},
    {"ID": 62, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "New Zealand", "Away Team": "Belgium", "Host Team": False, "Home Score": 1, "Away Score": 5},
    {"ID": 63, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "Cabo Verde", "Away Team": "Saudi Arabia", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 64, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "Uruguay", "Away Team": "Spain", "Host Team": False, "Home Score": 0, "Away Score": 1},
    {"ID": 65, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "Norway", "Away Team": "France", "Host Team": False, "Home Score": 1, "Away Score": 4},
    {"ID": 66, "Year": 2026, "Date": "2026-06-26", "Stage": "Group stage", "Home Team": "Senegal", "Away Team": "Iraq", "Host Team": False, "Home Score": 5, "Away Score": 0},
    {"ID": 67, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "Algeria", "Away Team": "Austria", "Host Team": False, "Home Score": 3, "Away Score": 3},
    {"ID": 68, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "Jordan", "Away Team": "Argentina", "Host Team": False, "Home Score": 1, "Away Score": 3},
    {"ID": 69, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "Colombia", "Away Team": "Portugal", "Host Team": False, "Home Score": 0, "Away Score": 0},
    {"ID": 70, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "DR Congo", "Away Team": "Uzbekistan", "Host Team": False, "Home Score": 3, "Away Score": 1},
    {"ID": 71, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "Panama", "Away Team": "England", "Host Team": False, "Home Score": 0, "Away Score": 2},
    {"ID": 72, "Year": 2026, "Date": "2026-06-27", "Stage": "Group stage", "Home Team": "Croatia", "Away Team": "Ghana", "Host Team": False, "Home Score": 2, "Away Score": 1}
]

def main():
    data_dir = PROJECT_ROOT / 'data'
    
    # 1. Update 2026_world_cup_matches.csv
    df_wc = pd.DataFrame(matches_data)
    df_wc['Penalty Winner'] = 'None'
    df_wc.to_csv(data_dir / '2026_world_cup_matches.csv', index=False)
    print(f"Saved {len(df_wc)} matches to 2026_world_cup_matches.csv")
    
    # 2. Update international_matches.csv
    intl_matches = pd.read_csv(data_dir / 'international_matches.csv')
    print(f"Loaded {len(intl_matches)} matches from international_matches.csv")
    
    updated_count = 0
    
    for match in matches_data:
        date = match['Date']
        h_team = normalize_team_name(match['Home Team'])
        a_team = normalize_team_name(match['Away Team'])
        h_goals = match['Home Score']
        a_goals = match['Away Score']
        
        # Find match on date
        mask_date = intl_matches['Date'] == date
        matches_on_date = intl_matches[mask_date]
        
        found = False
        for idx, row in matches_on_date.iterrows():
            intl_h = normalize_team_name(row['Home Team'])
            intl_a = normalize_team_name(row['Away Team'])
            
            if (intl_h == h_team and intl_a == a_team):
                intl_matches.at[idx, 'Home Goals'] = float(h_goals)
                intl_matches.at[idx, 'Away Goals'] = float(a_goals)
                found = True
                updated_count += 1
                break
            elif (intl_h == a_team and intl_a == h_team):
                intl_matches.at[idx, 'Home Goals'] = float(a_goals)
                intl_matches.at[idx, 'Away Goals'] = float(h_goals)
                found = True
                updated_count += 1
                break
                
        if not found:
            print(f"Warning: Match not found on {date} between {match['Home Team']} and {match['Away Team']}")
            
    # Save back
    intl_matches.to_csv(data_dir / 'international_matches.csv', index=False)
    print(f"Successfully updated {updated_count} matches in international_matches.csv")

if __name__ == '__main__':
    main()
