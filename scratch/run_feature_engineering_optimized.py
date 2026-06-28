import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
import bisect
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import EloRatingSystem
from src.features import (
    normalize_team_name, normalize_team_names_in_df, TEAM_NAME_MAP,
    calculate_wc_experience
)

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def build_team_history_cache(df, home_col='Home Team', away_col='Away Team', 
                             home_goals_col='Home Goals', away_goals_col='Away Goals', 
                             date_col='Date'):
    """
    Build a cache of all matches for each team sorted by date.
    Each team mapped to: {
        'dates': [datetime_objects],
        'goals_for': [list],
        'goals_against': [list]
    }
    """
    cache = {}
    
    # Sort dataframe by date to ensure matches are in order
    df_sorted = df.sort_values(date_col).copy()
    
    for idx, row in df_sorted.iterrows():
        date = row[date_col]
        home = row[home_col]
        away = row[away_col]
        h_goals = row[home_goals_col]
        a_goals = row[away_goals_col]
        
        if pd.isna(h_goals) or pd.isna(a_goals):
            continue
            
        h_goals = int(h_goals)
        a_goals = int(a_goals)
        
        # Home team record
        if home not in cache:
            cache[home] = {'dates': [], 'goals_for': [], 'goals_against': []}
        cache[home]['dates'].append(date)
        cache[home]['goals_for'].append(h_goals)
        cache[home]['goals_against'].append(a_goals)
        
        # Away team record
        if away not in cache:
            cache[away] = {'dates': [], 'goals_for': [], 'goals_against': []}
        cache[away]['dates'].append(date)
        cache[away]['goals_for'].append(a_goals)
        cache[away]['goals_against'].append(h_goals)
        
    return cache

def calculate_recent_form_optimized(history_cache, team, before_date, n_matches=10):
    """
    Fast optimized form calculation using binary search.
    """
    if team not in history_cache:
        return {
            'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
            'avg_goals_scored': 0.0, 'avg_goals_conceded': 0.0,
            'goal_diff_avg': 0.0, 'n_matches': 0,
        }
        
    hist = history_cache[team]
    dates = hist['dates']
    
    # Find position of first match on or after before_date
    idx = bisect.bisect_left(dates, before_date)
    
    if idx == 0:
        return {
            'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
            'avg_goals_scored': 0.0, 'avg_goals_conceded': 0.0,
            'goal_diff_avg': 0.0, 'n_matches': 0,
        }
        
    # Get up to n_matches matches before before_date
    start = max(0, idx - n_matches)
    
    gf = hist['goals_for'][start:idx]
    ga = hist['goals_against'][start:idx]
    
    total = idx - start
    
    wins = sum(1 for f, a in zip(gf, ga) if f > a)
    draws = sum(1 for f, a in zip(gf, ga) if f == a)
    losses = total - wins - draws
    
    avg_gf = sum(gf) / total if total > 0 else 0.0
    avg_ga = sum(ga) / total if total > 0 else 0.0
    avg_gd = sum(f - a for f, a in zip(gf, ga)) / total if total > 0 else 0.0
    
    return {
        'win_rate': float(wins / total),
        'draw_rate': float(draws / total),
        'loss_rate': float(losses / total),
        'avg_goals_scored': float(avg_gf),
        'avg_goals_conceded': float(avg_ga),
        'goal_diff_avg': float(avg_gd),
        'n_matches': int(total),
    }

def main():
    print("Step 1: Loading and Normalizing Data...")
    intl = pd.read_csv(DATA_DIR / 'international_matches.csv')
    wc_matches = pd.read_csv(DATA_DIR / 'world_cup_matches.csv')
    world_cups = pd.read_csv(DATA_DIR / 'world_cups.csv')
    wc2026_squads = pd.read_csv(DATA_DIR / '2026_world_cup_squads_lengkap.csv')
    wc2026_groups = pd.read_csv(DATA_DIR / '2026_world_cup_groups.csv')

    intl['Date'] = pd.to_datetime(intl['Date'])
    wc_matches['Date'] = pd.to_datetime(wc_matches['Date'])

    intl = intl.sort_values('Date').reset_index(drop=True)

    intl = normalize_team_names_in_df(intl, ['Home Team', 'Away Team'])
    wc_matches = normalize_team_names_in_df(wc_matches, ['Home Team', 'Away Team'])
    world_cups = normalize_team_names_in_df(world_cups, ['Winner', 'Runners-Up', 'Third', 'Fourth'])
    wc2026_groups = normalize_team_names_in_df(wc2026_groups, ['Team'])

    print(f"Loaded {len(intl)} international matches.")

    print("\nStep 2: Processing Elo ratings...")
    elo = EloRatingSystem(initial_elo=1500)
    intl_with_elo = elo.process_matches(
        intl, 
        home_col='Home Team', away_col='Away Team',
        home_goals_col='Home Goals', away_goals_col='Away Goals',
        tournament_col='Tournament', date_col='Date',
        verbose=True
    )

    # Save Elo History Plot
    print("Plotting Elo ratings history...")
    fig, ax = plt.subplots(figsize=(12, 7))
    top_teams = ['Spain', 'Argentina', 'France', 'England', 'Germany', 'Brazil', 'Colombia', 'Ecuador']
    for team in top_teams:
        hist_df = elo.get_history_df()
        if len(hist_df) > 0:
            home_hist = hist_df[hist_df['home_team'] == team][['date', 'elo_home_after']].rename(columns={'elo_home_after': 'elo'})
            away_hist = hist_df[hist_df['away_team'] == team][['date', 'elo_away_after']].rename(columns={'elo_away_after': 'elo'})
            team_hist = pd.concat([home_hist, away_hist]).sort_values('date')
            team_hist['date'] = pd.to_datetime(team_hist['date'])
            team_hist = team_hist[team_hist['date'] >= '2000-01-01']
            if len(team_hist) > 0:
                team_hist = team_hist.set_index('date').resample('3ME').last().dropna()
                ax.plot(team_hist.index, team_hist['elo'], linewidth=2, label=team, alpha=0.85)
    ax.set_title('Elo Rating History (2000-2026)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / '02_elo_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nStep 3: Preparing valid matches data...")
    intl_valid = intl_with_elo.dropna(subset=['Home Goals', 'Away Goals']).copy()
    intl_valid['Home Goals'] = intl_valid['Home Goals'].astype(int)
    intl_valid['Away Goals'] = intl_valid['Away Goals'].astype(int)

    intl_valid['result'] = 1  # Draw
    intl_valid.loc[intl_valid['Home Goals'] > intl_valid['Away Goals'], 'result'] = 2  # Home Win
    intl_valid.loc[intl_valid['Home Goals'] < intl_valid['Away Goals'], 'result'] = 0  # Away Win

    recent_data = intl_valid[intl_valid['Date'] >= '2000-01-01'].copy()
    total_rows = len(recent_data)
    print(f"Total training data from 2000 onwards: {total_rows} matches.")

    print("\nStep 4: Computing form features (optimized)...")
    history_cache = build_team_history_cache(intl_valid)
    form_features = []
    
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        if i % 5000 == 0:
            print(f"  Processed {i}/{total_rows} matches ({(i/total_rows)*100:.1f}%)...")
            
        home = row['Home Team']
        away = row['Away Team']
        date = row['Date']
        
        home_form = calculate_recent_form_optimized(history_cache, home, date, n_matches=10)
        away_form = calculate_recent_form_optimized(history_cache, away, date, n_matches=10)
        
        feat = {
            'idx': idx,
            'elo_home': row['elo_home'],
            'elo_away': row['elo_away'],
            'elo_diff': row['elo_diff'],
            'home_win_rate': home_form['win_rate'],
            'home_avg_goals_scored': home_form['avg_goals_scored'],
            'home_avg_goals_conceded': home_form['avg_goals_conceded'],
            'home_goal_diff_avg': home_form['goal_diff_avg'],
            'away_win_rate': away_form['win_rate'],
            'away_avg_goals_scored': away_form['avg_goals_scored'],
            'away_avg_goals_conceded': away_form['avg_goals_conceded'],
            'away_goal_diff_avg': away_form['goal_diff_avg'],
            'form_diff': home_form['win_rate'] - away_form['win_rate'],
            'goal_diff_form': home_form['goal_diff_avg'] - away_form['goal_diff_avg'],
        }
        form_features.append(feat)

    form_df = pd.DataFrame(form_features).set_index('idx')
    print("Form features computation complete.")

    print("\nStep 5: Computing World Cup experience features...")
    wc_exp_cache = {}
    all_teams_in_data = set(recent_data['Home Team'].unique()) | set(recent_data['Away Team'].unique())
    for team in all_teams_in_data:
        wc_exp_cache[team] = calculate_wc_experience(wc_matches, world_cups, team)

    wc_features = []
    for idx, row in recent_data.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        
        h_wc = wc_exp_cache.get(home, {'wc_appearances': 0, 'wc_titles': 0})
        a_wc = wc_exp_cache.get(away, {'wc_appearances': 0, 'wc_titles': 0})
        
        wc_features.append({
            'idx': idx,
            'home_wc_appearances': h_wc['wc_appearances'],
            'away_wc_appearances': a_wc['wc_appearances'],
            'home_wc_titles': h_wc['wc_titles'],
            'away_wc_titles': a_wc['wc_titles'],
            'wc_exp_diff': h_wc['wc_appearances'] - a_wc['wc_appearances'],
            'wc_titles_diff': h_wc['wc_titles'] - a_wc['wc_titles'],
        })

    wc_df = pd.DataFrame(wc_features).set_index('idx')

    print("\nStep 6: Combining and Saving Features...")
    feature_matrix = recent_data[['Date', 'Home Team', 'Away Team', 'Tournament',
                                   'Home Goals', 'Away Goals', 'result']].copy()
    feature_matrix = feature_matrix.join(form_df)
    feature_matrix = feature_matrix.join(wc_df)

    feature_matrix['is_wc'] = feature_matrix['Tournament'].str.contains('World Cup', na=False).astype(int)
    feature_matrix['is_friendly'] = (feature_matrix['Tournament'] == 'Friendly').astype(int)
    feature_matrix['is_qualifier'] = feature_matrix['Tournament'].str.contains('qualification|qualifier', case=False, na=False).astype(int)

    feature_matrix.to_csv(OUTPUT_DIR / 'predictions' / 'feature_matrix.csv', index=False)
    print("Feature matrix saved to predictions/feature_matrix.csv")

    elo_ratings_df = pd.DataFrame([
        {'Team': t, 'Elo': r} for t, r in elo.get_all_ratings().items()
    ])
    elo_ratings_df.to_csv(OUTPUT_DIR / 'predictions' / 'elo_ratings.csv', index=False)
    print("Elo ratings saved to predictions/elo_ratings.csv")

    print("\nStep 7: Building team profiles for WC 2026...")
    wc2026_profiles = []
    latest_date = intl_valid['Date'].max()
    for _, row in wc2026_groups.iterrows():
        team = row['Team']
        form = calculate_recent_form_optimized(history_cache, team, latest_date, n_matches=10)
        wc_exp = wc_exp_cache.get(team, {'wc_appearances': 0, 'wc_titles': 0})
        wc2026_profiles.append({
            'Team': team,
            'Group': row['Group'],
            'FIFA_Ranking': row['FIFA Ranking'],
            'Elo': elo.get_rating(team),
            'Win_Rate_Last10': form['win_rate'],
            'Avg_Goals_Scored': form['avg_goals_scored'],
            'Avg_Goals_Conceded': form['avg_goals_conceded'],
            'Goal_Diff_Avg': form['goal_diff_avg'],
            'WC_Appearances': wc_exp['wc_appearances'],
            'WC_Titles': wc_exp.get('wc_titles', 0)
        })

    wc2026_profile_df = pd.DataFrame(wc2026_profiles).sort_values('Elo', ascending=False)
    wc2026_profile_df.to_csv(OUTPUT_DIR / 'predictions' / 'wc2026_team_profiles.csv', index=False)
    print("WC 2026 team profiles saved to predictions/wc2026_team_profiles.csv")
    print("[SUCCESS] Feature engineering script complete!")

if __name__ == '__main__':
    main()
