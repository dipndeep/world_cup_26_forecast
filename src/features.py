"""
Feature Engineering untuk prediksi pertandingan sepak bola internasional.

Modul ini menyediakan fungsi-fungsi untuk membuat fitur prediktif
dari data pertandingan mentah, termasuk:
- Form terkini (recent form)
- Head-to-head statistics
- Goal statistics
- World Cup experience
- Team name normalization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# Team Name Normalization
# ============================================================

# Mapping nama tim dari berbagai dataset ke nama standar
TEAM_NAME_MAP = {
    # 2026 WC groups → international_matches
    'Korea Republic': 'South Korea',
    'Türkiye': 'Turkey',
    'Côte d\'Ivoire': 'Ivory Coast',
    'IR Iran': 'Iran',
    'Cabo Verde': 'Cape Verde',
    'Congo DR': 'DR Congo',
    'Czechia': 'Czech Republic',
    # Historical variations
    'Germany FR': 'Germany',
    'Soviet Union': 'Russia',
    'China PR': 'China',
    'Chinese Taipei': 'Taiwan',
    'USA': 'United States',
    'Korea DPR': 'North Korea',
}


def normalize_team_name(name: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """Normalisasi nama tim ke format standar."""
    if mapping is None:
        mapping = TEAM_NAME_MAP
    return mapping.get(name, name)


def normalize_team_names_in_df(df: pd.DataFrame, 
                                columns: List[str],
                                mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Normalisasi nama tim di beberapa kolom DataFrame.
    
    Args:
        df: DataFrame input
        columns: List nama kolom yang berisi nama tim
        mapping: Custom mapping dictionary
    
    Returns:
        DataFrame dengan nama tim yang sudah dinormalisasi
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_team_name(x, mapping))
    return df


# ============================================================
# Recent Form Features
# ============================================================

def calculate_recent_form(df: pd.DataFrame, 
                          team: str,
                          before_date: str,
                          n_matches: int = 10,
                          home_col: str = 'Home Team',
                          away_col: str = 'Away Team',
                          home_goals_col: str = 'Home Goals',
                          away_goals_col: str = 'Away Goals',
                          date_col: str = 'Date') -> Dict[str, float]:
    """
    Hitung form terkini suatu tim dari n pertandingan terakhir sebelum tanggal tertentu.
    
    Returns:
        Dict berisi: win_rate, draw_rate, loss_rate, avg_goals_scored, 
                     avg_goals_conceded, goal_diff_avg
    """
    # Filter pertandingan tim sebelum tanggal
    mask_home = (df[home_col] == team) & (df[date_col] < before_date)
    mask_away = (df[away_col] == team) & (df[date_col] < before_date)
    
    df_home = df[mask_home][[date_col, home_goals_col, away_goals_col]].rename(
        columns={home_goals_col: 'goals_for', away_goals_col: 'goals_against', date_col: 'date'}
    )
    df_away = df[mask_away][[date_col, away_goals_col, home_goals_col]].rename(
        columns={away_goals_col: 'goals_for', home_goals_col: 'goals_against', date_col: 'date'}
    )
    
    if len(df_home) == 0 and len(df_away) == 0:
        return {
            'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
            'avg_goals_scored': 0.0, 'avg_goals_conceded': 0.0,
            'goal_diff_avg': 0.0, 'n_matches': 0,
        }
    
    # Sort by date descending dan ambil n terakhir
    matches_df = pd.concat([df_home, df_away]).sort_values('date', ascending=False).head(n_matches)
    
    wins = (matches_df['goals_for'] > matches_df['goals_against']).sum()
    draws = (matches_df['goals_for'] == matches_df['goals_against']).sum()
    losses = (matches_df['goals_for'] < matches_df['goals_against']).sum()
    total = len(matches_df)
    
    return {
        'win_rate': float(wins / total),
        'draw_rate': float(draws / total),
        'loss_rate': float(losses / total),
        'avg_goals_scored': float(matches_df['goals_for'].mean()),
        'avg_goals_conceded': float(matches_df['goals_against'].mean()),
        'goal_diff_avg': float((matches_df['goals_for'] - matches_df['goals_against']).mean()),
        'n_matches': int(total),
    }


def add_form_features(df: pd.DataFrame, n_matches: int = 10,
                      home_col: str = 'Home Team',
                      away_col: str = 'Away Team',
                      date_col: str = 'Date') -> pd.DataFrame:
    """
    Tambahkan fitur form terkini untuk home dan away team pada setiap baris.
    
    PERHATIAN: Fungsi ini bisa lambat untuk dataset besar.
    Gunakan build_form_features_vectorized() untuk performa lebih baik.
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    form_cols = ['win_rate', 'draw_rate', 'avg_goals_scored', 
                 'avg_goals_conceded', 'goal_diff_avg']
    
    for prefix, team_col in [('home', home_col), ('away', away_col)]:
        for col in form_cols:
            df[f'{prefix}_{col}'] = np.nan
    
    # Cache untuk menghindari perhitungan ulang
    cache = {}
    
    for idx, row in df.iterrows():
        for prefix, team_col in [('home', home_col), ('away', away_col)]:
            team = row[team_col]
            date = row[date_col]
            cache_key = (team, date)
            
            if cache_key not in cache:
                form = calculate_recent_form(df, team, date, n_matches,
                                             home_col, away_col,
                                             'Home Goals', 'Away Goals', date_col)
                cache[cache_key] = form
            
            form = cache[cache_key]
            for col in form_cols:
                df.at[idx, f'{prefix}_{col}'] = form[col]
    
    return df


# ============================================================
# Head-to-Head Features
# ============================================================

def calculate_h2h(df: pd.DataFrame,
                  team_a: str, team_b: str,
                  before_date: str,
                  home_col: str = 'Home Team',
                  away_col: str = 'Away Team',
                  home_goals_col: str = 'Home Goals',
                  away_goals_col: str = 'Away Goals',
                  date_col: str = 'Date') -> Dict[str, float]:
    """
    Hitung statistik head-to-head antara dua tim.
    
    Returns:
        Dict: h2h_win_rate_a, h2h_draw_rate, h2h_goal_diff_a, h2h_total_matches
    """
    mask1 = ((df[home_col] == team_a) & (df[away_col] == team_b) & 
             (df[date_col] < before_date))
    mask2 = ((df[home_col] == team_b) & (df[away_col] == team_a) & 
             (df[date_col] < before_date))
    
    df1 = df[mask1][[home_goals_col, away_goals_col]].dropna()
    df2 = df[mask2][[away_goals_col, home_goals_col]].dropna()
    
    df1.columns = ['gf_a', 'ga_a']
    df2.columns = ['gf_a', 'ga_a']
    
    merged = pd.concat([df1, df2])
    total = len(merged)
    
    if total == 0:
        return {
            'h2h_win_rate_a': 0.5,
            'h2h_draw_rate': 0.25,
            'h2h_goal_diff_a': 0.0,
            'h2h_total_matches': 0,
        }
    
    wins_a = (merged['gf_a'] > merged['ga_a']).sum()
    draws = (merged['gf_a'] == merged['ga_a']).sum()
    total_gd = (merged['gf_a'] - merged['ga_a']).sum()
    
    return {
        'h2h_win_rate_a': float(wins_a / total),
        'h2h_draw_rate': float(draws / total),
        'h2h_goal_diff_a': float(total_gd / total),
        'h2h_total_matches': int(total),
    }


# ============================================================
# World Cup Experience Features
# ============================================================

def calculate_wc_experience(wc_matches_df: pd.DataFrame,
                            wc_summary_df: pd.DataFrame,
                            team: str) -> Dict[str, int]:
    """
    Hitung pengalaman Piala Dunia suatu tim.
    
    Returns:
        Dict: wc_appearances, wc_matches_played, wc_wins, wc_titles, 
              wc_finals, wc_goals_scored
    """
    # Dari world_cup_matches
    home_matches = wc_matches_df[wc_matches_df['Home Team'] == team]
    away_matches = wc_matches_df[wc_matches_df['Away Team'] == team]
    
    total_matches = len(home_matches) + len(away_matches)
    
    wins = 0
    goals = 0
    
    for _, row in home_matches.iterrows():
        if pd.notna(row.get('Home Goals')):
            goals += int(row['Home Goals'])
            if row['Home Goals'] > row['Away Goals']:
                wins += 1
    
    for _, row in away_matches.iterrows():
        if pd.notna(row.get('Away Goals')):
            goals += int(row['Away Goals'])
            if row['Away Goals'] > row['Home Goals']:
                wins += 1
    
    # Appearances (edisi WC yang diikuti)
    years_home = set(home_matches['Year'].unique())
    years_away = set(away_matches['Year'].unique())
    appearances = len(years_home | years_away)
    
    # Titles dan Finals dari world_cups summary
    titles = 0
    finals = 0
    if 'Winner' in wc_summary_df.columns:
        titles = (wc_summary_df['Winner'] == team).sum()
        finals = titles + (wc_summary_df['Runners-Up'] == team).sum()
    
    return {
        'wc_appearances': appearances,
        'wc_matches_played': total_matches,
        'wc_wins': wins,
        'wc_titles': titles,
        'wc_finals': finals,
        'wc_goals_scored': goals,
    }


# ============================================================
# Squad Features (dari data 2022)
# ============================================================

def calculate_squad_features(squads_df: pd.DataFrame, 
                              team: str) -> Dict[str, float]:
    """
    Hitung fitur skuad dari data 2022 (sebagai proksi untuk 2026).
    
    Returns:
        Dict: avg_age, avg_caps, avg_goals, total_wc_goals, 
              n_goalkeepers, n_defenders, n_midfielders, n_forwards,
              league_diversity
    """
    team_squad = squads_df[squads_df['Team'] == team]
    
    if len(team_squad) == 0:
        return {
            'avg_age': np.nan, 'avg_caps': np.nan, 'avg_goals': np.nan,
            'total_wc_goals': 0, 'league_diversity': 0,
        }
    
    return {
        'avg_age': team_squad['Age'].mean(),
        'avg_caps': team_squad['Caps'].mean(),
        'avg_goals': team_squad['Goals'].mean(),
        'total_wc_goals': team_squad['WC Goals'].sum(),
        'league_diversity': team_squad['League'].nunique(),
    }


# ============================================================
# Combined Feature Builder
# ============================================================

def build_match_features(row: pd.Series,
                         df_all: pd.DataFrame,
                         wc_matches_df: pd.DataFrame,
                         wc_summary_df: pd.DataFrame,
                         elo_system,
                         n_form_matches: int = 10) -> Dict[str, float]:
    """
    Bangun semua fitur untuk satu pertandingan.
    
    Args:
        row: Satu baris dari DataFrame pertandingan
        df_all: DataFrame lengkap pertandingan internasional
        wc_matches_df: DataFrame pertandingan WC
        wc_summary_df: DataFrame ringkasan WC
        elo_system: Instance EloRatingSystem
        n_form_matches: Jumlah pertandingan untuk menghitung form
        
    Returns:
        Dict berisi semua fitur
    """
    home = row['Home Team']
    away = row['Away Team']
    date = row['Date']
    
    features = {}
    
    # 1. Elo Ratings
    features['elo_home'] = elo_system.get_rating(home)
    features['elo_away'] = elo_system.get_rating(away)
    features['elo_diff'] = features['elo_home'] - features['elo_away']
    
    # 2. Recent Form
    home_form = calculate_recent_form(df_all, home, date, n_form_matches)
    away_form = calculate_recent_form(df_all, away, date, n_form_matches)
    
    for key, val in home_form.items():
        if key != 'n_matches':
            features[f'home_{key}'] = val
    for key, val in away_form.items():
        if key != 'n_matches':
            features[f'away_{key}'] = val
    
    # 3. Head-to-Head
    h2h = calculate_h2h(df_all, home, away, date)
    features.update(h2h)
    
    # 4. WC Experience
    home_wc = calculate_wc_experience(wc_matches_df, wc_summary_df, home)
    away_wc = calculate_wc_experience(wc_matches_df, wc_summary_df, away)
    
    for key, val in home_wc.items():
        features[f'home_{key}'] = val
    for key, val in away_wc.items():
        features[f'away_{key}'] = val
    
    # 5. Derived features
    features['wc_experience_diff'] = (home_wc['wc_appearances'] - 
                                       away_wc['wc_appearances'])
    features['wc_titles_diff'] = (home_wc['wc_titles'] - 
                                   away_wc['wc_titles'])
    features['form_diff'] = (home_form['win_rate'] - away_form['win_rate'])
    features['goal_diff_form'] = (home_form['goal_diff_avg'] - 
                                   away_form['goal_diff_avg'])
    
    return features
