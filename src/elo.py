"""
Elo Rating System untuk pertandingan sepak bola internasional.

Sistem Elo mengukur kekuatan relatif tim berdasarkan riwayat pertandingan.
Rating awal = 1500 untuk semua tim. Setiap pertandingan memperbarui rating
kedua tim berdasarkan hasil aktual vs hasil yang diharapkan.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class EloRatingSystem:
    """
    Implementasi Elo Rating yang disesuaikan untuk sepak bola internasional.
    
    Fitur:
    - K-factor berbeda berdasarkan jenis turnamen (WC, Qualifier, Friendly)
    - Goal difference multiplier
    - Optional time decay
    """
    
    # K-factor berdasarkan jenis turnamen
    K_FACTORS = {
        'FIFA World Cup': 60,
        'FIFA World Cup qualification': 40,
        'Copa America': 50,
        'UEFA Euro': 50,
        'UEFA Euro qualification': 40,
        'African Cup of Nations': 50,
        'AFC Asian Cup': 50,
        'UEFA Nations League': 40,
        'CONCACAF Gold Cup': 45,
        'Confederations Cup': 45,
        'Friendly': 20,
    }
    DEFAULT_K = 30
    INITIAL_ELO = 1500
    
    def __init__(self, k_factors: Optional[Dict[str, int]] = None, 
                 initial_elo: float = 1500):
        """
        Args:
            k_factors: Dictionary mapping tournament name → K-factor.
                       Jika None, gunakan default.
            initial_elo: Rating awal untuk tim baru.
        """
        self.ratings: Dict[str, float] = {}
        self.k_factors = k_factors or self.K_FACTORS
        self.initial_elo = initial_elo
        self.history: list = []  # Track rating changes over time
    
    def get_rating(self, team: str) -> float:
        """Ambil rating tim. Jika tim belum ada, return initial_elo."""
        return self.ratings.get(team, self.initial_elo)
    
    def get_all_ratings(self) -> Dict[str, float]:
        """Return semua rating saat ini."""
        return dict(sorted(self.ratings.items(), key=lambda x: -x[1]))
    
    def expected_result(self, elo_a: float, elo_b: float) -> float:
        """
        Hitung expected result (probabilitas menang) untuk tim A.
        
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    
    def _get_k_factor(self, tournament: str) -> int:
        """Ambil K-factor berdasarkan jenis turnamen."""
        for key, k in self.k_factors.items():
            if key.lower() in tournament.lower():
                return k
        return self.DEFAULT_K
    
    def _goal_diff_multiplier(self, goal_diff: int) -> float:
        """
        Multiplier berdasarkan selisih gol.
        Kemenangan besar lebih berpengaruh pada perubahan rating.
        
        Formula: G = 1 + (goal_diff - 1) * 0.5 jika goal_diff >= 2
        """
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        else:
            return 1.75 + (abs_diff - 3) * 0.375
    
    def update(self, home_team: str, away_team: str, 
               home_goals: int, away_goals: int,
               tournament: str = 'Friendly',
               date: Optional[str] = None) -> Tuple[float, float]:
        """
        Update Elo rating setelah pertandingan.
        
        Args:
            home_team: Nama tim tuan rumah
            away_team: Nama tim tamu
            home_goals: Gol tim tuan rumah
            away_goals: Gol tim tamu
            tournament: Jenis turnamen
            date: Tanggal pertandingan (opsional, untuk tracking)
            
        Returns:
            Tuple (new_elo_home, new_elo_away)
        """
        elo_home = self.get_rating(home_team)
        elo_away = self.get_rating(away_team)
        
        # Tentukan actual score (S)
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
        
        # Expected scores
        e_home = self.expected_result(elo_home, elo_away)
        e_away = 1.0 - e_home
        
        # K-factor dan goal difference multiplier
        k = self._get_k_factor(tournament)
        gdm = self._goal_diff_multiplier(home_goals - away_goals)
        
        # Update ratings
        new_elo_home = elo_home + k * gdm * (s_home - e_home)
        new_elo_away = elo_away + k * gdm * (s_away - e_away)
        
        self.ratings[home_team] = new_elo_home
        self.ratings[away_team] = new_elo_away
        
        # Track history
        if date:
            self.history.append({
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'tournament': tournament,
                'elo_home_before': elo_home,
                'elo_away_before': elo_away,
                'elo_home_after': new_elo_home,
                'elo_away_after': new_elo_away,
            })
        
        return new_elo_home, new_elo_away
    
    def process_matches(self, df: pd.DataFrame, 
                        home_col: str = 'Home Team',
                        away_col: str = 'Away Team',
                        home_goals_col: str = 'Home Goals',
                        away_goals_col: str = 'Away Goals',
                        tournament_col: str = 'Tournament',
                        date_col: str = 'Date',
                        verbose: bool = False) -> pd.DataFrame:
        """
        Proses seluruh dataframe pertandingan dan hitung Elo rating.
        
        Data HARUS sudah terurut berdasarkan tanggal (ascending).
        
        Returns:
            DataFrame dengan kolom tambahan: elo_home, elo_away (rating sebelum pertandingan)
        """
        elo_home_list = []
        elo_away_list = []
        
        for idx, row in df.iterrows():
            home = row[home_col]
            away = row[away_col]
            
            # Simpan Elo SEBELUM pertandingan (ini yang jadi fitur)
            elo_home_list.append(self.get_rating(home))
            elo_away_list.append(self.get_rating(away))
            
            # Update Elo setelah pertandingan
            h_goals = row[home_goals_col]
            a_goals = row[away_goals_col]
            
            if pd.notna(h_goals) and pd.notna(a_goals):
                tournament = row.get(tournament_col, 'Friendly')
                date = str(row.get(date_col, ''))
                self.update(home, away, int(h_goals), int(a_goals), 
                           tournament, date)
        
        df_result = df.copy()
        df_result['elo_home'] = elo_home_list
        df_result['elo_away'] = elo_away_list
        df_result['elo_diff'] = df_result['elo_home'] - df_result['elo_away']
        
        if verbose:
            print(f"Processed {len(df)} matches for {len(self.ratings)} teams")
            top5 = sorted(self.ratings.items(), key=lambda x: -x[1])[:5]
            print("Top 5 teams:", [(t, round(r, 1)) for t, r in top5])
        
        return df_result
    
    def get_history_df(self) -> pd.DataFrame:
        """Return history sebagai DataFrame."""
        return pd.DataFrame(self.history)
    
    def predict_match(self, team_a: str, team_b: str) -> Dict[str, float]:
        """
        Prediksi probabilitas hasil pertandingan berdasarkan Elo.
        
        Returns:
            Dict dengan P(team_a_win), P(draw), P(team_b_win)
        """
        elo_a = self.get_rating(team_a)
        elo_b = self.get_rating(team_b)
        
        e_a = self.expected_result(elo_a, elo_b)
        e_b = 1.0 - e_a
        
        # Estimasi draw probability (dari analisis empiris sepak bola)
        # Draw prob lebih tinggi ketika tim seimbang
        draw_base = 0.25
        elo_diff = abs(elo_a - elo_b)
        draw_prob = draw_base * np.exp(-elo_diff / 600)
        
        # Adjust win/loss probabilities
        win_a = e_a * (1 - draw_prob)
        win_b = e_b * (1 - draw_prob)
        
        return {
            f'{team_a}_win': round(win_a, 4),
            'draw': round(draw_prob, 4),
            f'{team_b}_win': round(win_b, 4),
            'elo_a': round(elo_a, 1),
            'elo_b': round(elo_b, 1),
        }
