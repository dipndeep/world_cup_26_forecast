"""
Simulasi Turnamen FIFA World Cup 2026 (format 48 tim).

Format WC 2026:
- 12 grup × 4 tim
- Fase grup: round-robin dalam grup
- Top 2 per grup (24 tim) + 8 best third-place = 32 tim
- Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final

Menggunakan Monte Carlo simulation untuk mengestimasi probabilitas
setiap tim menjuara, semifinalis, dll.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from collections import defaultdict
import random


class WorldCup2026Simulator:
    """
    Simulator turnamen FIFA World Cup 2026.
    """
    
    # Bracket Round of 32 (berdasarkan aturan FIFA):
    # Match 1: 1A vs 2C, Match 2: 1B vs 3A/D/E
    # dst. — simplified bracket
    # Referensi: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026
    
    # Simplified bracket pairing for Round of 32
    # Format: (pos_from_group, group_letter)
    # 1X = 1st in group X, 2X = 2nd in group X, 3X = 3rd place from group X
    R32_BRACKET = [
        # Top half
        ('1A', '2C'),   # Match R32-1
        ('1B', '3A/D/E/F'),  # Match R32-2
        ('1C', '2A'),   # Match R32-3
        ('1D', '3B/E/F'),    # Match R32-4
        ('1E', '2G'),   # Match R32-5
        ('1F', '3C/D/G/H'),  # Match R32-6
        ('1G', '2E'),   # Match R32-7
        ('1H', '3A/B/G/H'),  # Match R32-8
        # Bottom half
        ('1I', '2K'),   # Match R32-9
        ('1J', '3I/J/K/L'),  # Match R32-10
        ('1K', '2I'),   # Match R32-11
        ('1L', '3C/D/I/J'),  # Match R32-12
        ('2B', '2D'),   # Match R32-13
        ('2F', '2H'),   # Match R32-14
        ('2J', '2L'),   # Match R32-15
        ('3E/F/I/J', '3G/H/K/L'),  # Match R32-16
    ]
    
    def __init__(self, groups_df: pd.DataFrame,
                 predict_fn: Optional[Callable] = None,
                 seed: int = 42):
        """
        Args:
            groups_df: DataFrame dengan kolom 'Group', 'Team', 'FIFA Ranking'
            predict_fn: Fungsi yang menerima (team_a, team_b) dan return 
                       dict {'home_win': p, 'draw': p, 'away_win': p}
                       Jika None, gunakan Elo-based prediction.
            seed: Random seed
        """
        self.groups_df = groups_df.copy()
        self.predict_fn = predict_fn
        self.rng = np.random.RandomState(seed)
        
        # Parse groups
        self.groups = {}
        for group_name, group_df in groups_df.groupby('Group'):
            self.groups[group_name] = group_df['Team'].tolist()
    
    def set_predict_fn(self, predict_fn: Callable):
        """Set fungsi prediksi setelah inisialisasi."""
        self.predict_fn = predict_fn
    
    def simulate_match(self, team_a: str, team_b: str, 
                       allow_draw: bool = True) -> Tuple[str, int, int]:
        """
        Simulasikan satu pertandingan.
        
        Args:
            team_a, team_b: Nama tim
            allow_draw: Apakah seri diperbolehkan (False untuk knockout)
            
        Returns:
            Tuple (winner, goals_a, goals_b)
        """
        if self.predict_fn:
            probs = self.predict_fn(team_a, team_b)
            p_a = probs.get('home_win', probs.get(f'{team_a}_win', 0.4))
            p_draw = probs.get('draw', 0.25)
            p_b = probs.get('away_win', probs.get(f'{team_b}_win', 0.35))
        else:
            # Fallback: berdasarkan FIFA Ranking (lower = better)
            rank_a = self._get_ranking(team_a)
            rank_b = self._get_ranking(team_b)
            strength_a = 1.0 / (rank_a + 1)
            strength_b = 1.0 / (rank_b + 1)
            total = strength_a + strength_b
            p_a = strength_a / total * 0.75  # Scale down for draw
            p_b = strength_b / total * 0.75
            p_draw = 1.0 - p_a - p_b
        
        # Normalize
        total = p_a + p_draw + p_b
        p_a /= total
        p_draw /= total
        p_b /= total
        
        if not allow_draw:
            # Redistribute draw probability
            total_win = p_a + p_b
            p_a = p_a / total_win
            p_b = p_b / total_win
            p_draw = 0.0
        
        # Sample result
        r = self.rng.random()
        if r < p_a:
            # Generate plausible scoreline for team_a win
            goals_a = self.rng.choice([1, 2, 3, 2, 1], p=[0.3, 0.35, 0.2, 0.1, 0.05])
            goals_b = self.rng.choice(range(goals_a), 
                                       p=self._goal_probs(goals_a))
            return team_a, goals_a, goals_b
        elif r < p_a + p_draw:
            goals = self.rng.choice([0, 1, 1, 2, 2, 3], 
                                     p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03])
            return 'draw', goals, goals
        else:
            goals_b = self.rng.choice([1, 2, 3, 2, 1], p=[0.3, 0.35, 0.2, 0.1, 0.05])
            goals_a = self.rng.choice(range(goals_b),
                                       p=self._goal_probs(goals_b))
            return team_b, goals_a, goals_b
    
    def _goal_probs(self, max_goals: int) -> list:
        """Generate probability distribution for losing team's goals."""
        if max_goals <= 0:
            return [1.0]
        probs = [1.0 / (i + 1) for i in range(max_goals)]
        total = sum(probs)
        return [p / total for p in probs]
    
    def _get_ranking(self, team: str) -> int:
        """Get FIFA ranking for a team."""
        row = self.groups_df[self.groups_df['Team'] == team]
        if len(row) > 0:
            return int(row.iloc[0]['FIFA Ranking'])
        return 50  # Default ranking
    
    def simulate_group_stage(self, as_dataframe: bool = True) -> Dict[str, Union[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        Simulasikan fase grup lengkap.
        
        Returns:
            Dict mapping group_name → standings DataFrame or list of dicts
        """
        all_standings = {}
        
        for group_name, teams in self.groups.items():
            # Inisialisasi tabel
            table = {team: {'Team': team, 'MP': 0, 'W': 0, 'D': 0, 'L': 0, 
                           'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0, 'Group': group_name}
                     for team in teams}
            
            # Round-robin: setiap tim main melawan semua tim lain
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a, team_b = teams[i], teams[j]
                    winner, goals_a, goals_b = self.simulate_match(
                        team_a, team_b, allow_draw=True)
                    
                    table[team_a]['MP'] += 1
                    table[team_b]['MP'] += 1
                    table[team_a]['GF'] += goals_a
                    table[team_a]['GA'] += goals_b
                    table[team_b]['GF'] += goals_b
                    table[team_b]['GA'] += goals_a
                    
                    if winner == team_a:
                        table[team_a]['W'] += 1
                        table[team_a]['Pts'] += 3
                        table[team_b]['L'] += 1
                    elif winner == team_b:
                        table[team_b]['W'] += 1
                        table[team_b]['Pts'] += 3
                        table[team_a]['L'] += 1
                    else:  # Draw
                        table[team_a]['D'] += 1
                        table[team_b]['D'] += 1
                        table[team_a]['Pts'] += 1
                        table[team_b]['Pts'] += 1
                    
                    table[team_a]['GD'] = table[team_a]['GF'] - table[team_a]['GA']
                    table[team_b]['GD'] = table[team_b]['GF'] - table[team_b]['GA']
            
            # Sort teams using python built-in sort (extremely fast)
            sorted_teams = sorted(table.values(), key=lambda x: (-x['Pts'], -x['GD'], -x['GF']))
            
            if as_dataframe:
                standings = pd.DataFrame(sorted_teams)
                standings['Position'] = range(1, len(standings) + 1)
                all_standings[group_name] = standings
            else:
                for idx, t in enumerate(sorted_teams):
                    t['Position'] = idx + 1
                all_standings[group_name] = sorted_teams
        
        return all_standings
    
    def get_qualified_teams(self, standings: Dict[str, Union[pd.DataFrame, List[Dict[str, Any]]]]) -> Dict[str, List[str]]:
        """
        Tentukan tim yang lolos dari fase grup.
        
        Returns:
            Dict: {'first': [...], 'second': [...], 'third': [...]}
        """
        firsts = []
        seconds = []
        thirds = []
        
        for group_name in sorted(standings.keys()):
            group_data = standings[group_name]
            if isinstance(group_data, pd.DataFrame):
                firsts.append(group_data.iloc[0]['Team'])
                seconds.append(group_data.iloc[1]['Team'])
                thirds.append({
                    'team': group_data.iloc[2]['Team'],
                    'group': group_name,
                    'pts': group_data.iloc[2]['Pts'],
                    'gd': group_data.iloc[2]['GD'],
                    'gf': group_data.iloc[2]['GF'],
                })
            else: # list of dicts
                firsts.append(group_data[0]['Team'])
                seconds.append(group_data[1]['Team'])
                thirds.append({
                    'team': group_data[2]['Team'],
                    'group': group_name,
                    'pts': group_data[2]['Pts'],
                    'gd': group_data[2]['GD'],
                    'gf': group_data[2]['GF'],
                })
        
        # Sort thirds by Pts, GD, GF dan ambil top 8
        thirds_sorted = sorted(thirds, key=lambda x: (-x['pts'], -x['gd'], -x['gf']))
        best_thirds = [t['team'] for t in thirds_sorted[:8]]
        
        return {
            'first': firsts,
            'second': seconds,
            'third': best_thirds,
        }
    
    def simulate_knockout(self, qualified: Dict[str, List[str]], 
                          standings: Dict[str, pd.DataFrame]) -> Dict:
        """
        Simulasikan fase knockout (Round of 32 → Final).
        
        Returns:
            Dict berisi hasil setiap ronde dan pemenang
        """
        # Simplified: pair up teams for knockout
        all_qualified = qualified['first'] + qualified['second'] + qualified['third']
        
        # Shuffle third-place teams into bracket positions
        # (Simplified — real FIFA bracket is more complex)
        round_of_32_teams = []
        
        # Pair 1st place with 2nd/3rd from other groups
        for i, first in enumerate(qualified['first']):
            # Pair with second from a different group
            second_idx = (i + 2) % len(qualified['second'])
            round_of_32_teams.append((first, qualified['second'][second_idx]))
        
        # Pair remaining with third-place teams
        remaining_seconds = [s for i, s in enumerate(qualified['second']) 
                            if i not in [(j + 2) % len(qualified['second']) 
                                        for j in range(len(qualified['first']))]]
        
        third_idx = 0
        for s in remaining_seconds:
            if third_idx < len(qualified['third']):
                round_of_32_teams.append((s, qualified['third'][third_idx]))
                third_idx += 1
        
        # Fill remaining with third-place matchups
        while third_idx + 1 < len(qualified['third']):
            round_of_32_teams.append(
                (qualified['third'][third_idx], qualified['third'][third_idx + 1]))
            third_idx += 2
        
        # Ensure we have 16 matches for Round of 32
        while len(round_of_32_teams) < 16:
            # Fallback: add from qualified teams
            break
        
        results = {
            'round_of_32': [],
            'round_of_16': [],
            'quarter_finals': [],
            'semi_finals': [],
            'third_place': None,
            'final': None,
            'champion': None,
        }
        
        # Round of 32
        r32_winners = []
        for team_a, team_b in round_of_32_teams[:16]:
            winner, ga, gb = self.simulate_match(team_a, team_b, allow_draw=False)
            r32_winners.append(winner)
            results['round_of_32'].append({
                'team_a': team_a, 'team_b': team_b, 
                'winner': winner, 'score': f'{ga}-{gb}'
            })
        
        # Round of 16
        r16_winners = []
        for i in range(0, len(r32_winners) - 1, 2):
            if i + 1 < len(r32_winners):
                winner, ga, gb = self.simulate_match(
                    r32_winners[i], r32_winners[i+1], allow_draw=False)
                r16_winners.append(winner)
                results['round_of_16'].append({
                    'team_a': r32_winners[i], 'team_b': r32_winners[i+1],
                    'winner': winner, 'score': f'{ga}-{gb}'
                })
        
        # Quarter-finals
        qf_winners = []
        for i in range(0, len(r16_winners) - 1, 2):
            if i + 1 < len(r16_winners):
                winner, ga, gb = self.simulate_match(
                    r16_winners[i], r16_winners[i+1], allow_draw=False)
                qf_winners.append(winner)
                results['quarter_finals'].append({
                    'team_a': r16_winners[i], 'team_b': r16_winners[i+1],
                    'winner': winner, 'score': f'{ga}-{gb}'
                })
        
        # Semi-finals
        sf_winners = []
        sf_losers = []
        for i in range(0, len(qf_winners) - 1, 2):
            if i + 1 < len(qf_winners):
                winner, ga, gb = self.simulate_match(
                    qf_winners[i], qf_winners[i+1], allow_draw=False)
                loser = qf_winners[i+1] if winner == qf_winners[i] else qf_winners[i]
                sf_winners.append(winner)
                sf_losers.append(loser)
                results['semi_finals'].append({
                    'team_a': qf_winners[i], 'team_b': qf_winners[i+1],
                    'winner': winner, 'score': f'{ga}-{gb}'
                })
        
        # Third place match
        if len(sf_losers) >= 2:
            winner, ga, gb = self.simulate_match(
                sf_losers[0], sf_losers[1], allow_draw=False)
            results['third_place'] = {
                'team_a': sf_losers[0], 'team_b': sf_losers[1],
                'winner': winner, 'score': f'{ga}-{gb}'
            }
        
        # Final
        if len(sf_winners) >= 2:
            winner, ga, gb = self.simulate_match(
                sf_winners[0], sf_winners[1], allow_draw=False)
            results['final'] = {
                'team_a': sf_winners[0], 'team_b': sf_winners[1],
                'winner': winner, 'score': f'{ga}-{gb}'
            }
            results['champion'] = winner
        
        return results
    
    def run_simulation(self, n_simulations: int = 10000,
                       verbose: bool = True) -> pd.DataFrame:
        """
        Jalankan Monte Carlo simulation lengkap.
        
        Args:
            n_simulations: Jumlah simulasi
            verbose: Print progress
            
        Returns:
            DataFrame dengan probabilitas per tim
        """
        team_stats = defaultdict(lambda: {
            'group_stage_exit': 0,
            'round_of_32': 0,
            'round_of_16': 0,
            'quarter_finals': 0,
            'semi_finals': 0,
            'runner_up': 0,
            'third_place': 0,
            'champion': 0,
        })
        
        for sim in range(n_simulations):
            if verbose and (sim + 1) % 1000 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}...")
            
            # 1. Group stage
            standings = self.simulate_group_stage(as_dataframe=False)
            qualified = self.get_qualified_teams(standings)
            
            all_qualified = set(qualified['first'] + qualified['second'] + 
                              qualified['third'])
            all_teams = set()
            for teams in self.groups.values():
                all_teams.update(teams)
            
            # Mark group stage exits
            for team in all_teams - all_qualified:
                team_stats[team]['group_stage_exit'] += 1
            
            for team in all_qualified:
                team_stats[team]['round_of_32'] += 1
            
            # 2. Knockout stage
            results = self.simulate_knockout(qualified, standings)
            
            # Track progress
            r16_teams = {m['winner'] for m in results['round_of_16']}
            qf_teams = {m['winner'] for m in results['quarter_finals']}
            sf_teams = {m['winner'] for m in results['semi_finals']}
            
            for team in r16_teams:
                team_stats[team]['round_of_16'] += 1
            for team in qf_teams:
                team_stats[team]['quarter_finals'] += 1
            for team in sf_teams:
                team_stats[team]['semi_finals'] += 1
            
            if results['final']:
                finalist_a = results['final']['team_a']
                finalist_b = results['final']['team_b']
                champion = results['champion']
                runner_up = finalist_b if champion == finalist_a else finalist_a
                
                team_stats[champion]['champion'] += 1
                team_stats[runner_up]['runner_up'] += 1
            
            if results['third_place']:
                team_stats[results['third_place']['winner']]['third_place'] += 1
        
        # Convert to probabilities
        rows = []
        for team, stats in team_stats.items():
            row = {'Team': team}
            for key, count in stats.items():
                row[f'p_{key}'] = count / n_simulations
            rows.append(row)
        
        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values('p_champion', ascending=False).reset_index(drop=True)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Top 10 Most Likely Champions:")
            print(f"{'='*50}")
            for _, row in result_df.head(10).iterrows():
                print(f"  {row['Team']:20s} {row['p_champion']*100:5.1f}%")
        
        return result_df
