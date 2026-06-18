import json
from pathlib import Path

def main():
    notebook_path = Path(__file__).resolve().parent.parent / 'notebooks' / '04_tournament_simulation.ipynb'
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    updated_cells = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Check for cell 6 (Run one detailed simulation)
            if "sim = WorldCup2026Simulator(wc2026_groups, predict_fn=predict_match_elo, seed=42)" in source:
                new_source = [
                    "# Run one detailed simulation\n",
                    "played_matches_df = pd.read_csv(DATA_DIR / '2026_world_cup_matches.csv')\n",
                    "sim = WorldCup2026Simulator(wc2026_groups, predict_fn=predict_match_elo, seed=42, played_matches_df=played_matches_df)\n",
                    "\n",
                    "# Group stage\n",
                    "print('\U0001f3df\ufe0f  Simulating Group Stage...')\n",
                    "print('=' * 60)\n",
                    "standings = sim.simulate_group_stage()\n",
                    "\n",
                    "for group_name in sorted(standings.keys()):\n",
                    "    df = standings[group_name]\n",
                    "    print(f'\\nGroup {group_name}:')\n",
                    "    print(df[['Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']].to_string(index=False))"
                ]
                cell['source'] = new_source
                updated_cells += 1
                print("Updated single simulation cell in notebook")
                
            # Check for cell 9 (Run Monte Carlo simulation)
            elif "mc_sim = WorldCup2026Simulator(wc2026_groups, predict_fn=predict_match_elo, seed=2026)" in source:
                new_source = [
                    "# Run Monte Carlo simulation\n",
                    "print('\U0001f3b2 Running Monte Carlo Simulation (10,000 iterations)...')\n",
                    "print('   This may take a few minutes...\\n')\n",
                    "\n",
                    "played_matches_df = pd.read_csv(DATA_DIR / '2026_world_cup_matches.csv')\n",
                    "mc_sim = WorldCup2026Simulator(wc2026_groups, predict_fn=predict_match_elo, seed=2026, played_matches_df=played_matches_df)\n",
                    "mc_results = mc_sim.run_simulation(n_simulations=10000, verbose=True)"
                ]
                cell['source'] = new_source
                updated_cells += 1
                print("Updated Monte Carlo simulation cell in notebook")
                
    if updated_cells > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {updated_cells} cells in 04_tournament_simulation.ipynb")
    else:
        print("No cells matched pattern for update")

if __name__ == '__main__':
    main()
