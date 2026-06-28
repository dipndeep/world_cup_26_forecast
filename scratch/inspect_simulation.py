import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
notebook_path = Path(__file__).resolve().parent.parent / 'notebooks' / '04_tournament_simulation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"--- Cell {idx} ---")
        print(source)
        print()
