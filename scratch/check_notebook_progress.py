import json
from pathlib import Path
import sys

# Ensure stdout uses utf-8
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path(__file__).resolve().parent.parent / 'notebooks' / '02_feature_engineering.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"--- Cell {idx} ---")
        print(source)
        print()
