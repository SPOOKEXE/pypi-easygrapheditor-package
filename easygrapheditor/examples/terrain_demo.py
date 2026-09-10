"""Terrain demo stub (M2 target, mirrors plans/ 15-41-09).

Lists terrain nodes; headless run of implemented nodes only.
Full numpy terrain (Noise/Ridged/Warp/...) lands in M2.
Run: uv run easygrapheditor/examples/terrain_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easygrapheditor.ui.library import search_nodes

for q in ["noise", "erode", "colourise", "arithmetic", "staged"]:
    print(f"query={q!r}:")
    for n in search_nodes(q):
        print(f"  - {n.title} ({n.type_id}) [{n.category}]: {n.description}")

print("\nM2 TODO: numpy Field payloads + terrain node fns.")
