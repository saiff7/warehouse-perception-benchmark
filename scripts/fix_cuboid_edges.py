"""
Automatically determine correct cuboid edges by finding, for each vertex,
its 3 nearest neighbors among the other 7 points (a box has exactly 3
adjacent corners per vertex). This replaces the hardcoded CUBOID_EDGES
index assumption, which was proven wrong by the mismatched edge lengths.
"""

import numpy as np
import yaml

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

cuboid = np.array(raw["objects"]["40-Pringles"]["cuboid"])

edges = set()
for i in range(8):
    dists = [(np.linalg.norm(cuboid[i] - cuboid[j]), j) for j in range(8) if j != i]
    dists.sort()
    for _, j in dists[:3]:
        edges.add(tuple(sorted((i, j))))

print("Recovered edges:", sorted(edges))
print("Count:", len(edges))
