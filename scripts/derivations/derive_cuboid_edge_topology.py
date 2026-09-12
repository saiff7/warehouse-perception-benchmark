"""
Correctly reconstruct 12 box edges from 8 unordered vertices.
Strategy: split into two 4-vertex face groups by connectivity, find the
4-cycle within each face (shortest 4 edges, excluding diagonals), then
match top-to-bottom vertices by nearest long edge.
"""

import numpy as np
import yaml
from itertools import combinations

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

cuboid = np.array(raw["objects"]["40-Pringles"]["cuboid"])
dist = np.linalg.norm(cuboid[:, None] - cuboid[None, :], axis=2)

face_a = [0, 1, 4, 5]
face_b = [2, 3, 6, 7]

def face_cycle(face, dist):
    pairs = list(combinations(face, 2))
    pairs.sort(key=lambda p: dist[p[0], p[1]])
    edges, deg = [], {v: 0 for v in face}
    for i, j in pairs:
        if deg[i] < 2 and deg[j] < 2:
            edges.append((i, j))
            deg[i] += 1
            deg[j] += 1
        if len(edges) == 4:
            break
    return edges

edges = face_cycle(face_a, dist) + face_cycle(face_b, dist)

for i in face_a:
    j = min(face_b, key=lambda x: dist[i, x])
    edges.append((i, j))

print("Final edges:", sorted(set(tuple(sorted(e)) for e in edges)))
print("Count:", len(set(tuple(sorted(e)) for e in edges)))
