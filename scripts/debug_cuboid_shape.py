"""
Check if the cuboid's 3D dimensions and quaternion look sane, and whether
projected edge lengths match expected object size. A stretched box with
one anchored corner suggests a dimension or rotation bug, not translation.
"""

import numpy as np
import yaml

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

obj = raw["objects"]["40-Pringles"]
cuboid = np.array(obj["cuboid"])
position = np.array(obj["position"])
quat = np.array(obj["quaternion"])

print("position:", position)
print("quaternion:", quat, "norm:", np.linalg.norm(quat))
print("cuboid points (3D):")
print(cuboid)

edge_lengths_3d = [np.linalg.norm(cuboid[i] - cuboid[j]) for i, j in [(0,1),(1,2),(2,3),(0,4)]]
print("sample 3D edge lengths:", edge_lengths_3d)

proj = np.array(obj["projected_cuboid"])
edge_lengths_2d = [np.linalg.norm(proj[i] - proj[j]) for i, j in [(0,1),(1,2),(2,3),(0,4)]]
print("sample 2D projected edge lengths (pixels):", edge_lengths_2d)
