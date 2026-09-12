"""
Sample segmentation color at the object's own 2D_centroid coordinate
(from YAML, in full-resolution pixel space) instead of guessing from a
resized screenshot, then find that color's true silhouette extent.
"""

import cv2
import numpy as np
import yaml

seg = cv2.imread("data/raw/cepb/scenes_dev/left_camera_scene_1_segmentation.png")

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

obj = raw["objects"]["40-Pringles"]
cx, cy = obj["2D_centroid"]
cx, cy = int(cx), int(cy)
print("centroid pixel:", cx, cy)
print("color at centroid:", seg[cy, cx])

for dx, dy in [(-20,0),(20,0),(0,-20),(0,20),(0,0)]:
    print(f"offset ({dx},{dy}):", seg[cy+dy, cx+dx])

proj = np.array(obj["projected_cuboid"])
print("\nColors at each projected_cuboid corner:")
for i, (px, py) in enumerate(proj):
    px, py = int(px), int(py)
    print(f"corner {i} ({px},{py}):", seg[py, px])
