"""
For every object in the scene, sample the segmentation color at its own
2D_centroid to build a color->object_name mapping, then check which
object the Pringles' centroid color and corner colors actually belong to.
"""

import cv2
import numpy as np
import yaml

seg = cv2.imread("data/raw/cepb/scenes_dev/left_camera_scene_1_segmentation.png")

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

for name, obj in raw["objects"].items():
    cx, cy = obj["2D_centroid"]
    cx, cy = int(cx), int(cy)
    color = seg[cy, cx]
    print(f"{name}: centroid=({cx},{cy}) color={color}")
