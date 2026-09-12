"""
Render all visible objects in scene 1 using the calibrated manual
projection, to confirm the tz=+0.028 fix generalizes beyond Pringles.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene
from warehouse_perception.dataset.projection import project_points

CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))

for obj in scene.annotation.objects:
    if obj.visibility < 0.05:
        continue
    proj = project_points(np.array(obj.cuboid_3d), camera_id="left")
    pts = proj.astype(np.int32)
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2)

cv2.imwrite("experiments/results/final_projection.png", img)
print("saved")
