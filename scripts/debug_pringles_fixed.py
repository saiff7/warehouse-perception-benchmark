"""
Redraw the Pringles cuboid using the corrected edge topology recovered
from actual 3D distances, replacing the previously wrong hardcoded
CUBOID_EDGES index assumption.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene

CUBOID_EDGES = [
    (0, 1), (1, 5), (5, 4), (4, 0),
    (2, 3), (3, 7), (7, 6), (6, 2),
    (0, 3), (1, 2), (4, 7), (5, 6),
]

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))

target = next(o for o in scene.annotation.objects if o.object_id == "40-Pringles")
pts = np.array(target.cuboid_2d, dtype=np.int32)
for i, j in CUBOID_EDGES:
    cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 3)
cx, cy = int(target.centroid_2d[0]), int(target.centroid_2d[1])
cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

cv2.imwrite("experiments/results/pringles_fixed.png", img)
print("saved")
