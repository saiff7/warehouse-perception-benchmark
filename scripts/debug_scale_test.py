"""
Test whether cuboid corners are double their true extent by scaling each
object's 8 corners toward its own centroid by 0.5 and re-rendering, to
check if boxes then correctly hug their objects.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene

CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))

for obj in scene.annotation.objects:
    if obj.visibility < 0.05:
        continue
    pts = np.array(obj.cuboid_2d, dtype=np.float64)
    center = pts.mean(axis=0)
    pts_scaled = center + (pts - center) * 0.5
    pts_scaled = pts_scaled.astype(np.int32)
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts_scaled[i]), tuple(pts_scaled[j]), (255, 0, 255), 2)

cv2.imwrite("experiments/results/scale_test.png", img)
print("saved")
