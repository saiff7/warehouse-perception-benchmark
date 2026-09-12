"""
Isolate and draw a single object's cuboid wireframe for debugging.

Why this exists: with 7 overlapping boxes in a cluttered scene, visual
verification is unreliable. Isolating one high-visibility, simple-shaped
object (Pringles can, visibility=0.982) removes clutter as a confound and
lets us verify projection correctness against one unambiguous ground truth.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))

target = next(o for o in scene.annotation.objects if o.object_id == "40-Pringles")
print("position:", target.position)
print("cuboid_3d[0]:", target.cuboid_3d[0])
print("cuboid_2d:", target.cuboid_2d)
print("centroid_2d:", target.centroid_2d)

pts = np.array(target.cuboid_2d, dtype=np.int32)
for i, j in CUBOID_EDGES:
    cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 3)
cx, cy = int(target.centroid_2d[0]), int(target.centroid_2d[1])
cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

cv2.imwrite("experiments/results/pringles_only.png", img)
print("saved")
