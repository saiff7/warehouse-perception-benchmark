"""
Overlay the Pringles cuboid/centroid on the segmentation mask instead of
the RGB image. Segmentation is rendered from the same camera and gives an
unambiguous per-pixel ground truth, ruling out RGB/GT mismatch.
"""

import cv2
import numpy as np

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

seg = cv2.imread("data/raw/cepb/scenes_dev/left_camera_scene_1_segmentation.png")
print("Segmentation shape:", seg.shape)
print("Unique colors sample:", np.unique(seg.reshape(-1, 3), axis=0)[:10])

cuboid_2d = [[1336, 514], [1234, 305], [732, 675], [814, 860], [1196, 531], [1090, 309], [582, 698], [666, 893]]
centroid_2d = (938, 611)

pts = np.array(cuboid_2d, dtype=np.int32)
for i, j in CUBOID_EDGES:
    cv2.line(seg, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 3)
cv2.circle(seg, centroid_2d, 8, (0, 0, 255), -1)

cv2.imwrite("experiments/results/pringles_on_segmentation.png", seg)
print("saved")
