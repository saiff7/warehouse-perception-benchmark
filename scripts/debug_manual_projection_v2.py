"""
Follow CEPB's documented steps in exact order: 1) flip Y sign on raw
cuboid points, 2) transform into camera frame via inverse extrinsic
matrix, 3) apply intrinsics K to project to image space.
"""

import numpy as np
import yaml
import cv2

K = np.array([[3200, 0, 1024], [0, 3200, 768], [0, 0, 1]], dtype=float)

H_world_to_left_camera = np.array([
    [1, 0, 0, -0.03],
    [0, 0, -1, 1.25],
    [0, 1, 0, -0.028],
    [0, 0, 0, 1],
])

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

obj = raw["objects"]["40-Pringles"]
cuboid_3d = np.array(obj["cuboid"])

def project_point(p_world):
    p_flipped = np.array([p_world[0], -p_world[1], p_world[2]])
    p_h = np.append(p_flipped, 1.0)
    p_cam = np.linalg.inv(H_world_to_left_camera).dot(p_h)
    p_img = K.dot(p_cam[:3])
    return p_img[:2] / p_img[2]

projected = np.array([project_point(p) for p in cuboid_3d])
print("Projected points (v2):")
print(projected)

img = cv2.imread("data/raw/cepb/scenes_dev/left_camera_directional_light_scene_1_rgb.png")
CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]
pts = projected.astype(np.int32)
for i, j in CUBOID_EDGES:
    cv2.line(img, tuple(pts[i]), tuple(pts[j]), (255, 128, 0), 3)
cv2.imwrite("experiments/results/manual_projection_v2.png", img)
print("saved")
