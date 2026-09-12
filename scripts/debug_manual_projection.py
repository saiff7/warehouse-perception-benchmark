"""
Manually reproject Pringles cuboid using CEPB's documented left-to-right-
handed conversion steps and camera intrinsics/extrinsics, instead of
trusting the possibly-unreliable baked-in projected_cuboid field.
"""

import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation as R

K = np.array([[3200, 0, 1024], [0, 3200, 768], [0, 0, 1]], dtype=float)

H_world_to_left_camera = np.array([
    [1, 0, 0, 0.03],
    [0, 0, -1, 1.25],
    [0, 1, 0, 0.028],
    [0, 0, 0, 1],
])

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

obj = raw["objects"]["40-Pringles"]
cuboid_3d = np.array(obj["cuboid"])

def project_point(p_world):
    p_h = np.append(p_world, 1.0)
    p_h = [p_h[0], -p_h[1], p_h[2], p_h[3]]
    p_cam = np.linalg.inv(H_world_to_left_camera).dot(p_h)
    p_cam_3 = p_cam[:3]
    p_img = K.dot(p_cam_3)
    p_img = p_img[:2] / p_img[2]
    return p_img

projected = np.array([project_point(p) for p in cuboid_3d])
print("Manually projected points:")
print(projected)

img = cv2.imread("data/raw/cepb/scenes_dev/left_camera_directional_light_scene_1_rgb.png")
CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]
pts = projected.astype(np.int32)
for i, j in CUBOID_EDGES:
    cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 255), 3)
cv2.imwrite("experiments/results/manual_projection.png", img)
print("saved")
