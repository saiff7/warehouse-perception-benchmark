"""
Test alternate sign conventions for the left camera's x/z position offset
in the extrinsic matrix, to close the small remaining gap between the
projected cuboid and the true can position seen in v2.
"""

import numpy as np
import yaml
import cv2

K = np.array([[3200, 0, 1024], [0, 3200, 768], [0, 0, 1]], dtype=float)

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)
obj = raw["objects"]["40-Pringles"]
cuboid_3d = np.array(obj["cuboid"])

CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]

def make_H(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 0, -1, ty],
        [0, 1, 0, tz],
        [0, 0, 0, 1],
    ])

def project_point(p_world, H):
    p_flipped = np.array([p_world[0], -p_world[1], p_world[2]])
    p_h = np.append(p_flipped, 1.0)
    p_cam = np.linalg.inv(H).dot(p_h)
    p_img = K.dot(p_cam[:3])
    return p_img[:2] / p_img[2]

variants = {
    "tx=-0.03": make_H(-0.03, 1.25, -0.028),
    "tx=+0.03_tz=+0.028": make_H(0.03, 1.25, 0.028),
    "tx=-0.03_tz=+0.028": make_H(-0.03, 1.25, 0.028),
}

img_base = cv2.imread("data/raw/cepb/scenes_dev/left_camera_directional_light_scene_1_rgb.png")
colors = {"tx=-0.03": (0,0,255), "tx=+0.03_tz=+0.028": (0,255,0), "tx=-0.03_tz=+0.028": (255,0,0)}

img = img_base.copy()
for name, H in variants.items():
    proj = np.array([project_point(p, H) for p in cuboid_3d])
    pts = proj.astype(np.int32)
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), colors[name], 2)
    print(name, proj[0])

cv2.imwrite("experiments/results/manual_projection_v3.png", img)
print("saved")
