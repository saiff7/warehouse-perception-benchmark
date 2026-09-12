"""
Calibrate tz sign for middle and right cameras the same way we did for
left: project Pringles cuboid with both signs and overlay on each
camera's own RGB image to determine the correct convention empirically.
"""

import numpy as np
import yaml
import cv2

K = np.array([[3200, 0, 1024], [0, 3200, 768], [0, 0, 1]], dtype=float)

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

cameras = {
    "middle": {"file": "GT_middle_camera_1.yaml", "rgb": "middle_camera_directional_light_scene_1_rgb.png", "tx": 0.0, "tz_options": [0.0]},
    "right": {"file": "GT_right_camera_1.yaml", "rgb": "right_camera_directional_light_scene_1_rgb.png", "tx": 0.03, "tz_options": [-0.028, 0.028]},
}

for cam_name, cfg in cameras.items():
    with open(f"data/raw/cepb/scenes_dev/{cfg['file']}") as f:
        raw = yaml.safe_load(f)
    cuboid_3d = np.array(raw["objects"]["40-Pringles"]["cuboid"])
    img_base = cv2.imread(f"data/raw/cepb/scenes_dev/{cfg['rgb']}")

    for tz in cfg["tz_options"]:
        img = img_base.copy()
        H = make_H(cfg["tx"], 1.25, tz)
        proj = np.array([project_point(p, H) for p in cuboid_3d])
        pts = proj.astype(np.int32)
        for i, j in CUBOID_EDGES:
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 0, 255), 3)
        out = f"experiments/results/calib_{cam_name}_tz_{tz}.png"
        cv2.imwrite(out, img)
        print(f"{cam_name} tz={tz}: first_point={proj[0]}, saved={out}")
