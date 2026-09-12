"""
Check whether the YAML annotation's expected image resolution matches
the actual RGB image's pixel dimensions. A mismatch would explain a
consistent directional/scale offset in all projected 2D points.
"""

import cv2
import yaml
from warehouse_perception.dataset.loaders import load_scene

scenes_dir = "data/raw/cepb/scenes_dev"
scene = load_scene(scenes_dir, scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))
print("RGB image path:", scene.rgb_path)
print("RGB image shape (h, w, c):", img.shape)

with open(f"{scenes_dir}/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

def find_keys(d, keys, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if any(key in str(k).lower() for key in keys):
                print(f"{path}/{k}: {v}")
            find_keys(v, keys, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d[:1]):
            find_keys(v, keys, f"{path}[{i}]")

find_keys(raw, ["width", "height", "resolution", "intrinsic", "camera_matrix", "fx", "fy", "cx", "cy"])
