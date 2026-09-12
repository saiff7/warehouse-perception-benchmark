"""
Test whether cuboid/centroid misalignment is caused by a camera-id mismatch
between the loaded RGB image and its paired annotation file.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene, load_scene_annotation

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

scenes_dir = "data/raw/cepb/scenes_dev"
rgb_scene = load_scene(scenes_dir, scene_id="1", camera_id="left")
img_base = cv2.imread(str(rgb_scene.rgb_path))

for cam in ("left", "middle", "right"):
    img = img_base.copy()
    ann = load_scene_annotation(f"{scenes_dir}/GT_{cam}_camera_1.yaml")
    target = next(o for o in ann.objects if o.object_id == "40-Pringles")
    pts = np.array(target.cuboid_2d, dtype=np.int32)
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 3)
    cx, cy = int(target.centroid_2d[0]), int(target.centroid_2d[1])
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
    out = f"experiments/results/pringles_annotation_{cam}.png"
    cv2.imwrite(out, img)
    print(f"Saved {out}")
