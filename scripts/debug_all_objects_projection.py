"""
Render each visible object's projected cuboid individually (one color
per object) to identify which specific objects are projecting incorrectly,
since Pringles alone is confirmed correct.
"""

import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene
from warehouse_perception.dataset.projection import project_points

CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,255)]

for idx, obj in enumerate([o for o in scene.annotation.objects if o.visibility >= 0.05]):
    img = cv2.imread(str(scene.rgb_path))
    proj = project_points(np.array(obj.cuboid_3d), camera_id="left")
    pts = proj.astype(np.int32)
    color = colors[idx % len(colors)]
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 3)
    cv2.putText(img, obj.object_id, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    out = f"experiments/results/obj_{idx}_{obj.object_id}.png"
    cv2.imwrite(out, img)
    print(f"{obj.object_id}: visibility={obj.visibility:.2f}, first_point={proj[0]}, saved={out}")
