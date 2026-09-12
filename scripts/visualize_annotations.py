"""
Standalone script to visually verify scene annotations against RGB image.

Why this exists: numeric correctness (tests passing) does not guarantee
annotations are semantically correct. Drawing the true 3D cuboid wireframe
(not a convex hull) over the RGB image is the correct way to visually
verify pose/projection correctness.

Cuboid corner ordering follows the NVIDIA DOPE convention (matches CEPB's
field names: cuboid, projected_cuboid, 2D_centroid):
  0: front-top-right     4: rear-top-right
  1: front-top-left      5: rear-top-left
  2: front-bottom-left   6: rear-bottom-left
  3: front-bottom-right  7: rear-bottom-right

Wireframe = front face (0-1-2-3-0) + rear face (4-5-6-7-4) +
            4 connecting edges (0-4, 1-5, 2-6, 3-7).
This is a fixed 12-edge topology, NOT a convex hull -- a convex hull over
projected 3D corners produces a bloated, meaningless outer envelope when
the box is viewed at an angle (this is exactly the bug we observed).
"""

import sys
import cv2
import numpy as np

from warehouse_perception.dataset.loaders import load_scene
from warehouse_perception.dataset.projection import project_points

from warehouse_perception.dataset.projection import CUBOID_EDGES


def draw_cuboid_wireframe(img, corners_2d, color=(0, 255, 0), thickness=2):
    pts = np.array(corners_2d, dtype=np.int32)
    for i, j in CUBOID_EDGES:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)


def draw_scene_annotations(scenes_dir: str, scene_id: str, camera_id: str, out_path: str):
    scene = load_scene(scenes_dir, scene_id=scene_id, camera_id=camera_id)
    img = cv2.imread(str(scene.rgb_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {scene.rgb_path}")

    drawn = 0
    for obj in scene.annotation.objects:
        if obj.visibility < 0.05:
            continue
        drawn += 1

        proj_cuboid = project_points(np.array(obj.cuboid_3d), camera_id=camera_id)
        draw_cuboid_wireframe(img, proj_cuboid)

        centroid_3d = np.array(obj.cuboid_3d).mean(axis=0)
        proj_centroid = project_points(centroid_3d[None, :], camera_id=camera_id)[0]
        cx, cy = int(proj_centroid[0]), int(proj_centroid[1])
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{obj.object_id} ({obj.visibility:.2f})"
        cv2.putText(img, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to {out_path}")
    print(f"Drawn objects (visibility >= 0.05): {drawn} / {len(scene.annotation.objects)} total")


if __name__ == "__main__":
    scenes_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cepb/scenes_dev"
    scene_id = sys.argv[2] if len(sys.argv) > 2 else "1"
    camera_id = sys.argv[3] if len(sys.argv) > 3 else "left"
    out_path = f"experiments/results/annotated_scene_{scene_id}_{camera_id}_v2.png"
    draw_scene_annotations(scenes_dir, scene_id, camera_id, out_path)
