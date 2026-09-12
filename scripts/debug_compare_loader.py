"""
Compare Pringles cuboid_3d as read via the loader vs raw YAML, since the
manual single-object test worked but the full-scene loader-based render
regressed completely.
"""

import numpy as np
import yaml
from warehouse_perception.dataset.loaders import load_scene
from warehouse_perception.dataset.projection import project_points

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
target = next(o for o in scene.annotation.objects if o.object_id == "40-Pringles")
print("Loader cuboid_3d:", np.array(target.cuboid_3d))

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)
print("Raw YAML cuboid:", np.array(raw["objects"]["40-Pringles"]["cuboid"]))

proj = project_points(np.array(target.cuboid_3d), camera_id="left")
print("Projected via loader path:", proj)
