"""
Validate get_cuboid_edges() against several objects of different shapes
to confirm the face_a/face_b grouping assumption holds generally, not
just for the Pringles can.
"""

import yaml
from warehouse_perception.dataset.geometry import get_cuboid_edges

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

for name in ["40-Pringles", "32-Wine_Glass", "37-T-shirt", "38-Rolodex_Jumbo_Pencil_Cup"]:
    cuboid = raw["objects"][name]["cuboid"]
    edges = get_cuboid_edges(cuboid)
    print(f"{name}: {len(edges)} edges -> {edges}")
