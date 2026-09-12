"""
Print the full top-level and nested key structure of the GT YAML to find
where camera intrinsics / resolution / extrinsics actually live, since a
targeted keyword search found nothing.
"""

import yaml
import json

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

def summarize(d, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        print(indent + "...")
        return
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{indent}{k}: {type(v).__name__}")
            summarize(v, depth + 1, max_depth)
    elif isinstance(d, list):
        print(f"{indent}[list of {len(d)}]")
        if d:
            summarize(d[0], depth + 1, max_depth)

summarize(raw)
