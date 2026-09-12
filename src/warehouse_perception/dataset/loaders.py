"""
Loaders for CEPB scene data on disk.

Why this file exists:
Translates the real file/folder layout of an extracted CEPB subset (as
confirmed in data/raw/cepb/scenes_dev) into typed Scene/SceneAnnotation
objects defined in schemas.py. This is the only module that should know
about file naming conventions -- if CEPB changes naming or we add a new
dataset, only this file changes.
"""

import glob
import os
import re
from pathlib import Path

import yaml

from warehouse_perception.dataset.schemas import (
    ObjectAnnotation,
    Scene,
    SceneAnnotation,
)

CAMERAS = ("left", "middle", "right")
LIGHTINGS = ("directional", "point", "spot")


def _parse_annotation_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        text = f.read()
    data = yaml.safe_load(text)
    return data


def load_scene_annotation(yaml_path: str) -> SceneAnnotation:
    yaml_path = Path(yaml_path)
    match = re.match(r"GT_(\w+)_camera_(\d+)\.yaml", yaml_path.name)
    if not match:
        raise ValueError(f"Unexpected annotation filename: {yaml_path.name}")
    camera_id, scene_id = match.group(1), match.group(2)

    data = _parse_annotation_yaml(yaml_path)
    objects_dict = data.get("objects", {}) if data else {}

    objects = []
    for obj_id, fields in objects_dict.items():
        objects.append(ObjectAnnotation(
            object_id=obj_id,
            position=tuple(fields["position"]),
            quaternion=tuple(fields["quaternion"]),
            cuboid_3d=fields["cuboid"],
            cuboid_2d=fields["projected_cuboid"],
            centroid_2d=tuple(fields["2D_centroid"]),
            visibility=float(fields["visibility"][0])
            if isinstance(fields["visibility"], list)
            else float(fields["visibility"]),
        ))

    return SceneAnnotation(scene_id=scene_id, camera_id=camera_id, objects=objects)


def load_scene(
    scenes_dir: str,
    scene_id: str,
    camera_id: str,
    lighting: str = "directional",
) -> Scene:
    scenes_dir = Path(scenes_dir)
    prefix = f"{camera_id}_camera_{lighting}_light_scene_{scene_id}"

    rgb_path = scenes_dir / f"{prefix}_rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB file: {rgb_path}")

    depth_path = scenes_dir / f"{prefix}_depth.png"
    normals_path = scenes_dir / f"{prefix}_normals.png"
    if lighting != "directional" or not depth_path.exists():
        depth_path = None
    if lighting != "directional" or not normals_path.exists():
        normals_path = None

    segmentation_path = scenes_dir / f"{camera_id}_camera_scene_{scene_id}_segmentation.png"
    if not segmentation_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {segmentation_path}")

    yaml_path = scenes_dir / f"GT_{camera_id}_camera_{scene_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {yaml_path}")
    annotation = load_scene_annotation(str(yaml_path))

    return Scene(
        scene_id=scene_id,
        camera_id=camera_id,
        lighting=lighting,
        rgb_path=rgb_path,
        depth_path=depth_path,
        normals_path=normals_path,
        segmentation_path=segmentation_path,
        annotation=annotation,
    )


def list_available_scene_ids(scenes_dir: str) -> list:
    scenes_dir = Path(scenes_dir)
    pattern = str(scenes_dir / "GT_left_camera_*.yaml")
    ids = []
    for path in glob.glob(pattern):
        match = re.match(r"GT_left_camera_(\d+)\.yaml", os.path.basename(path))
        if match:
            ids.append(match.group(1))
    return sorted(ids, key=int)
