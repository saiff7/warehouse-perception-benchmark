"""
Unit tests for dataset/loaders.py and dataset/schemas.py.

Why this file exists: loaders.py is the entry point for all real scene
data used by detection, depth evaluation, and failure analysis. These
tests lock in correct parsing of CEPB's real file layout (confirmed
empirically in scenes_dev) so future changes cannot silently corrupt
scene loading.
"""

import os
import pytest
from warehouse_perception.dataset.loaders import load_scene, list_available_scene_ids

SCENES_DIR = "data/raw/cepb/scenes_dev"


def _skip_if_missing():
    if not os.path.exists(SCENES_DIR):
        pytest.skip(f"CEPB dev scenes not present at {SCENES_DIR}")


def test_list_available_scene_ids():
    _skip_if_missing()
    ids = list_available_scene_ids(SCENES_DIR)
    assert ids == ["1", "2", "3", "4", "5"]


def test_load_scene_directional_has_depth_and_normals():
    _skip_if_missing()
    scene = load_scene(SCENES_DIR, scene_id="1", camera_id="left", lighting="directional")
    assert scene.depth_path is not None
    assert scene.normals_path is not None
    assert scene.rgb_path.exists()
    assert scene.depth_path.exists()
    assert scene.segmentation_path.exists()


def test_load_scene_point_lighting_has_no_depth():
    _skip_if_missing()
    scene = load_scene(SCENES_DIR, scene_id="1", camera_id="left", lighting="point")
    assert scene.depth_path is None
    assert scene.normals_path is None
    assert scene.rgb_path.exists()


def test_load_scene_annotation_parses_objects():
    _skip_if_missing()
    scene = load_scene(SCENES_DIR, scene_id="1", camera_id="left")
    assert len(scene.annotation.objects) > 0
    obj = scene.annotation.objects[0]
    assert isinstance(obj.object_id, str)
    assert len(obj.position) == 3
    assert len(obj.quaternion) == 4
    assert len(obj.cuboid_3d) == 8
    assert 0.0 <= obj.visibility <= 1.0


def test_load_scene_missing_file_raises():
    _skip_if_missing()
    with pytest.raises(FileNotFoundError):
        load_scene(SCENES_DIR, scene_id="9999", camera_id="left")


def test_all_cameras_loadable_for_scene_1():
    _skip_if_missing()
    for camera_id in ("left", "middle", "right"):
        scene = load_scene(SCENES_DIR, scene_id="1", camera_id=camera_id)
        assert scene.camera_id == camera_id
        assert len(scene.annotation.objects) > 0
