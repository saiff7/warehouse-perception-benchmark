"""
Data schemas for CEPB scene data.

Why this file exists:
This defines the typed shape of a single CEPB scene, independent of how
files are stored on disk or parsed from YAML. Every other module (dataset
loaders, detection evaluation, depth evaluation, failure analysis) depends
on these dataclasses instead of raw dicts/YAML, so a change in file format
only requires updating loaders.py, not every downstream consumer.

Confirmed real structure (from data/raw/cepb/scenes_dev, 5-scene sample):
- One annotation YAML per (scene_id, camera_id): GT_<camera>_camera_<scene_id>.yaml
- YAML contains one entry per visible-or-occluded object in the scene, with:
  position (3-vec), quaternion (4-vec, wxyz), cuboid (8x3 3D corners),
  projected_cuboid (8x2 pixel coords), 2D_centroid (2-vec pixel coords),
  visibility (float 0-1, 0 = fully occluded, 1 = fully visible).
- RGB images exist for 3 lighting conditions (directional, point, spot)
  per camera per scene.
- Depth and normals images exist ONLY for the directional lighting
  condition (confirmed empirically -- point/spot depth files do not exist).
- Segmentation mask exists once per camera per scene (lighting-independent).

This does not belong in loaders.py because schema (what the data IS) and
loading (how we GET it from disk) are separate concerns -- schema is
reused by tests and by any future loader (e.g. a different CEPB subset
format) without change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ObjectAnnotation:
    """Ground-truth pose and visibility for a single object in a scene."""
    object_id: str
    position: tuple
    quaternion: tuple
    cuboid_3d: list
    cuboid_2d: list
    centroid_2d: tuple
    visibility: float


@dataclass
class SceneAnnotation:
    """All object annotations for one camera view of one scene."""
    scene_id: str
    camera_id: str
    objects: list


@dataclass
class Scene:
    """A single loadable scene: image paths + annotation, for one camera.

    Depth and normals paths are Optional because they only exist for the
    directional lighting condition (measured fact about this dataset).
    """
    scene_id: str
    camera_id: str
    lighting: str
    rgb_path: Path
    depth_path: Optional[Path]
    normals_path: Optional[Path]
    segmentation_path: Path
    annotation: SceneAnnotation
