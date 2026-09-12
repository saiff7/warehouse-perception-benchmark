"""
Mesh loading and point cloud generation for CEPB CAD models.

Why this file exists:
This is the single entry point for turning a CAD model file (OBJ, DAE) into
a usable Open3D geometry (mesh or point cloud) for the rest of the pipeline
(grasp candidate generation, collision checks, visualization). It exists as
its own module because loading is genuinely messy in this dataset -- we
measured real failures (non-triangle OBJ primitives, unit-scale bugs across
39-format files) and centralizing the fix here means every other module
(candidates.py, normals.py) can assume a clean, meter-scale, triangulated
mesh without re-solving these problems.

Does not belong in dataset/loaders.py: that module is for RGB-D scene data
(images + annotations), not standalone CAD geometry -- different data
shape, different failure modes.

Measured findings this module encodes (see experiments/results/cad_model_inspection.csv):
- 32/38 OBJ models load directly via Open3D.
- 6/38 OBJ models require a trimesh fallback (Open3D rejects non-triangle
  face primitives it cannot parse, e.g. stabilo_box.obj).
- 11/38 OBJ models ("textured.obj" filename, YCB-style objects) have
  bounding boxes ~100x too large -- likely authored in cm, loaded as m.
  Dividing extents by 100 produces plausible real-world object sizes.
- 1/40 objects (Padlock) ships as COLLADA (.dae) instead of OBJ.
- 2 objects (Sleeve, Stabilo box) show implausible scale that does NOT
  cleanly resolve with a 100x correction -- flagged as unresolved, not
  silently "fixed".
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import trimesh

logger = logging.getLogger(__name__)

IMPLAUSIBLE_EXTENT_M = 2.0
SCALE_CORRECTION_FACTOR = 100.0
KNOWN_UNRESOLVED_SCALE = {"14-Sleeve", "30-Stabilo_OHPen"}


@dataclass
class LoadedMesh:
    mesh: o3d.geometry.TriangleMesh
    loader_used: str
    scale_corrected: bool
    max_extent_m: float
    object_id: Optional[str] = None


def _mesh_from_trimesh(tm) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def load_mesh(file_path: str) -> LoadedMesh:
    """Load a CAD model file (.obj or .dae) into an Open3D triangle mesh.

    Tries Open3D's native reader first. Falls back to trimesh for OBJ
    files with face topology Open3D cannot parse. Applies a measured
    scale correction if the mesh's bounding box is implausibly large,
    unless the object is a known unresolved case.
    """
    ext = os.path.splitext(file_path)[1].lower()
    object_id = os.path.basename(os.path.dirname(file_path))

    mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)
    loader_used = "open3d_dae" if ext == ".dae" else "open3d"

    if len(mesh.vertices) == 0:
        try:
            tm = trimesh.load(file_path, force="mesh")
            mesh = _mesh_from_trimesh(tm)
            loader_used = "trimesh_fallback" if ext == ".obj" else "trimesh_fallback_dae"
        except Exception as e:
            raise ValueError(f"Failed to load mesh {file_path}: {e}")

    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh {file_path}: no vertices from any loader")

    bbox = mesh.get_axis_aligned_bounding_box()
    max_extent = float(max(bbox.get_extent()))
    scale_corrected = False

    if max_extent > IMPLAUSIBLE_EXTENT_M and object_id not in KNOWN_UNRESOLVED_SCALE:
        mesh.scale(1.0 / SCALE_CORRECTION_FACTOR, center=mesh.get_center())
        bbox = mesh.get_axis_aligned_bounding_box()
        max_extent = float(max(bbox.get_extent()))
        scale_corrected = True
        logger.warning(
            "Applied %.0fx scale correction to %s (was implausibly large)",
            SCALE_CORRECTION_FACTOR, object_id,
        )
    elif max_extent > IMPLAUSIBLE_EXTENT_M:
        logger.warning(
            "%s has implausible scale (%.3fm) and is a KNOWN unresolved "
            "case -- NOT auto-corrected. Inspect manually.",
            object_id, max_extent,
        )

    return LoadedMesh(
        mesh=mesh,
        loader_used=loader_used,
        scale_corrected=scale_corrected,
        max_extent_m=max_extent,
        object_id=object_id,
    )


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, num_points: int = 2048) -> o3d.geometry.PointCloud:
    """Sample a point cloud uniformly from a mesh surface."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.estimate_normals()
    return pcd
