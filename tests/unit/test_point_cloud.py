"""
Unit tests for geometry/point_cloud.py.

Why this file exists: point_cloud.py encodes real, measured failure-mode
fixes (dual loader, scale correction). These tests lock in that behavior
so future changes cannot silently break it (regression protection), per
the project's testing requirements.
"""

import os
import pytest
from warehouse_perception.geometry.point_cloud import load_mesh, mesh_to_point_cloud

CEPB_MODELS = "data/raw/cepb/models/models"


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(f"CEPB model not present at {path}")


def test_load_mesh_open3d_direct():
    path = os.path.join(CEPB_MODELS, "33-Key/Key.obj")
    _skip_if_missing(path)
    result = load_mesh(path)
    assert result.loader_used == "open3d"
    assert result.max_extent_m < 0.2
    assert not result.scale_corrected


def test_load_mesh_trimesh_fallback():
    path = os.path.join(CEPB_MODELS, "22-E27_adapter/E27_adapter.obj")
    _skip_if_missing(path)
    result = load_mesh(path)
    assert result.loader_used == "trimesh_fallback"


def test_load_mesh_scale_correction_applied():
    path = os.path.join(CEPB_MODELS, "1-Cheez-it/textured.obj")
    _skip_if_missing(path)
    result = load_mesh(path)
    assert result.scale_corrected is True
    assert result.max_extent_m < 2.0


def test_load_mesh_known_unresolved_not_autocorrected():
    path = os.path.join(CEPB_MODELS, "30-Stabilo_OHPen/stabilo_box.obj")
    _skip_if_missing(path)
    result = load_mesh(path)
    assert result.scale_corrected is False
    assert result.max_extent_m > 2.0


def test_load_mesh_dae_format():
    path = os.path.join(CEPB_MODELS, "11-Padlock/Padlock.dae")
    _skip_if_missing(path)
    result = load_mesh(path)
    assert result.loader_used == "trimesh_fallback_dae"


def test_mesh_to_point_cloud_sample_count():
    path = os.path.join(CEPB_MODELS, "33-Key/Key.obj")
    _skip_if_missing(path)
    result = load_mesh(path)
    pcd = mesh_to_point_cloud(result.mesh, num_points=512)
    assert len(pcd.points) == 512
    assert pcd.has_normals()
