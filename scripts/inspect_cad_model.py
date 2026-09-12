"""
Standalone script to verify a single CEPB CAD model loads correctly.

Why this exists: CEPB's OBJ files are inconsistent — some load fine with
Open3D's triangle mesh reader, others use face topology Open3D's basic
reader rejects (observed: "non-triangle primitive geometry" on at least
one real file). This script documents and tests a fallback strategy:
try Open3D first, fall back to trimesh (more permissive OBJ parser) and
convert to an Open3D mesh if Open3D returns zero vertices.

This dual-loader logic will move into geometry/point_cloud.py once
validated here.
"""

import sys
import numpy as np
import open3d as o3d
import trimesh


def load_mesh_with_fallback(obj_path: str):
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    if len(mesh.vertices) > 0:
        return mesh, "open3d"

    tm = trimesh.load(obj_path, force="mesh")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh, "trimesh_fallback"


def inspect_model(obj_path: str) -> None:
    mesh, loader_used = load_mesh_with_fallback(obj_path)

    print(f"File: {obj_path}")
    print(f"Loader used: {loader_used}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Has vertex normals: {mesh.has_vertex_normals()}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"Bounding box extent (x,y,z) in meters: {extent}")

    if len(mesh.vertices) == 0:
        print("WARNING: Both loaders failed — file may be corrupt.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw/cepb/models/models/33-Key/Key.obj"
    inspect_model(path)
