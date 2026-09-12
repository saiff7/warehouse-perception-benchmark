"""
Batch inspection of all CEPB CAD models.

Why this exists: rather than testing objects one at a time, this script
runs the dual-loader logic (Open3D -> trimesh fallback) across every
object folder, logs which loader succeeded, and flags implausible
bounding boxes (unit-scale bugs). Produces a CSV used as real, measured
evidence for docs/methodology.md.
"""

import glob
import os
import csv
import numpy as np
import open3d as o3d
import trimesh


def load_mesh_with_fallback(obj_path: str):
    mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    if len(mesh.vertices) > 0:
        return mesh, "open3d", None

    try:
        tm = trimesh.load(obj_path, force="mesh")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
        mesh.compute_vertex_normals()
        return mesh, "trimesh_fallback", None
    except Exception as e:
        return None, "failed", str(e)


def inspect_all(models_root: str, out_csv: str):
    obj_files = sorted(glob.glob(os.path.join(models_root, "*", "*.obj")))
    rows = []
    for obj_path in obj_files:
        folder = os.path.basename(os.path.dirname(obj_path))
        mesh, loader, error = load_mesh_with_fallback(obj_path)
        if mesh is None:
            rows.append({
                "object": folder, "file": os.path.basename(obj_path),
                "loader": loader, "vertices": 0, "triangles": 0,
                "extent_x": None, "extent_y": None, "extent_z": None,
                "max_extent_m": None, "implausible_scale": True,
                "error": error,
            })
            continue

        bbox = mesh.get_axis_aligned_bounding_box()
        ex, ey, ez = bbox.get_extent()
        max_extent = max(ex, ey, ez)
        rows.append({
            "object": folder, "file": os.path.basename(obj_path),
            "loader": loader, "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "extent_x": ex, "extent_y": ey, "extent_z": ez,
            "max_extent_m": max_extent,
            "implausible_scale": max_extent > 2.0,
            "error": None,
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = inspect_all(
        "data/raw/cepb/models/models",
        "experiments/results/cad_model_inspection.csv",
    )
    total = len(rows)
    open3d_ok = sum(1 for r in rows if r["loader"] == "open3d")
    fallback = sum(1 for r in rows if r["loader"] == "trimesh_fallback")
    failed = sum(1 for r in rows if r["loader"] == "failed")
    implausible = sum(1 for r in rows if r["implausible_scale"])

    print(f"Total models inspected: {total}")
    print(f"Loaded via Open3D directly: {open3d_ok}")
    print(f"Required trimesh fallback: {fallback}")
    print(f"Failed to load entirely: {failed}")
    print(f"Implausible scale (>2m in any dim): {implausible}")
