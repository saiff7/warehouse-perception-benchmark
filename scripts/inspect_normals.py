"""
Validates point-cloud-estimated normals against mesh-derived ground-truth
normals, for a single CAD model.

Why this exists: Open3D's estimate_normals() on a point cloud has no
inherent "outward" reference and can produce flipped normals on complex
geometry. Meshes, however, encode a ground-truth normal per face via
triangle winding order. This script measures how often the point-cloud
estimate disagrees with the mesh-derived ground truth, using the dot
product between the two normals sampled at the same surface locations.

A dot product near +1 means agreement (correct direction).
A dot product near -1 means the estimated normal is flipped.
"""

import sys
import numpy as np
from warehouse_perception.geometry.point_cloud import load_mesh, mesh_to_point_cloud


def validate_normals(obj_path: str, num_points: int = 2000) -> dict:
    loaded = load_mesh(obj_path)
    mesh = loaded.mesh
    mesh.compute_triangle_normals()

    pcd = mesh.sample_points_uniformly(
        number_of_points=num_points, use_triangle_normal=True
    )
    ground_truth_normals = np.asarray(pcd.normals)

    pcd_for_estimate = pcd
    pcd_for_estimate.estimate_normals()
    pcd_for_estimate.orient_normals_consistent_tangent_plane(k=10)
    estimated_normals = np.asarray(pcd_for_estimate.normals)

    dots = np.sum(ground_truth_normals * estimated_normals, axis=1)
    flipped_fraction = float(np.mean(dots < 0))
    mean_agreement = float(np.mean(dots))

    return {
        "object": loaded.object_id,
        "num_points": num_points,
        "mean_dot_agreement": mean_agreement,
        "flipped_fraction": flipped_fraction,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw/cepb/models/models/33-Key/Key.obj"
    result = validate_normals(path)
    for k, v in result.items():
        print(f"{k}: {v}")
