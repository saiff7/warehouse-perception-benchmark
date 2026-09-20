"""
Evaluate stereo-estimated depth against ground-truth object depth
(derived from each object's cuboid centroid position along the
camera's viewing axis, as recorded in the CEPB scene YAML).
"""

import numpy as np


def cuboid_centroid_depth(cuboid_points, camera_ty=1.25):
    """
    Approximate ground-truth depth of an object as the mean distance
    of its cuboid corners from the camera along the world y-axis
    (camera height/depth reference used in dataset/projection.py).
    """
    centroid = np.mean(np.asarray(cuboid_points), axis=0)
    return abs(camera_ty - centroid[1])


def compare_depth_at_point(depth_map, pixel_xy, gt_depth):
    """Compare the stereo depth map's estimate at a pixel against ground truth."""
    x, y = pixel_xy
    estimated = float(depth_map[int(y), int(x)])
    if estimated <= 0:
        return {"estimated": None, "ground_truth": gt_depth, "abs_error": None, "valid": False}
    abs_error = abs(estimated - gt_depth)
    return {
        "estimated": estimated,
        "ground_truth": gt_depth,
        "abs_error": abs_error,
        "rel_error": abs_error / gt_depth if gt_depth else None,
        "valid": True,
    }
