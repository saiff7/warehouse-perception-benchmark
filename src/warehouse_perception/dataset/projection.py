"""
Manual 3D-to-2D cuboid projection, replacing CEPB's baked-in
projected_cuboid/2D_centroid fields, which exhibit a systematic
directional offset across all objects and cameras. Extrinsics tz sign
empirically calibrated against ground-truth Pringles can position.
"""

import numpy as np

K = np.array([[3200, 0, 1024], [0, 3200, 768], [0, 0, 1]], dtype=float)

CAMERA_EXTRINSICS = {
    "left":   {"tx": -0.03, "ty": 1.25, "tz": 0.028},
    "middle": {"tx": 0.0,   "ty": 1.25, "tz": 0.0},
    "right":  {"tx": 0.03,  "ty": 1.25, "tz": -0.028},
}

CUBOID_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]


def _make_H(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 0, -1, ty],
        [0, 1, 0, tz],
        [0, 0, 0, 1],
    ])


def project_points(points_3d, camera_id="left"):
    """Project an array of (N, 3) world points to (N, 2) image points."""
    ext = CAMERA_EXTRINSICS[camera_id]
    H = _make_H(ext["tx"], ext["ty"], ext["tz"])
    H_inv = np.linalg.inv(H)

    points_3d = np.asarray(points_3d, dtype=float)
    flipped = points_3d.copy()
    flipped[:, 1] *= -1
    homo = np.hstack([flipped, np.ones((len(flipped), 1))])
    cam_pts = (H_inv @ homo.T).T[:, :3]
    img_pts = (K @ cam_pts.T).T
    return img_pts[:, :2] / img_pts[:, 2:3]
