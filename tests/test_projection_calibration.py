"""Regression test locking in the calibrated camera extrinsics and
cuboid edge topology so they cannot silently regress."""

from warehouse_perception.dataset.projection import CAMERA_EXTRINSICS, CUBOID_EDGES

EXPECTED_EXTRINSICS = {
    "left":   {"tx": -0.03, "ty": 1.25, "tz": 0.028},
    "middle": {"tx": 0.0,   "ty": 1.25, "tz": 0.0},
    "right":  {"tx": 0.03,  "ty": 1.25, "tz": -0.028},
}

EXPECTED_EDGES = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
]

def test_camera_extrinsics_calibration():
    assert CAMERA_EXTRINSICS == EXPECTED_EXTRINSICS

def test_cuboid_edge_topology():
    assert CUBOID_EDGES == EXPECTED_EDGES
