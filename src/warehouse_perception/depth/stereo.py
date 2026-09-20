"""
Stereo depth estimation from the left/right camera pair, using
OpenCV's Semi-Global Block Matching (SGBM) algorithm.

Depth = (focal_length * baseline) / disparity

focal_length comes from the shared camera intrinsic matrix K.
baseline is the physical distance between the left and right cameras,
derived from their calibrated tx extrinsics (see dataset/projection.py).
"""

import numpy as np
import cv2

from warehouse_perception.dataset.projection import K, CAMERA_EXTRINSICS

FOCAL_LENGTH_PX = K[0, 0]
BASELINE = abs(CAMERA_EXTRINSICS["right"]["tx"] - CAMERA_EXTRINSICS["left"]["tx"])


def compute_disparity(left_gray, right_gray, num_disparities=192, block_size=9):
    """Compute a disparity map from rectified/aligned grayscale stereo images."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def disparity_to_depth(disparity, focal_length=FOCAL_LENGTH_PX, baseline=BASELINE):
    """Convert a disparity map (pixels) to a depth map (same units as baseline)."""
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length * baseline) / disparity[valid]
    return depth


def estimate_depth_from_stereo_pair(left_image_path, right_image_path):
    """Full pipeline: load stereo pair, compute disparity, convert to depth."""
    left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("Could not load one or both stereo images")
    disparity = compute_disparity(left, right)
    depth = disparity_to_depth(disparity)
    return disparity, depth
