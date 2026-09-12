"""
Find the Pringles object's actual silhouette extent in the segmentation
mask (by isolating its unique color) and compare against the projected_cuboid
points directly, to check if the 2D ground-truth points themselves are
wrong in image space.
"""

import cv2
import numpy as np
import yaml

seg = cv2.imread("data/raw/cepb/scenes_dev/left_camera_scene_1_segmentation.png")

# Sample the color at the known can location from the earlier screenshot
# (roughly x=450, y=550 in the 1245x945 downscaled view -> scale up)
h, w = seg.shape[:2]
scale_x = w / 1245
scale_y = h / 945
sample_x, sample_y = int(300 * scale_x), int(600 * scale_y)
color = seg[sample_y, sample_x]
print("Sampled color at can location:", color)

mask = np.all(seg == color, axis=-1)
ys, xs = np.where(mask)
print("Silhouette bbox: x=[{},{}] y=[{},{}]".format(xs.min(), xs.max(), ys.min(), ys.max()))
print("Silhouette pixel count:", mask.sum())

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

proj = np.array(raw["objects"]["40-Pringles"]["projected_cuboid"])
print("projected_cuboid points:")
print(proj)
print("projected bbox: x=[{},{}] y=[{},{}]".format(
    proj[:,0].min(), proj[:,0].max(), proj[:,1].min(), proj[:,1].max()))
