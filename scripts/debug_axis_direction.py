"""
Determine the true long-axis direction of the Pringles cuboid in 3D and
compare it against the object's forward/local axes implied by the
quaternion, to find which axis is mismatched or flipped.
"""

import numpy as np
import yaml

with open("data/raw/cepb/scenes_dev/GT_left_camera_1.yaml") as f:
    raw = yaml.safe_load(f)

obj = raw["objects"]["40-Pringles"]
cuboid = np.array(obj["cuboid"])
position = np.array(obj["position"])
quat = np.array(obj["quaternion"])

face_a_center = cuboid[[0,1,4,5]].mean(axis=0)
face_b_center = cuboid[[2,3,6,7]].mean(axis=0)
long_axis_world = face_a_center - face_b_center
long_axis_world /= np.linalg.norm(long_axis_world)
print("Long axis (world frame):", long_axis_world)

def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

R = quat_to_matrix(quat)
print("Local X axis in world:", R[:,0])
print("Local Y axis in world:", R[:,1])
print("Local Z axis in world:", R[:,2])
