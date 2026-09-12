import cv2
import numpy as np
from warehouse_perception.dataset.loaders import load_scene

scene = load_scene("data/raw/cepb/scenes_dev", scene_id="1", camera_id="left")
img = cv2.imread(str(scene.rgb_path))
print("Actual image shape (h, w, c):", img.shape)

all_x, all_y = [], []
for obj in scene.annotation.objects:
    for pt in obj.cuboid_2d:
        all_x.append(pt[0])
        all_y.append(pt[1])
    all_x.append(obj.centroid_2d[0])
    all_y.append(obj.centroid_2d[1])

print("Projected coord range x:", min(all_x), "to", max(all_x))
print("Projected coord range y:", min(all_y), "to", max(all_y))

obj0 = scene.annotation.objects[0]
print("Object:", obj0.object_id)
print("cuboid_2d:", obj0.cuboid_2d)
print("centroid_2d:", obj0.centroid_2d)
