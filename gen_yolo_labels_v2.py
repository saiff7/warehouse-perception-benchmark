import sys, yaml, numpy as np, os, glob, shutil
sys.path.insert(0, "src")
from warehouse_perception.dataset.projection import project_points

CLASSES = {}
def get_class_id(name):
    base = name.split("-", 1)[-1] if "-" in name else name
    if base not in CLASSES:
        CLASSES[base] = len(CLASSES)
    return CLASSES[base]

os.makedirs("data/yolo/images", exist_ok=True)
os.makedirs("data/yolo/labels", exist_ok=True)

img_w, img_h = 2048, 1536
lighting_conditions = ["directional_light", "point_light", "spot_light"]
camera_positions = ["left", "middle", "right"]

count = 0
for cam in camera_positions:
    for scene_id in range(1, 6):
        gt_path = f"data/raw/cepb/scenes_dev/GT_{cam}_camera_{scene_id}.yaml"
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as f:
            gt = yaml.safe_load(f)

        label_lines = []
        for name, obj in gt["objects"].items():
            cuboid = np.array(obj["cuboid"])
            px = project_points(cuboid, cam)
            x_min, y_min = px[:, 0].min(), px[:, 1].min()
            x_max, y_max = px[:, 0].max(), px[:, 1].max()
            x_min, x_max = np.clip([x_min, x_max], 0, img_w)
            y_min, y_max = np.clip([y_min, y_max], 0, img_h)
            if x_max <= x_min or y_max <= y_min:
                continue
            cx = (x_min + x_max) / 2 / img_w
            cy = (y_min + y_max) / 2 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            cls_id = get_class_id(name)
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        for lighting in lighting_conditions:
            img_path = f"data/raw/cepb/scenes_dev/{cam}_camera_{lighting}_scene_{scene_id}_rgb.png"
            if not os.path.exists(img_path):
                continue
            tag = f"{cam}_{lighting}_scene_{scene_id}"
            with open(f"data/yolo/labels/{tag}.txt", "w") as f:
                f.write("\n".join(label_lines))
            shutil.copy(img_path, f"data/yolo/images/{tag}.png")
            count += 1

with open("data/yolo/classes.txt", "w") as f:
    for name, idx in sorted(CLASSES.items(), key=lambda x: x[1]):
        f.write(f"{name}\n")

print(f"Generated {count} image-label pairs, {len(CLASSES)} classes")
print("Classes:", list(CLASSES.keys()))
