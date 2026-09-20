import os, shutil, glob

os.makedirs("data/yolo/train/images", exist_ok=True)
os.makedirs("data/yolo/train/labels", exist_ok=True)
os.makedirs("data/yolo/val/images", exist_ok=True)
os.makedirs("data/yolo/val/labels", exist_ok=True)

train_scenes = {"1", "2", "3", "4"}
val_scenes = {"5"}

for img_path in glob.glob("data/yolo/images/*.png"):
    tag = os.path.basename(img_path).replace(".png", "")
    scene_num = tag.split("_scene_")[-1]
    label_path = f"data/yolo/labels/{tag}.txt"

    if scene_num in train_scenes:
        shutil.copy(img_path, f"data/yolo/train/images/{tag}.png")
        shutil.copy(label_path, f"data/yolo/train/labels/{tag}.txt")
    elif scene_num in val_scenes:
        shutil.copy(img_path, f"data/yolo/val/images/{tag}.png")
        shutil.copy(label_path, f"data/yolo/val/labels/{tag}.txt")

with open("data/yolo/classes.txt") as f:
    classes = [l.strip() for l in f if l.strip()]

yaml_content = f"""path: {os.path.abspath("data/yolo")}
train: train/images
val: val/images
nc: {len(classes)}
names: {classes}
"""
with open("data/yolo/dataset.yaml", "w") as f:
    f.write(yaml_content)

n_train = len(glob.glob("data/yolo/train/images/*.png"))
n_val = len(glob.glob("data/yolo/val/images/*.png"))
print(f"Train: {n_train} images, Val: {n_val} images")
print(yaml_content)
