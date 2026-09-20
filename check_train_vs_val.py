import sys, glob
sys.path.insert(0, "src")
from ultralytics import YOLO

model = YOLO("runs/detect/output/yolo_finetune/cepb_run1/weights/best.pt")

for scene in [1, 2, 3, 4, 5]:
    for img_path in sorted(glob.glob(f"data/raw/cepb/scenes_dev/left_camera_*_scene_{scene}_rgb.png")):
        results = model(img_path, conf=0.01, verbose=False)
        tag = "TRAIN" if scene != 5 else "VAL(held-out)"
        print(f"[{tag}] {img_path.split('/')[-1]}: {len(results[0].boxes)} detections")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    {model.names[cls_id]}: {conf:.3f}")
