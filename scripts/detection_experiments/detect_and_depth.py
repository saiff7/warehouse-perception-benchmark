import sys, glob
sys.path.insert(0, "src")
from warehouse_perception.depth.stereo import estimate_depth_from_stereo_pair
from ultralytics import YOLO
import numpy as np
import cv2

left_img_path = glob.glob("data/raw/cepb/scenes_dev/left_camera_*_scene_5_rgb.png")[0]
right_img_path = glob.glob("data/raw/cepb/scenes_dev/right_camera_*_scene_5_rgb.png")[0]

print("Computing stereo depth...")
disparity, depth = estimate_depth_from_stereo_pair(left_img_path, right_img_path)

print("Loading fine-tuned detector...")
model = YOLO("runs/detect/output/yolo_finetune/cepb_run1/weights/best.pt")
results = model(left_img_path, conf=0.05)

left_img = cv2.imread(left_img_path)
vis = left_img.copy()

print(f"\n{'Object':30s} {'Confidence':>10s} {'BBox (x1,y1,x2,y2)':>25s} {'Median Depth':>14s}")
print("-" * 85)

detections = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_name = model.names[cls_id]

    region_depth = depth[y1:y2, x1:x2]
    valid_depth = region_depth[region_depth > 0]
    median_depth = float(np.median(valid_depth)) if valid_depth.size > 0 else None

    detections.append({
        "class": cls_name, "confidence": conf,
        "bbox": (x1, y1, x2, y2), "median_depth": median_depth,
        "valid_depth_fraction": float((region_depth > 0).mean())
    })

    depth_str = f"{median_depth:.3f}" if median_depth is not None else "N/A"
    print(f"{cls_name:30s} {conf:10.3f} {str((x1,y1,x2,y2)):>25s} {depth_str:>14s}")

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"{cls_name} {conf:.2f} d={depth_str}"
    cv2.putText(vis, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imwrite("output/detection_plus_depth_scene5.png", vis)
print(f"\nTotal detections: {len(detections)}")
print("Saved visualization to output/detection_plus_depth_scene5.png")
