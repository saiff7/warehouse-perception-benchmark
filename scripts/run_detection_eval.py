"""
Run the classical detector across all 5 dev scenes x 3 cameras,
compare detection count against ground-truth object count, save
annotated outputs, and write a summary CSV with honest recall numbers.
"""

import csv
import glob
import os
import sys
import yaml

sys.path.insert(0, "src")
from warehouse_perception.detection.detector import detect_from_segmentation, draw_detections

SCENES_DIR = "data/raw/cepb/scenes_dev"
OUT_DIR = "experiments/results/detection"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for cam in ["left", "middle", "right"]:
    for scene in range(1, 6):
        seg_matches = glob.glob(f"{SCENES_DIR}/{cam}_camera_scene_{scene}_segmentation.png")
        gt_matches = glob.glob(f"{SCENES_DIR}/GT_{cam}_camera_{scene}.yaml")
        if not seg_matches or not gt_matches:
            continue

        detections = detect_from_segmentation(seg_matches[0])

        with open(gt_matches[0]) as f:
            gt = yaml.safe_load(f)
        num_gt_objects = len(gt["objects"])

        rgb_matches = glob.glob(f"{SCENES_DIR}/{cam}_camera_*_scene_{scene}_rgb.png")
        if rgb_matches:
            out_path = f"{OUT_DIR}/detected_{cam}_scene_{scene}.png"
            draw_detections(rgb_matches[0], detections, out_path)

        recall = len(detections) / num_gt_objects if num_gt_objects else 0
        rows.append({
            "camera": cam,
            "scene": scene,
            "num_detections": len(detections),
            "num_gt_objects": num_gt_objects,
            "recall": round(recall, 3),
        })
        print(f"{cam} scene {scene}: {len(detections)}/{num_gt_objects} objects detected (recall={recall:.2f})")

with open(f"{OUT_DIR}/detection_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["camera", "scene", "num_detections", "num_gt_objects", "recall"])
    writer.writeheader()
    writer.writerows(rows)

avg_recall = sum(r["recall"] for r in rows) / len(rows)
print(f"\nSaved {len(rows)} rows to {OUT_DIR}/detection_summary.csv")
print(f"Average recall across all scenes/cameras: {avg_recall:.2%}")
