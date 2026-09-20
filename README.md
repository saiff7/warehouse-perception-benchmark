# Warehouse Perception Benchmark

A hands-on evaluation of stereo depth estimation and object detection on synthetic warehouse-bin imagery (CEPB dataset), built while preparing for robotics perception roles. This project prioritizes **rigorous debugging and honest evaluation** over polished demos — every claim below is backed by a script you can re-run against the data.

## What This Project Actually Does

Given stereo image pairs of cluttered warehouse bins (10 SKU-style objects: a Pringles can, wine glass, wrench, etc., shot from left/middle/right cameras under 3 lighting conditions), this project:

1. Computes per-pixel depth using classical stereo block matching (OpenCV `StereoSGBM`).
2. Validates that depth against dataset-provided ground-truth 3D object positions.
3. Attempts to fine-tune a YOLOv8n object detector on the same imagery.
4. Wraps the depth pipeline as a ROS2 node publishing live depth statistics.

## Part 1: Stereo Depth Estimation — Debugged and Validated

**Starting bug.** The stereo pipeline initially reported depth values up to 341 world units for a scene where the camera sits roughly 1.25 units from the bin — physically implausible.

**Root cause.** Ground-truth validation (projecting a known object's 3D position into both camera views to compute the *true* pixel disparity) showed the real disparity for a nearby object was ~143 px, but the matcher's `numDisparities` was capped at 128 — the search range was too narrow to find correct matches for near objects, silently clipping them.

**Fix.** Increased `numDisparities` from 128 to 192 (the nearest valid multiple of 16 comfortably above the observed need). This resolved the clipping for the tested case, though the search range was validated against only the specific objects checked here, not proven sufficient for all possible object distances in the dataset.

**Remaining gap fix.** One object (a book) still returned a hard "no match" (`-1`) at its ground-truth pixel location. Applied OpenCV inpainting to fill isolated no-match holes using neighboring valid disparity — reduced that object's error from undefined to +2.2%.

**Validated error characterization.** Cross-checked estimated vs. ground-truth disparity for all 10 objects in a scene:

| Object type | Disparity error |
|---|---|
| Textured rigid objects (Pringles can, wrench, key, nail) | 0.5% – 6.3% |
| Reflective/transparent/mesh objects (wine glass, plastic cup, wire pencil cup) | 9.1% – 12.1% |

This matches a visual inspection of the disparity map: dense, consistent matches on the printed Pringles label; sparse, broken matches on the wire-mesh cup's repetitive grid pattern (a classic stereo "aperture problem" — repetitive texture defeats block matching).

**Additional finding: lighting affects match density.** Valid-pixel fraction varied by lighting condition when measured across all 5 scenes (e.g., ~0.39–0.46 under directional/point light vs. up to 0.66 under spot light in one scene), suggesting spot lighting produces higher-contrast surfaces that are easier to match. This was observed via the ROS2 node's live output, not exhaustively studied across all scenes.

**Known limitations of this analysis:**
- Ground-truth comparison used a single projected centroid pixel per object, not a dense per-pixel comparison across each object's full surface.
- The `numDisparities` fix was validated against the object that was failing, not stress-tested against every possible object distance in the dataset.

## Part 2: Object Detection Fine-Tuning — Honest Negative Result

**Approach.** Auto-generated YOLO-format 2D bounding-box labels directly from the dataset's ground-truth 3D cuboids (projected into each camera view), avoiding manual labeling. Produced 45 labeled images (5 scenes × 3 cameras × 3 lighting conditions) across 10 classes. Split by scene number (scenes 1–4 train, scene 5 held out) to prevent geometry leakage between splits.

**Result.** Fine-tuned YOLOv8n (transfer learning, CPU-only) for 17 epochs (early-stopped). Final per-class mAP50:

| Class | mAP50 |
|---|---|
| T-shirt | 0.961 |
| Pringles | 0.223 |
| All other 8 classes | 0.000 |

At default confidence thresholds (0.05–0.25), the model produced **zero detections** on both training and held-out images. At a near-zero threshold (0.001), it produced 147 detections on a single image with confidences around 0.001–0.002 — far too low to be usable, and far too many to be meaningful (severe false-positive flooding).

**Diagnosis.** This is not a code or calibration bug — deeper investigation confirmed the exact same checkpoint, correctly loaded, at correct image size. It is a **data-scarcity failure**: 4 independent training scenes is insufficient for a 10-class detector to learn small, visually similar, cluttered objects. Only the single visually largest/most distinct class (T-shirt) learned any real signal.

**What was not tried, and why.** Given time constraints, heavier augmentation, layer-freezing variations, and classical (non-learned) detection approaches (e.g., color/contour-based methods, which may suit this constrained, low-diversity scene better than a data-hungry CNN) were considered but not implemented. This remains open, not proven unnecessary.

## Part 3: ROS2 Integration

A ROS2 node (`perception_pkg/stereo_depth_publisher.py`) wraps the validated stereo depth pipeline, cycling through all 45 scene image pairs on a 5-second timer and publishing per-scene depth statistics (valid-pixel fraction, min/max/median depth) as JSON over a `std_msgs/String` topic. Verified working end-to-end with an independent subscriber (`ros2 topic echo`) receiving live messages.

**Known limitations:** Uses a generic `String` message with embedded JSON rather than a proper custom ROS2 message type. No downstream consumer node exists yet — nothing currently acts on the published data. No launch file or parameterization; paths are hardcoded.

## Explicitly Not Done

- Detection + depth integration was scripted but never verified against a working detector (the detector's output was unusable noise, so the integration logic itself remains untested with real bounding boxes).
- No grasp-point selection logic.
- No vision-language model (VLM) component.
- No live/streaming camera input — all processing is on static pre-rendered images.

## Repository Structure

warehouse-perception-benchmark/
├── src/warehouse_perception/
│ ├── depth/stereo.py # Stereo depth estimation (fixed numDisparities bug)
│ └── dataset/projection.py # Ground-truth 3D-to-2D projection utilities
├── data/raw/cepb/scenes_dev/ # CEPB dataset (not included in repo — see Data section)
├── data/yolo/ # Auto-generated YOLO labels from ground truth
├── output/ # Generated reports, visualizations, trained weights
├── run_report.py # Per-object disparity error report vs. ground truth
├── gen_yolo_labels_v2.py # Converts GT cuboids to YOLO bounding-box labels
├── split_yolo_dataset.py # Train/val split by scene (avoids geometry leakage)
├── detect_and_depth.py # Detection + depth integration (unverified, see caveats)
└── visualize_disparity.py # Side-by-side left/right/disparity visualization

## Data

This repo does not include the CEPB dataset images/labels due to size. Place the dataset under `data/raw/cepb/scenes_dev/` following the existing naming convention (`{position}_camera_{lighting}_scene_{n}_rgb.png`, `GT_{position}_camera_{n}.yaml`) before running any script.

## Reproducing the Results

```bash
python3 run_report.py                    # Stereo depth error report (Part 1)
python3 gen_yolo_labels_v2.py             # Generate detection labels from ground truth
python3 split_yolo_dataset.py             # Train/val split
python3 -m ultralytics train data=data/yolo/dataset.yaml model=yolov8n.pt epochs=50 device=cpu  # Part 2
```

For the ROS2 node, see `perception_pkg/` — build with `colcon build --packages-select perception_pkg` inside a ROS2 workspace.
