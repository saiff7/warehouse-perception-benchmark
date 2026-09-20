# Warehouse Perception Benchmark

A benchmark suite for evaluating perception, depth estimation, and
grasp-candidate ranking in simulated warehouse picking scenes, built on
top of the CEPB dataset.

## Overview

This project provides tooling to load, annotate, project, and evaluate
3D object cuboids across multi-camera warehouse scenes, along with
downstream detection, depth-quality, and grasp-ranking evaluation
pipelines.

## Highlight: Cuboid Projection Calibration Fix

Fixed a systematic bug where 3D cuboid wireframes rendered crossed and
misaligned against their target objects across left/middle/right camera
views. Root cause was a combination of incorrect cuboid edge topology
and an ambiguous coordinate-handedness convention in the raw dataset,
requiring per-camera empirical calibration.

| Left Camera | Middle Camera | Right Camera |
|---|---|---|
| ![Left](docs/images/example_left_camera_fixed.png) | ![Middle](docs/images/example_middle_camera_fixed.png) | ![Right](docs/images/example_right_camera_fixed.png) |

- Full writeup: [`docs/postmortems/2026-09-camera-projection-fix.md`](docs/postmortems/2026-09-camera-projection-fix.md)
- Core fix: [`src/warehouse_perception/dataset/projection.py`](src/warehouse_perception/dataset/projection.py)
- Regression test: [`tests/test_projection_calibration.py`](tests/test_projection_calibration.py)

## Highlight: Stereo Depth Fix, Classical Detector, and ROS2 Integration

Fixed a stereo depth estimation bug (`numDisparities` search range too narrow, clipping near objects), validated the fix against ground truth with a per-object error breakdown, and diagnosed why a YOLOv8n fine-tune failed on this dataset (data scarcity) versus why a classical segmentation-based detector succeeded (68% avg recall, 15 scene/camera combos, no training data required). Wrapped the depth pipeline as a live ROS2 node.

- Full writeup: [`docs/postmortems/2026-09-stereo-depth-and-detection.md`](docs/postmortems/2026-09-stereo-depth-and-detection.md)
- Stereo depth fix: [`src/warehouse_perception/depth/stereo.py`](src/warehouse_perception/depth/stereo.py)
- Classical detector: [`src/warehouse_perception/detection/detector.py`](src/warehouse_perception/detection/detector.py)
- ROS2 node: [`ros2/perception_pkg/`](ros2/perception_pkg/)
- Error report: [`scripts/run_report.py`](scripts/run_report.py)

## Project Structure

- `src/warehouse_perception/` — core library (dataset, detection, depth, grasp, evaluation, geometry, ROS nodes)
- `scripts/` — data preprocessing, visualization, and calibration utilities
- `scripts/detection_experiments/` — YOLOv8n fine-tuning experiment (negative result, see postmortem)
- `ros2/perception_pkg/` — ROS2 node wrapping the stereo depth pipeline
- `tests/` — unit, integration, and regression tests
- `configs/` — YAML configs for dataset, detection, depth, grasp, and experiments
- `docs/` — architecture, methodology, results, and postmortems

**Additional finding: lighting affects match density.** Valid-pixel fraction varied by lighting condition when measured across all 5 scenes (e.g., ~0.39-0.46 under directional/point light vs. up to 0.66 under spot light in one scene), suggesting spot lighting produces higher-contrast surfaces that are easier to match. This was observed via the ROS2 node's live output, not exhaustively studied across all scenes.

**Known limitations of this analysis:**
- Ground-truth comparison used a single projected centroid pixel per object, not a dense per-pixel comparison across each object's full surface.
- The `numDisparities` fix was validated against the object that was failing, not stress-tested against every possible object distance in the dataset.

## Part 2: Object Detection Fine-Tuning - Honest Negative Result

**Approach.** Auto-generated YOLO-format 2D bounding-box labels directly from the dataset's ground-truth 3D cuboids (projected into each camera view), avoiding manual labeling. Produced 45 labeled images (5 scenes x 3 cameras x 3 lighting conditions) across 10 classes. Split by scene number (scenes 1-4 train, scene 5 held out) to prevent geometry leakage between splits.

**Result.** Fine-tuned YOLOv8n (transfer learning, CPU-only) for 17 epochs (early-stopped). Final per-class mAP50:

| Class | mAP50 |
|---|---|
| T-shirt | 0.961 |
| Pringles | 0.223 |
| All other 8 classes | 0.000 |

At default confidence thresholds (0.05-0.25), the model produced **zero detections** on both training and held-out images. At a near-zero threshold (0.001), it produced 147 detections on a single image with confidences around 0.001-0.002 - far too low to be usable, and far too many to be meaningful (severe false-positive flooding).

**Diagnosis.** This is not a code or calibration bug - deeper investigation confirmed the exact same checkpoint, correctly loaded, at correct image size. It is a **data-scarcity failure**: 4 independent training scenes is insufficient for a 10-class detector to learn small, visually similar, cluttered objects. Only the single visually largest/most distinct class (T-shirt) learned any real signal.

**What was not tried, and why.** Given time constraints, heavier augmentation, layer-freezing variations, and classical (non-learned) detection approaches (e.g., color/contour-based methods, which may suit this constrained, low-diversity scene better than a data-hungry CNN) were considered but not implemented. This remains open, not proven unnecessary.

## Part 3: ROS2 Integration

A ROS2 node (`perception_pkg/stereo_depth_publisher.py`) wraps the validated stereo depth pipeline, cycling through all 45 scene image pairs on a 5-second timer and publishing per-scene depth statistics (valid-pixel fraction, min/max/median depth) as JSON over a `std_msgs/String` topic. Verified working end-to-end with an independent subscriber (`ros2 topic echo`) receiving live messages.

**Known limitations:** Uses a generic `String` message with embedded JSON rather than a proper custom ROS2 message type. No downstream consumer node exists yet - nothing currently acts on the published data. No launch file or parameterization; paths are hardcoded.

## Explicitly Not Done

- Detection + depth integration was scripted but never verified against a working detector (the detector's output was unusable noise, so the integration logic itself remains untested with real bounding boxes).
- No grasp-point selection logic.
- No vision-language model (VLM) component.
- No live/streaming camera input - all processing is on static pre-rendered images.

## Data

This repo does not include the CEPB dataset images/labels due to size. Place the dataset under `data/raw/cepb/scenes_dev/` following the existing naming convention (`{position}_camera_{lighting}_scene_{n}_rgb.png`, `GT_{position}_camera_{n}.yaml`) before running any script.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Visualizing Annotations

```bash
python scripts/visualize_annotations.py data/raw/cepb/scenes_dev <scene_id> <camera_id>
```

Supported camera IDs: `left`, `middle`, `right`.

## Stereo Depth Error Report

```bash
python scripts/run_report.py
```

## Running the ROS2 Depth Publisher

```bash
export WAREHOUSE_PERCEPTION_ROOT=~/warehouse-perception-benchmark
ln -s $WAREHOUSE_PERCEPTION_ROOT/ros2/perception_pkg ~/ros2_ws/src/perception_pkg
cd ~/ros2_ws && colcon build --packages-select perception_pkg
source install/setup.bash
ros2 run perception_pkg stereo_depth_publisher
```
