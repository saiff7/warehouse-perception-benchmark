# Postmortem: Stereo Depth Disparity Clipping and Detection Data-Scarcity Findings

**Date:** 2026-09-20

## Summary

Two separate investigations: (1) a stereo depth estimation bug causing implausible depth values, root-caused and fixed; (2) a YOLOv8n fine-tuning experiment that failed due to insufficient training data, documented as an honest negative result rather than hidden or hand-waved.

## Part 1: Stereo Depth `numDisparities` Clipping

### Symptom

`estimate_depth_from_stereo_pair()` reported depth values up to 341 world units in a scene where the camera sits ~1.25 units from the bin -- physically implausible for a tabletop-scale scene.

### Root cause

Ground-truth validation via `project_points()` (the same projection function fixed in the camera-handedness postmortem) showed the true disparity for the Pringles can object was ~142.8 px, but `StereoSGBM` was configured with `numDisparities=128`, capping the maximum searchable disparity at 127. Any object at or beyond that true disparity was unmatchable by construction, silently returning `-1` or clipping to the ceiling.

### Fix

Increased `numDisparities` from 128 to 192 (nearest valid multiple of 16 above the observed requirement).

### Validation

Cross-checked estimated vs. ground-truth disparity for all 10 objects in scene 1 using each object's cuboid centroid, projected into both camera views:

| Object type | Disparity error |
|---|---|
| Textured rigid objects (Pringles, wrench, key, nail) | 0.5% - 6.3% |
| Reflective/transparent/mesh objects (wine glass, plastic cup, wire pencil cup) | 9.1% - 12.1% |

A residual `-1` (no-match) failure on one object was resolved with OpenCV inpainting on isolated invalid-disparity holes, reducing that object's error from undefined to +2.2%.

### Known limitations

- Validated against a single projected centroid pixel per object, not a dense per-pixel surface comparison.
- The new `numDisparities=192` ceiling was validated against the specific object that was failing, not stress-tested against the full range of possible object distances in the dataset. A future closer object could still exceed 192 and reproduce the same failure mode.

## Part 2: YOLOv8n Fine-Tuning -- Data-Scarcity Failure

### Approach

Auto-generated 2D bounding-box labels from ground-truth 3D cuboids (via `project_points()`), producing 45 labeled images (5 scenes x 3 cameras x 3 lighting conditions, 10 classes). Split by scene number (1-4 train, 5 held out) to avoid geometry leakage.

### Result

Fine-tuned YOLOv8n via transfer learning, CPU-only, 17 epochs (early-stopped). Final mAP50: T-shirt 0.961, Pringles 0.223, all other 8 classes 0.000. At default confidence thresholds, zero detections on both training and held-out images; at conf=0.001, 147 low-confidence (~0.001-0.002) detections on a single image -- unusable in either direction.

### Diagnosis

Not a code or checkpoint-loading bug (confirmed via direct inference on the model's own training-time validation image path). Root cause: 4 independent training scenes is insufficient for a 10-class detector to learn small, visually similar, cluttered objects. Only the single visually largest/most distinct class (T-shirt) learned usable signal.

### Relation to the classical detector

A separate classical segmentation-based detector (`src/warehouse_perception/detection/detector.py`, see `scripts/run_detection_eval.py`) achieved 68% average recall across 15 scene/camera combinations without requiring any training data -- directly illustrating why a classical, non-learned approach was better suited to this specific data-scarce, geometrically-constrained scenario than a data-hungry CNN fine-tune.

### What was not tried

Heavier augmentation, alternative layer-freezing strategies. Left open given time constraints, not proven unnecessary.

## Part 3: ROS2 Integration

`ros2/perception_pkg/` wraps the fixed stereo depth pipeline as a ROS2 node (`stereo_depth_publisher`), cycling through all 45 scene image pairs and publishing per-scene depth statistics as JSON over `std_msgs/String`. Verified end-to-end with an independent `ros2 topic echo` subscriber.

Observed as a side effect: valid-pixel fraction varied by lighting condition (~0.39-0.46 under directional/point light vs. up to 0.66 under spot light in one scene), suggesting lighting materially affects stereo match density -- not exhaustively studied, but worth flagging for future work.

### Known limitations

Generic `String`/JSON message rather than a custom ROS2 message type. No downstream consumer node. No launch file; paths configurable via `WAREHOUSE_PERCEPTION_ROOT` env var but otherwise unparameterized.
