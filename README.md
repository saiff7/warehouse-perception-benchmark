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

## Project Structure

- `src/warehouse_perception/` — core library (dataset, detection, depth, grasp, evaluation, geometry, ROS nodes)
- `scripts/` — data preprocessing, visualization, and calibration utilities
- `tests/` — unit, integration, and regression tests
- `configs/` — YAML configs for dataset, detection, depth, grasp, and experiments
- `docs/` — architecture, methodology, results, and postmortems

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

Supported camera IDs: \`left\`, \`middle\`, \`right\`.
