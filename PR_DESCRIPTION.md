## Fix: Cuboid projection wireframe misalignment across cameras

### Problem
Cuboid annotation wireframes were crossed/tangled and misaligned
against actual objects across left/middle/right camera scenes.

### Root Cause
- Incorrect cuboid edge topology (vertices connected in the wrong order).
- Unreliable baked-in `projected_cuboid`/`2D_centroid` fields in raw CEPB YAML.
- Ambiguous CEPB left-to-right-handed z-axis convention requiring
  per-camera `tz` sign calibration.

### Fix
- Added correct `CUBOID_EDGES` topology to `src/warehouse_perception/dataset/projection.py`.
- Self-computed `project_points()` replacing baked-in projection fields.
- Calibrated `CAMERA_EXTRINSICS`: left tz=+0.028, middle tz=0.0, right tz=-0.028.
- Updated `scripts/visualize_annotations.py` to import shared constants
  from `projection.py` instead of redefining them locally.

### Cleanup
- Removed 11 redundant one-off debug scripts now superseded by `projection.py`.
- Preserved the edge-topology derivation logic in
  `scripts/derivations/derive_cuboid_edge_topology.py` for reference.
- Consolidated the calibration utility into `scripts/calibration_tools/calibrate_camera_tz.py`.

### Verification
- Regenerated wireframes across 5 scenes x 3 cameras — all correct.
- Added regression test: `tests/test_projection_calibration.py` (2 passed).

### Docs
Full debugging writeup: `docs/postmortems/2026-09-camera-projection-fix.md`
