# Postmortem: Cuboid Projection Wireframe Misalignment

## Summary
Cuboid bounding-box wireframes rendered crossed/tangled and offset from
their actual objects across left/middle/right camera scenes in CEPB.
Root cause was twofold: an incorrect 12-edge topology for the cuboid
vertex ordering, and an ambiguous left-handed-to-right-handed z-axis
sign convention in CEPB's per-camera extrinsics that required empirical
calibration per camera.

## Impact
All cuboid annotation visualizations across scenes were unreliable,
blocking annotation QA and downstream benchmark validation.

## Root Cause
1. `CUBOID_EDGES` used a hardcoded edge list that did not match CEPB's
   actual vertex ordering, causing crossed wireframes.
2. Baked-in `projected_cuboid` / `2D_centroid` fields in the raw YAML
   were unreliable; we needed to self-compute projections.
3. CEPB documentation on left-handed-to-right-handed coordinate
   conversion for camera extrinsics was ambiguous, and the `tz` sign
   differs per camera (left: +0.028, middle: 0.0, right: -0.028).

## False Leads Investigated
- Camera swap (assumed left/right images mislabeled) — ruled out.
- Resolution mismatch between K matrix and rendered image — ruled out.
- Scale bug in cuboid dimensions — ruled out.
- Occlusion/rendering-order theory — ruled out.

## Fix
- Corrected `CUBOID_EDGES` to the empirically verified 12-edge topology.
- Replaced baked-in projection fields with a self-computed
  `project_points()` in `projection.py`.
- Calibrated and hardcoded per-camera `tz` extrinsics after empirical
  sign testing against each camera's own RGB image.

## Verification
Regenerated annotated wireframes across 5 scenes x 3 cameras (15
renders total); all cuboids now hug their target objects correctly.

## Follow-ups
- Add regression test locking in `CUBOID_EDGES` and `CAMERA_EXTRINSICS`.
- Report ambiguous handedness convention upstream to CEPB dataset maintainers.
