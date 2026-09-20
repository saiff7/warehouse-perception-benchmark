import sys, yaml, numpy as np, cv2
sys.path.insert(0, 'src')
from warehouse_perception.dataset.projection import project_points
from warehouse_perception.depth.stereo import estimate_depth_from_stereo_pair
import glob

with open('data/raw/cepb/scenes_dev/GT_left_camera_1.yaml') as f:
    gt = yaml.safe_load(f)

left_img = glob.glob('data/raw/cepb/scenes_dev/left_camera_*_scene_1_rgb.png')[0]
right_img = glob.glob('data/raw/cepb/scenes_dev/right_camera_*_scene_1_rgb.png')[0]
disparity, depth = estimate_depth_from_stereo_pair(left_img, right_img)

mask = (disparity <= 0).astype('uint8')
disp_filled = disparity.copy()
disp_filled[disparity <= 0] = 0
disp_filled = cv2.inpaint(disp_filled.astype('float32'), mask, 5, cv2.INPAINT_TELEA)

rows = []
for name, obj in gt['objects'].items():
    cuboid = np.array(obj['cuboid'])
    left_px = project_points(cuboid, 'left').mean(axis=0)
    right_px = project_points(cuboid, 'right').mean(axis=0)
    true_disp = left_px[0] - right_px[0]
    x, y = int(round(left_px[0])), int(round(left_px[1]))
    if 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
        raw = disparity[y, x]
        filled = disp_filled[y, x]
        used = filled if raw <= 0 else raw
        err_pct = 100 * (used - true_disp) / true_disp
        rows.append((name, true_disp, raw, filled, used, err_pct))

import csv
with open('output/stereo_error_report.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['object', 'true_disparity', 'raw_est', 'filled_est', 'used_est', 'error_pct'])
    for r in rows:
        w.writerow([r[0], f'{r[1]:.2f}', f'{r[2]:.2f}', f'{r[3]:.2f}', f'{r[4]:.2f}', f'{r[5]:+.1f}'])

print('Report written to output/stereo_error_report.csv')
for r in sorted(rows, key=lambda r: -abs(r[5])):
    print(f'{r[0]:35s} true={r[1]:7.2f} raw={r[2]:7.2f} used={r[4]:7.2f} err%={r[5]:+6.1f}')
