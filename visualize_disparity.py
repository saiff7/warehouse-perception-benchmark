import sys
sys.path.insert(0, 'src')
from warehouse_perception.depth.stereo import estimate_depth_from_stereo_pair
import glob, numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

left_img_path = glob.glob('data/raw/cepb/scenes_dev/left_camera_*_scene_1_rgb.png')[0]
right_img_path = glob.glob('data/raw/cepb/scenes_dev/right_camera_*_scene_1_rgb.png')[0]
disparity, depth = estimate_depth_from_stereo_pair(left_img_path, right_img_path)

left_img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2RGB)
right_img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2RGB)

disp_vis = disparity.copy()
disp_vis[disp_vis < 0] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
axes[0].imshow(left_img); axes[0].set_title('Left camera')
axes[1].imshow(right_img); axes[1].set_title('Right camera')
im = axes[2].imshow(disp_vis, cmap='turbo')
axes[2].set_title('Disparity map (black = no match / -1)')
plt.colorbar(im, ax=axes[2], label='disparity (px)')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('output/disparity_visualization.png', dpi=100)
print('Saved output/disparity_visualization.png')
