import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import os
import json
import glob
import numpy as np

REPO_ROOT = os.environ.get(
    'WAREHOUSE_PERCEPTION_ROOT',
    os.path.expanduser('~/warehouse-perception-benchmark')
)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from warehouse_perception.depth.stereo import estimate_depth_from_stereo_pair


class StereoDepthPublisher(Node):
    def __init__(self):
        super().__init__('stereo_depth_publisher')
        self.publisher_ = self.create_publisher(String, 'stereo_depth_topic', 10)
        self.timer = self.create_timer(5.0, self.publish_depth_summary)
        self.scene_paths = self._find_scene_pairs()
        self.scene_index = 0
        self.get_logger().info(f'Stereo depth publisher started. Found {len(self.scene_paths)} scene pairs.')

    def _find_scene_pairs(self):
        base = os.path.join(REPO_ROOT, 'data/raw/cepb/scenes_dev')
        left_imgs = sorted(glob.glob(f'{base}/left_camera_*_scene_*_rgb.png'))
        pairs = []
        for left in left_imgs:
            right = left.replace('left_camera', 'right_camera')
            pairs.append((left, right))
        return pairs

    def publish_depth_summary(self):
        if not self.scene_paths:
            self.get_logger().warn('No scene pairs found. Check WAREHOUSE_PERCEPTION_ROOT env var and dataset path.')
            return

        left_path, right_path = self.scene_paths[self.scene_index % len(self.scene_paths)]
        self.scene_index += 1

        try:
            disparity, depth = estimate_depth_from_stereo_pair(left_path, right_path)
            valid = depth[depth > 0]

            summary = {
                'scene_file': left_path.split('/')[-1],
                'valid_pixel_fraction': float((disparity > 0).mean()),
                'depth_min': float(valid.min()) if valid.size > 0 else None,
                'depth_max': float(valid.max()) if valid.size > 0 else None,
                'depth_median': float(np.median(valid)) if valid.size > 0 else None,
            }

            msg = String()
            msg.data = json.dumps(summary)
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published depth summary for {summary["scene_file"]}')

        except Exception as e:
            self.get_logger().error(f'Failed to process {left_path}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
