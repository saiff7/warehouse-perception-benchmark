"""
Classical object detector: takes a scene's segmentation mask and RGB
image, extracts per-object bounding boxes via connected-component
analysis, and returns detections in a format comparable to ground
truth cuboids for IoU evaluation.
"""

import numpy as np
import cv2


def detect_from_segmentation(segmentation_path, min_area=5000, max_area_frac=0.10):
    """
    Given a segmentation PNG where each object has a distinct pixel
    color/label, return a list of detected bounding boxes.

    Excludes value=0 (true background) and any region covering more
    than max_area_frac of the image (catches secondary background-like
    regions such as a bright bin/tray surface, which are not objects).

    min_area=5000 and max_area_frac=0.10 were chosen empirically: on a
    2048x1536 test image, real objects ranged ~10k-160k px, while
    anti-aliased edge-pixel noise stayed below ~3k px, and the actual
    background/tray regions covered 13%-72% of the frame.

    Returns: list of dicts {"bbox": (x, y, w, h), "area": int, "mask_value": int}
    """
    seg = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise FileNotFoundError(segmentation_path)
    if seg.ndim == 3:
        seg_flat = seg[:, :, 0].astype(np.int32) * 65536 + \
                   seg[:, :, 1].astype(np.int32) * 256 + \
                   seg[:, :, 2].astype(np.int32)
    else:
        seg_flat = seg.astype(np.int32)

    total_px = seg_flat.shape[0] * seg_flat.shape[1]
    detections = []
    unique_vals = np.unique(seg_flat)
    for val in unique_vals:
        if val == 0:
            continue
        mask = (seg_flat == val).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area or (area / total_px) > max_area_frac:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        detections.append({"bbox": (x, y, w, h), "area": area, "mask_value": int(val)})
    return detections


def bbox_iou(box_a, box_b):
    """IoU between two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def draw_detections(image_path, detections, out_path):
    img = cv2.imread(image_path)
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    return out_path
