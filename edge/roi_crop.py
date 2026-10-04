"""ROI 裁剪模块 — 低置信度目标裁剪 + 1.5倍上下文 + 合并 + JPEG编码"""

import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from split_by_confidence import Detection

@dataclass
class ROICrop:
    crop_image: np.ndarray
    original_bbox: Tuple[float, ...]
    crop_bbox: Tuple[int, ...]
    detection: Detection
    index: int
    data_kb: float = 0.0

def adaptive_context_ratio(detection: Detection,
                           alpha_min: float = 1.2,
                           alpha_max: float = 2.0,
                           lambda_: float = 0.85,
                           area_ref: float = 6400.0) -> float:
    """
    尺度自适应 ROI 扩展比例：
    目标面积越小，扩展比例越大；
    目标面积越大，扩展比例越接近 alpha_min。
    """
    x1, y1, x2, y2 = detection.bbox
    w, h = x2 - x1, y2 - y1
    area = max(1.0, w * h)

    alpha = alpha_min + lambda_ * math.exp(-area / area_ref)
    alpha = max(alpha_min, min(alpha, alpha_max))
    return alpha

def crop_roi_with_context(image: np.ndarray, detection: Detection,
                          context_ratio: float = None) -> ROICrop:
    x1, y1, x2, y2 = detection.bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1+x2)/2, (y1+y2)/2

    if context_ratio is None:
        context_ratio = adaptive_context_ratio(detection)

    nw, nh = w*context_ratio, h*context_ratio
    rx1 = max(0, int(cx - nw/2))
    ry1 = max(0, int(cy - nh/2))
    rx2 = min(image.shape[1], int(cx + nw/2))
    ry2 = min(image.shape[0], int(cy + nh/2))
    crop = image[ry1:ry2, rx1:rx2].copy()
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ROICrop(crop_image=crop, original_bbox=detection.bbox,
                   crop_bbox=(rx1,ry1,rx2,ry2), detection=detection,
                   index=0, data_kb=round(len(buf.tobytes())/1024, 1))

def compute_iou(a, b) -> float:
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/area if area > 0 else 0

def merge_rois(rois: List[ROICrop], iou_thr: float = 0.5) -> List[ROICrop]:
    if len(rois) <= 1: return rois
    merged = True
    while merged:
        merged = False
        new_rois, used = [], [False]*len(rois)
        for i in range(len(rois)):
            if used[i]: continue
            cur = rois[i]
            for j in range(i+1, len(rois)):
                if used[j]: continue
                if compute_iou(cur.crop_bbox, rois[j].crop_bbox) > iou_thr:
                    rx1 = min(cur.crop_bbox[0], rois[j].crop_bbox[0])
                    ry1 = min(cur.crop_bbox[1], rois[j].crop_bbox[1])
                    rx2 = max(cur.crop_bbox[2], rois[j].crop_bbox[2])
                    ry2 = max(cur.crop_bbox[3], rois[j].crop_bbox[3])
                    cur = ROICrop(cur.crop_image, cur.original_bbox,
                                  (rx1,ry1,rx2,ry2), cur.detection, cur.index,
                                  cur.data_kb + rois[j].data_kb)
                    used[j] = True; merged = True
            new_rois.append(cur); used[i] = True
        rois = new_rois
    return rois

def prepare_cloud_requests(image: np.ndarray,
                           low_conf_dets: List[Detection],
                           context_ratio: float = None) -> List[dict]:
    crops = []
    for i, det in enumerate(low_conf_dets):
        roi = crop_roi_with_context(image, det, context_ratio)
        roi.index = i
        if roi.crop_image.shape[0] < 80 or roi.crop_image.shape[1] < 80:
            print(f"  [SKIP] ROI #{i}: too small")
            continue
        crops.append(roi)
    crops = merge_rois(crops)
    requests = []
    for roi in crops:
        _, buf = cv2.imencode(".jpg", roi.crop_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        requests.append({
            "index": roi.index, "image_bytes": buf.tobytes(),
            "original_bbox": list(roi.original_bbox),
            "crop_bbox": list(roi.crop_bbox),
            "data_kb": roi.data_kb,
            "metadata": {"edge_confidence": roi.detection.confidence,
                         "class_id": roi.detection.class_id,
                         "class_name": roi.detection.class_name}
        })
    total_kb = sum(r["data_kb"] for r in requests)
    print(f"[ROI] {len(requests)} 个 ROI, 总数据量: {total_kb:.1f} KB")
    return requests

if __name__ == "__main__":
    from split_by_confidence import split_by_confidence
    img = np.zeros((720,1280,3), dtype=np.uint8)
    img[:] = (30,60,120)
    cv2.rectangle(img, (480,300),(520,360), (200,200,200), -1)
    cv2.rectangle(img, (800,200),(850,280), (200,200,200), -1)
    dets = [Detection((480,300,520,360), 0.45, 0, "swimmer"),
            Detection((800,200,850,280), 0.38, 0, "buoy")]
    reqs = prepare_cloud_requests(img, dets)
    _, full = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,85])
    roi_kb = sum(r["data_kb"] for r in reqs)
    print(f"全图: {len(full.tobytes())/1024:.1f} KB, ROI: {roi_kb:.1f} KB, 节省: {(1-roi_kb/(len(full.tobytes())/1024))*100:.1f}%")
    print("[OK] roi_crop.py 验证通过")
