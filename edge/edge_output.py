"""高置信度结果直接返回 — JSON 格式化 + 可视化"""

import json
import cv2
import numpy as np
from typing import List, Optional
from split_by_confidence import Detection, SplitResult

COLOR_MAP = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

def format_edge_results(split_result: SplitResult,
                        image_id: str = "", latency_ms: float = 0.0) -> dict:
    return {
        "image_id": image_id, "source": "edge",
        "model": "YOLOv8s-AscendNPU",
        "latency_ms": round(latency_ms, 2),
        "total_detections": len(split_result.high_conf_detections),
        "cloud_pending": split_result.edge_output["num_cloud_requests"],
        "detections": [
            {"id": i, "bbox": {"x1": round(d.bbox[0],1), "y1": round(d.bbox[1],1),
              "x2": round(d.bbox[2],1), "y2": round(d.bbox[3],1)},
             "confidence": round(d.confidence,4), "class_id": d.class_id,
             "class_name": d.class_name}
            for i, d in enumerate(split_result.high_conf_detections)
        ]
    }

def visualize_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = COLOR_MAP[d.class_id % len(COLOR_MAP)]
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1-th-4), (x1+tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return vis

if __name__ == "__main__":
    from split_by_confidence import split_by_confidence
    dets = [
        Detection((100,200,180,310), 0.92, 0, "person"),
        Detection((300,150,420,280), 0.85, 1, "boat"),
        Detection((500,400,530,450), 0.45, 0, "swimmer"),
    ]
    result = split_by_confidence(dets)
    out = format_edge_results(result, image_id="test_001", latency_ms=15.3)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("[OK] edge_output.py 验证通过")
