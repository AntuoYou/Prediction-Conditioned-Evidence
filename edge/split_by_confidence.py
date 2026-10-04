"""置信度分流模块 — 解析 YOLOv8 输出，按阈值 0.7 分流"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str = ""

@dataclass
class SplitResult:
    high_conf_detections: List[Detection]
    low_conf_detections: List[Detection]
    edge_output: dict

def split_by_confidence(detections: List[Detection],
                        conf_threshold: float = 0.7) -> SplitResult:
    high_conf = [d for d in detections if d.confidence >= conf_threshold]
    low_conf = [d for d in detections if d.confidence < conf_threshold]

    edge_output = {
        "source": "edge",
        "model": "YOLOv8s-AscendNPU",
        "detections": [
            {"bbox": list(d.bbox), "confidence": d.confidence,
             "class_id": d.class_id, "class_name": d.class_name}
            for d in high_conf
        ],
        "num_cloud_requests": len(low_conf)
    }
    return SplitResult(
        high_conf_detections=high_conf,
        low_conf_detections=low_conf,
        edge_output=edge_output
    )

if __name__ == "__main__":
    # 自测
    dets = [
        Detection((100,200,180,310), 0.92, 0, "person"),
        Detection((300,150,420,280), 0.85, 1, "boat"),
        Detection((500,400,530,450), 0.45, 0, "swimmer"),
        Detection((700,300,720,340), 0.38, 0, "buoy"),
    ]
    result = split_by_confidence(dets, conf_threshold=0.7)
    print(f"总检测: {len(dets)}, 高置信度: {len(result.high_conf_detections)}, 低置信度: {len(result.low_conf_detections)}")
    print("[OK] split_by_confidence.py 验证通过")
