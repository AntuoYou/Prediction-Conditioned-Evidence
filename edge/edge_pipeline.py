"""
端到端云边协同推理 pipeline
流程: YOLO检测 → 置信度分流 → 高置信度返回 / ROI裁剪 → 上传云端DETR → 合并
"""

import sys
import os
import time
import json
import cv2
import torch
import torch_npu
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, '/app/Salience-DETR')
from ultralytics import YOLO

from split_by_confidence import Detection, split_by_confidence
from edge_output import format_edge_results, visualize_detections
from roi_crop import prepare_cloud_requests
from client import CloudClient

# ============================================================
# 配置
# ============================================================
YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
YOLO_IMG_DIR = "/app/data/yolo_seadronessee/images/val"
DETR_IMG_DIR = "/app/data/seadronessee/compressed_version/val2017"
CLOUD_URL = "http://127.0.0.1:8000"

YOLO_CONF = 0.25      # YOLO 检测置信度阈值
SPLIT_CONF = 0.7      # 分流阈值
IOU_THRESH = 0.45     # YOLO NMS IoU 阈值
CLOUD_CONF = 0.3      # 云端 DETR 置信度阈值
USE_FP16 = True

COCO_NAMES = {0:"swimmer", 1:"boat", 2:"jetski", 3:"life_saving_appliances", 4:"buoy"}
SEA_NAMES = ["ignored","swimmer","boat","jetski","life_saving_appliances"]

# ============================================================
# YOLO 推理
# ============================================================
yolo_model = None
yolo_device = None

def init_yolo():
    global yolo_model, yolo_device
    if yolo_model is not None:
        return
    yolo_device = torch.device("npu:0")
    print(f"[Pipeline] 加载 YOLO 模型: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(yolo_device)
    print("[Pipeline] YOLO 预热中...")
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imwrite("/tmp/yolo_warmup.jpg", dummy)
    yolo_model.predict(source="/tmp/yolo_warmup.jpg", conf=0.25, verbose=False, device=yolo_device)
    print("[Pipeline] YOLO 就绪")

def run_yolo(image_path: str) -> List[Detection]:
    """YOLOv8 推理，返回 Detection 列表"""
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        iou=IOU_THRESH,
        half=USE_FP16,
        verbose=False,
        device=yolo_device,
    )
    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = COCO_NAMES.get(cls_id, f"coco_{cls_id}")
                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))
    return detections

# ============================================================
# 结果合并
# ============================================================
def detection_to_dict(det, source: str = "edge") -> dict:
    """将 Detection 对象或云端 dict 统一转成最终 JSON 格式"""
    if isinstance(det, dict):
        bbox = det.get("bbox")
        if isinstance(bbox, dict):
            bbox_dict = {
                "x1": round(float(bbox["x1"]), 1),
                "y1": round(float(bbox["y1"]), 1),
                "x2": round(float(bbox["x2"]), 1),
                "y2": round(float(bbox["y2"]), 1),
            }
        else:
            bbox_dict = {
                "x1": round(float(bbox[0]), 1),
                "y1": round(float(bbox[1]), 1),
                "x2": round(float(bbox[2]), 1),
                "y2": round(float(bbox[3]), 1),
            }
        raw_cls_id = int(det.get("class_id", det.get("category_id", -1)))

        # 云端 DETR 在 ROI 裁剪图上检测，bbox 是 ROI 局部坐标；
        # cloud_refine 进入最终结果前必须映射回原图坐标。
        if source == "cloud_refine":
            crop_bbox = det.get("crop_bbox")
            if crop_bbox is not None:
                offset_x, offset_y = float(crop_bbox[0]), float(crop_bbox[1])
                bbox_dict = {
                    "x1": round(float(bbox_dict["x1"]) + offset_x, 1),
                    "y1": round(float(bbox_dict["y1"]) + offset_y, 1),
                    "x2": round(float(bbox_dict["x2"]) + offset_x, 1),
                    "y2": round(float(bbox_dict["y2"]) + offset_y, 1),
                }

        # 云端 DETR 输出是 COCO category_id: 1-5
        # 为了 Final NMS 和最终 JSON 内部一致，这里统一转成 YOLO 的 0-based: 0-4
        if source == "cloud_refine":
            out_cls_id = raw_cls_id - 1
        else:
            out_cls_id = raw_cls_id

        return {
            "bbox": bbox_dict,
            "confidence": round(float(det.get("confidence", det.get("score", 0.0))), 4),
            "class_id": int(out_cls_id),
            "class_name": det.get("class_name", str(raw_cls_id)),
            "source": source
        }

    return {
        "bbox": {
            "x1": round(float(det.bbox[0]), 1),
            "y1": round(float(det.bbox[1]), 1),
            "x2": round(float(det.bbox[2]), 1),
            "y2": round(float(det.bbox[3]), 1),
        },
        "confidence": round(float(det.confidence), 4),
        "class_id": int(det.class_id),
        "class_name": det.class_name,
        "source": source
    }

def get_det_score(det) -> float:
    if det is None:
        return 0.0
    if isinstance(det, dict):
        return float(det.get("confidence", det.get("score", 0.0)))
    return float(det.confidence)

def get_det_class_id(det) -> int:
    if det is None:
        return -1
    if isinstance(det, dict):
        return int(det.get("class_id", det.get("category_id", -1)))
    return int(det.class_id)

def get_edge_class_id(det, source: str = None) -> int:
    """
    统一转换到边缘 YOLO 的 0-based 类别体系：
    YOLO: 0-4
    Cloud DETR: COCO category_id 1-5，所以转成 0-4
    """
    cls_id = get_det_class_id(det)

    if isinstance(det, dict):
        # 云端 DETR 输出为 COCO category_id: 1-5
        return cls_id - 1

    # YOLO Detection 输出为 0-based: 0-4
    return cls_id

def get_coco_class_id_from_final(det: dict) -> int:
    """
    最终评估用：
    edge_high / yolo_fallback 来自 YOLO，0-based，需要 +1
    cloud_refine 来自云端 DETR，原始是 COCO 1-based，但在最终输出中我们会统一保存为 0-based，所以也 +1
    """
    return int(det["class_id"]) + 1

def select_best_cloud_detection(cloud_result: dict):
    """从单个 ROI 的云端结果中选择置信度最高的检测，并保留 ROI 到原图的映射信息"""
    if not cloud_result or not cloud_result.get("success", False):
        return None
    detections = cloud_result.get("detections", [])
    if not detections:
        return None

    crop_bbox = cloud_result.get("crop_bbox")
    low_conf_index = cloud_result.get("low_conf_index")

    for d in detections:
        if isinstance(d, dict):
            d["crop_bbox"] = crop_bbox
            d["low_conf_index"] = low_conf_index

    return max(detections, key=get_det_score)

def is_cloud_better(yolo_det,
                    cloud_det,
                    score_delta: float = 0.10,
                    cloud_min_conf: float = 0.50,
                    cloud_high_conf: float = 0.70) -> bool:
    """
    Cloud Better 判定规则：
    1. 云端无结果：不用云端
    2. 云端分数太低：不用云端
    3. 同类别时，云端分数至少比 YOLO 高 score_delta 才替换
    4. 不同类别时，云端必须达到较高置信度 cloud_high_conf 才替换
    """
    if cloud_det is None:
        return False

    yolo_score = get_det_score(yolo_det)
    cloud_score = get_det_score(cloud_det)

    if cloud_score < cloud_min_conf:
        return False

    # 统一到 YOLO 的 0-based 类别体系后再比较
    yolo_cls = get_edge_class_id(yolo_det)
    cloud_cls = get_edge_class_id(cloud_det)

    if cloud_cls == yolo_cls and cloud_score >= yolo_score + score_delta:
        return True

    if cloud_cls != yolo_cls and cloud_score >= cloud_high_conf:
        return True

    return False

def refine_low_conf_detections(low_conf_detections: List[Detection],
                               cloud_results: List[dict]) -> tuple:
    """对低置信 YOLO 结果进行云端精修：Cloud Better 则替换，否则保留 YOLO"""
    refined = []
    stats = {
        "cloud_refine_count": 0,
        "yolo_fallback_count": 0,
        "cloud_no_detection_count": 0,
        "cloud_not_better_count": 0
    }

    cloud_result_map = {
        cr.get("low_conf_index"): cr
        for cr in cloud_results
        if cr.get("low_conf_index") is not None
    }

    for i, yolo_det in enumerate(low_conf_detections):
        cloud_result = cloud_result_map.get(i, {"success": False, "detections": []})
        best_cloud_det = select_best_cloud_detection(cloud_result)

        if best_cloud_det is None:
            refined.append((yolo_det, "yolo_fallback"))
            stats["yolo_fallback_count"] += 1
            stats["cloud_no_detection_count"] += 1
            continue

        if is_cloud_better(yolo_det, best_cloud_det):
            refined.append((best_cloud_det, "cloud_refine"))
            stats["cloud_refine_count"] += 1
        else:
            refined.append((yolo_det, "yolo_fallback"))
            stats["yolo_fallback_count"] += 1
            stats["cloud_not_better_count"] += 1

    return refined, stats

def bbox_iou(box1: dict, box2: dict) -> float:
    x1 = max(float(box1["x1"]), float(box2["x1"]))
    y1 = max(float(box1["y1"]), float(box2["y1"]))
    x2 = min(float(box1["x2"]), float(box2["x2"]))
    y2 = min(float(box1["y2"]), float(box2["y2"]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1["x2"]) - float(box1["x1"])) * max(0.0, float(box1["y2"]) - float(box1["y1"]))
    area2 = max(0.0, float(box2["x2"]) - float(box2["x1"])) * max(0.0, float(box2["y2"]) - float(box2["y1"]))
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union

def final_nms(detections: List[dict], iou_threshold: float = 0.50) -> List[dict]:
    """按类别进行最终 NMS，去掉 YOLO 和云端精修之间可能产生的重复框"""
    if not detections:
        return []

    final = []
    class_ids = sorted(set(d["class_id"] for d in detections))

    for cls_id in class_ids:
        cls_dets = [d for d in detections if d["class_id"] == cls_id]
        cls_dets = sorted(cls_dets, key=lambda x: x["confidence"], reverse=True)

        kept = []
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [
                d for d in cls_dets
                if bbox_iou(best["bbox"], d["bbox"]) < iou_threshold
            ]

        final.extend(kept)

    return final

def merge_results(high_conf_detections: List[Detection],
                  refined_low_conf: List[tuple],
                  image_id: str = "",
                  latency_ms: float = 0.0,
                  refine_stats: dict = None,
                  nms_iou: float = 0.50) -> dict:
    """合并高置信 YOLO 和精修后的低置信结果，并执行 Final NMS"""
    all_dets = []

    for d in high_conf_detections:
        all_dets.append(detection_to_dict(d, source="edge_high"))

    for det, source in refined_low_conf:
        all_dets.append(detection_to_dict(det, source=source))

    before_nms_count = len(all_dets)
    all_dets = final_nms(all_dets, iou_threshold=nms_iou)
    after_nms_count = len(all_dets)

    edge_high_count = len(high_conf_detections)
    refine_stats = refine_stats or {}

    return {
        "image_id": image_id,
        "source": "edge_cloud_roi_refinement",
        "model": "YOLOv8s-AscendNPU + Cloud Salience-DETR",
        "latency_ms": round(latency_ms, 2),
        "total_detections": len(all_dets),
        "edge_count": edge_high_count + refine_stats.get("yolo_fallback_count", 0),
        "cloud_count": refine_stats.get("cloud_refine_count", 0),
        "edge_high_count": edge_high_count,
        "cloud_refine_count": refine_stats.get("cloud_refine_count", 0),
        "yolo_fallback_count": refine_stats.get("yolo_fallback_count", 0),
        "cloud_no_detection_count": refine_stats.get("cloud_no_detection_count", 0),
        "cloud_not_better_count": refine_stats.get("cloud_not_better_count", 0),
        "before_nms_count": before_nms_count,
        "after_nms_count": after_nms_count,
        "nms_removed_count": before_nms_count - after_nms_count,
        "detections": all_dets
    }

# ============================================================
# 主 pipeline
# ============================================================
def run_pipeline(image_path: str,
                 conf_threshold: float = SPLIT_CONF,
                 cloud_url: str = CLOUD_URL,
                 save_vis: bool = False,
                 vis_dir: str = "experiments/visualizations") -> dict:
    img_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    pipeline_output = {"image_id": img_name, "image_size": [img_w, img_h]}
    timings = {}

    # Step 1: YOLO 推理
    t0 = time.time()
    all_dets = run_yolo(image_path)
    timings["yolo_inference_ms"] = round((time.time() - t0) * 1000, 2)

    # Step 2: 置信度分流
    t0 = time.time()
    split_result = split_by_confidence(all_dets, conf_threshold=conf_threshold)
    edge_output = format_edge_results(split_result, image_id=img_name,
                                      latency_ms=timings["yolo_inference_ms"])
    timings["split_ms"] = round((time.time() - t0) * 1000, 2)

    # Step 3: ROI 裁剪
    t0 = time.time()
    cloud_requests = prepare_cloud_requests(img, split_result.low_conf_detections)
    timings["roi_crop_ms"] = round((time.time() - t0) * 1000, 2)
    total_bandwidth_kb = sum(r["data_kb"] for r in cloud_requests)
    timings["bandwidth_kb"] = round(total_bandwidth_kb, 1)

    # Step 4: 上传云端 DETR
    t0 = time.time()
    cloud_client = CloudClient(url=cloud_url)
    if cloud_requests and cloud_client.health():
        cloud_results = cloud_client.batch_detect(cloud_requests, conf_threshold=CLOUD_CONF)
    else:
        cloud_results = []
        if cloud_requests:
            print(f"  [WARN] 云端不可用，跳过 {len(cloud_requests)} 个 ROI")
    timings["cloud_inference_ms"] = round((time.time() - t0) * 1000, 2)

    # Step 5: ROI Refinement + Final NMS
    t0 = time.time()
    refined_low_conf, refine_stats = refine_low_conf_detections(
        split_result.low_conf_detections,
        cloud_results
    )
    print(
        f"  [ROI Refinement] cloud_refine={refine_stats['cloud_refine_count']}, "
        f"yolo_fallback={refine_stats['yolo_fallback_count']}, "
        f"cloud_no_detection={refine_stats['cloud_no_detection_count']}, "
        f"cloud_not_better={refine_stats['cloud_not_better_count']}"
    )

    merged = merge_results(
        split_result.high_conf_detections,
        refined_low_conf,
        image_id=img_name,
        latency_ms=timings["yolo_inference_ms"],
        refine_stats=refine_stats,
        nms_iou=0.50
    )
    timings["merge_ms"] = round((time.time() - t0) * 1000, 2)

    timings["total_ms"] = sum(timings[k] for k in ["yolo_inference_ms","split_ms",
                                                     "roi_crop_ms","cloud_inference_ms","merge_ms"])

    pipeline_output.update({
        "edge_count": merged["edge_count"],
        "cloud_count": merged["cloud_count"],
        "total_count": merged["total_detections"],
        "edge_high_count": merged["edge_high_count"],
        "cloud_refine_count": merged["cloud_refine_count"],
        "yolo_fallback_count": merged["yolo_fallback_count"],
        "cloud_no_detection_count": merged["cloud_no_detection_count"],
        "cloud_not_better_count": merged["cloud_not_better_count"],
        "before_nms_count": merged["before_nms_count"],
        "after_nms_count": merged["after_nms_count"],
        "nms_removed_count": merged["nms_removed_count"],
        "split_threshold": conf_threshold,
        "timings": timings,
        "detections": merged["detections"]
    })

    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
        vis_img = img.copy()
        for d in merged["detections"]:
            bbox = d["bbox"]
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            else:
                x1, y1, x2, y2 = map(int, bbox)
            if d["source"] == "edge_high":
                color = (0, 255, 0)
            elif d["source"] == "cloud_refine":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            label = f"{d['class_name']} {d['confidence']:.2f} [{d['source']}]"
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 4)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(vis_img, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        vis_path = os.path.join(vis_dir, f"vis_{img_name}")
        cv2.imwrite(vis_path, vis_img)

    return pipeline_output

# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="云边协同推理 pipeline")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, default=YOLO_IMG_DIR, help="图像目录")
    parser.add_argument("--num", type=int, default=None, help="测试图像数量")
    parser.add_argument("--threshold", type=float, default=0.7, help="分流阈值")
    parser.add_argument("--cloud_url", type=str, default=CLOUD_URL, help="云端服务地址")
    parser.add_argument("--save_vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--output", type=str, default="experiments/week3_results.json", help="输出文件")
    args = parser.parse_args()

    init_yolo()

    if args.image:
        print(f"\n{'='*60}")
        print(f"  单张测试: {args.image}")
        print(f"{'='*60}")
        result = run_pipeline(args.image, conf_threshold=args.threshold,
                             cloud_url=args.cloud_url, save_vis=args.save_vis)
        print(f"  边缘检测: {result['edge_count']}")
        print(f"  云端检测: {result['cloud_count']}")
        print(f"  总计:     {result['total_count']}")
        print(f"  耗时:     {result['timings']}")
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {args.output}")
    else:
        img_dir = Path(args.dir)
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:args.num]
        print(f"\n{'='*60}")
        print(f"  批量测试: {len(images)} 张图像")
        print(f"  分流阈值: {args.threshold}")
        print(f"{'='*60}")

        all_results = []
        for i, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            print(f"\n[{i+1}/{len(images)}] {img_name}")
            result = run_pipeline(str(img_path), conf_threshold=args.threshold,
                                 cloud_url=args.cloud_url, save_vis=args.save_vis)
            t = result["timings"]
            print(f"  边缘: {result['edge_count']}, 云端: {result['cloud_count']}, "
                  f"总计: {result['total_count']}, "
                  f"YOLO: {t['yolo_inference_ms']}ms, "
                  f"云端: {t['cloud_inference_ms']}ms, "
                  f"总耗时: {t['total_ms']}ms")
            all_results.append(result)

        total_edge = sum(r["edge_count"] for r in all_results)
        total_cloud = sum(r["cloud_count"] for r in all_results)
        total_all = sum(r["total_count"] for r in all_results)
        avg_yolo = np.mean([r["timings"]["yolo_inference_ms"] for r in all_results])
        avg_cloud = np.mean([r["timings"]["cloud_inference_ms"] for r in all_results])
        avg_total = np.mean([r["timings"]["total_ms"] for r in all_results])
        avg_bw = np.mean([r["timings"]["bandwidth_kb"] for r in all_results])

        summary = {
            "num_images": len(all_results),
            "threshold": args.threshold,
            "total_edge": total_edge,
            "total_cloud": total_cloud,
            "total_all": total_all,
            "edge_ratio_pct": round(total_edge / max(total_all, 1) * 100, 1),
            "cloud_ratio_pct": round(total_cloud / max(total_all, 1) * 100, 1),
            "avg_yolo_ms": round(avg_yolo, 2),
            "avg_cloud_ms": round(avg_cloud, 2),
            "avg_total_ms": round(avg_total, 2),
            "avg_bandwidth_kb": round(avg_bw, 1),
            "results": all_results
        }

        print(f"\n{'='*60}")
        print("  汇总")
        print(f"{'='*60}")
        print(f"  图像数: {len(all_results)}")
        print(f"  边缘检测: {total_edge} ({summary['edge_ratio_pct']}%)")
        print(f"  云端检测: {total_cloud} ({summary['cloud_ratio_pct']}%)")
        print(f"  总检测数: {total_all}")
        print(f"  平均 YOLO 耗时: {avg_yolo:.1f}ms")
        print(f"  平均云端耗时: {avg_cloud:.1f}ms")
        print(f"  平均总耗时: {avg_total:.1f}ms")
        print(f"  平均上传带宽: {avg_bw:.1f} KB/图")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")
