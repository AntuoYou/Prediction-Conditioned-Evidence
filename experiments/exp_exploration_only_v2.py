import os, sys, json, io, time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import requests
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMAGES_DIR = "/app/data/seadronessee/compressed_version/val2017/"
ANN_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
EDGE_WEIGHTS = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"

PATCH_SIZE = 16
LOW_RES_SCALE = 8
EXPAND_RATIO = 0.2
EXP_BUDGET = 49
W_DISTANCE = 0.4
W_TEXTURE = 0.3
W_LOWCONF = 0.3
CLOUD_CONF = 0.05
NMS_IOU = 0.5
CONF_SPLIT = 0.7
MAX_IMAGES = 5

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    label: int
    score: float

@dataclass
class EdgeOutput:
    image_id: str
    image_shape: Tuple[int, int]
    detections: List[Detection] = field(default_factory=list)

    @property
    def high_confidence(self):
        return [d for d in self.detections if d.score >= CONF_SPLIT]

    @property
    def low_confidence(self):
        return [d for d in self.detections if d.score < CONF_SPLIT]

@dataclass
class Patch:
    x: int
    y: int
    size: int = 16
    source: str = "exploration"
    score: float = 0.0

def expand_bbox(bbox, alpha, image_shape, min_context=16):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = max(w * alpha, min_context)
    dy = max(h * alpha, min_context)
    H, W = image_shape
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(W, int(x2 + dx)), min(H, int(y2 + dy)))

def generate_support_regions(edge_output, alpha=0.5, min_context=16):
    regions = []
    for det in edge_output.detections:
        regions.append(expand_bbox(det.bbox, alpha, edge_output.image_shape, min_context))
    return regions

def get_support_mask(regions, image_shape, patch_size=16):
    H, W = image_shape
    gh, gw = H // patch_size, W // patch_size
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in regions:
        gy1 = max(0, y1 // patch_size)
        gy2 = min(gh, (y2 + patch_size - 1) // patch_size)
        gx1 = max(0, x1 // patch_size)
        gx2 = min(gw, (x2 + patch_size - 1) // patch_size)
        mask[gy1:gy2, gx1:gx2] = True
    return mask

def compute_patch_texture(image, x, y, size=16):
    patch = image[y:y+size, x:x+size]
    if patch.size == 0:
        return 0.0
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(patch.std())

def compute_patch_distance(x, y, size, support_regions, image_shape):
    if not support_regions:
        return 1.0
    px_center = (x + size / 2, y + size / 2)
    diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    min_dist = float('inf')
    for x1, y1, x2, y2 in support_regions:
        dx = max(x1 - px_center[0], 0, px_center[0] - x2)
        dy = max(y1 - px_center[1], 0, px_center[1] - y2)
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = min(min_dist, dist)
    return float(min_dist / diag)

def compute_low_conf_density(x, y, size, low_conf_dets, sigma=50.0):
    if not low_conf_dets:
        return 0.0
    px_center = np.array([x + size / 2, y + size / 2])
    density = 0.0
    for det in low_conf_dets:
        dx1, dy1, dx2, dy2 = det.bbox
        det_center = np.array([(dx1 + dx2) / 2, (dy1 + dy2) / 2])
        dist = np.linalg.norm(px_center - det_center)
        density += np.exp(-dist / sigma)
    return float(density)

def generate_exploration_candidates(image, support_mask, patch_size=16, stride=16):
    H, W = image.shape[:2]
    gh, gw = support_mask.shape
    candidates = []
    for gy in range(gh):
        for gx in range(gw):
            if not support_mask[gy, gx]:
                x = gx * patch_size
                y = gy * patch_size
                if x + patch_size <= W and y + patch_size <= H:
                    candidates.append((x, y))
    return candidates

def score_exploration_candidates(image, candidates, edge_output,
                                  support_regions, patch_size=16,
                                  w_distance=0.4, w_texture=0.3, w_lowconf=0.3):
    low_conf_dets = edge_output.low_confidence
    scores_raw = []
    for x, y in candidates:
        d_score = compute_patch_distance(x, y, patch_size, support_regions, edge_output.image_shape)
        t_score = compute_patch_texture(image, x, y, patch_size)
        lc_score = compute_low_conf_density(x, y, patch_size, low_conf_dets)
        scores_raw.append((x, y, d_score, t_score, lc_score))
    textures = np.array([s[3] for s in scores_raw])
    lowconfs = np.array([s[4] for s in scores_raw])
    tex_max = textures.max() if textures.max() > 0 else 1.0
    lc_max = lowconfs.max() if lowconfs.max() > 0 else 1.0
    scored = []
    for x, y, d_score, t_score, lc_score in scores_raw:
        d_norm = d_score
        t_norm = t_score / tex_max
        lc_norm = lc_score / lc_max
        s = w_distance * d_norm + w_texture * t_norm + w_lowconf * lc_norm
        scored.append((x, y, float(s)))
    return scored

def select_exploration_patches(scored_candidates, budget_patches):
    scored_candidates.sort(key=lambda t: t[2], reverse=True)
    selected = scored_candidates[:budget_patches]
    patches = [Patch(x=x, y=y, size=16, source="exploration", score=score)
               for x, y, score in selected]
    return patches

def compute_exploration_patches(image, edge_output, support_regions,
                                 support_mask, budget_patches,
                                 patch_size=16,
                                 w_distance=0.4, w_texture=0.3, w_lowconf=0.3):
    candidates = generate_exploration_candidates(image, support_mask, patch_size)
    if not candidates:
        return []
    scored = score_exploration_candidates(image, candidates, edge_output,
                                           support_regions, patch_size,
                                           w_distance, w_texture, w_lowconf)
    patches = select_exploration_patches(scored, budget_patches)
    return patches

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def build_lowres_canvas(img, patches):
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // LOW_RES_SCALE, H // LOW_RES_SCALE))
    canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    for p in patches:
        x, y, s = p.x, p.y, p.size
        if 0 <= y < H - s + 1 and 0 <= x < W - s + 1:
            canvas[y:y+s, x:x+s] = img[y:y+s, x:x+s]
    return canvas

def send_cloud(canvas, conf=CLOUD_CONF):
    pil = Image.fromarray(canvas)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(
        "http://localhost:8000/detect",
        files={"file": ("img.png", buf, "image/png")},
        params={"conf": str(conf)},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    dets = []
    boxes = data.get("boxes", data.get("predictions", []))
    scores = data.get("scores", data.get("confidences", []))
    labels = data.get("labels", data.get("classes", []))
    if not labels:
        labels = [0] * len(boxes)
    for b, s, l in zip(boxes, scores, labels):
        dets.append(Detection(
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            label=int(l), score=float(s)
        ))
    return dets

def nms_fuse(edge_high, cloud_dets, iou_thresh=NMS_IOU):
    all_dets = list(edge_high) + list(cloud_dets)
    all_dets.sort(key=lambda d: d.score, reverse=True)
    fused = []
    for det in all_dets:
        keep = True
        for kept in fused:
            if det.label == kept.label:
                if compute_iou(det.bbox, kept.bbox) > iou_thresh:
                    keep = False
                    break
        if keep:
            fused.append(det)
    return fused

def detect_edge(img, image_id):
    from ultralytics import YOLO
    model = YOLO(EDGE_WEIGHTS)
    try:
        import torch_npu
        model.to("npu:0")
    except:
        pass
    H, W = img.shape[:2]
    results = model(img, conf=0.01, iou=0.7, imgsz=640, verbose=False, device="npu:0")
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)
        for bbox, score, label in zip(boxes, scores, labels):
            detections.append(Detection(
                bbox=tuple(bbox.tolist()), label=int(label), score=float(score)
            ))
    return EdgeOutput(image_id=image_id, image_shape=(H, W), detections=detections)

def evaluate(results_list, name, img_ids_used):
    coco_gt = COCO(ANN_PATH)
    coco_dt = coco_gt.loadRes(results_list)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = list(img_ids_used)
    e.evaluate(); e.accumulate(); e.summarize()
    return {
        "AP50": float(e.stats[1]),
        "mAP": float(e.stats[0]),
        "AP75": float(e.stats[2]),
        "APsmall": float(e.stats[3]),
        "APmedium": float(e.stats[4]),
        "APlarge": float(e.stats[5]),
        "AR": float(e.stats[8]),
        "num_dets": len(results_list),
    }

def main():
    print("[1/4] 加载 GT...")
    coco_gt = COCO(ANN_PATH)
    image_map = {img["id"]: img for img in coco_gt.dataset["images"]}
    split_files = [img["file_name"] for img in coco_gt.dataset["images"][:MAX_IMAGES]]

    print(f"[2/4] 加载边缘检测器...")
    from ultralytics import YOLO
    yolo = YOLO(EDGE_WEIGHTS)
    try:
        import torch_npu
        yolo.to("npu:0")
    except:
        pass

    results_coco = []
    img_ids_used = set()
    total_payload = 0

    print(f"[3/4] 跑 exploration-only (正确算法 + 低分辨率背景, {len(split_files)} 张)...")
    for fname in tqdm(split_files):
        img_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            continue
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
        H, W = img.shape[:2]

        coco_img_id = None
        for cid, cinfo in image_map.items():
            if cinfo["file_name"] == fname:
                coco_img_id = cid
                break
        if coco_img_id is None:
            continue
        img_ids_used.add(coco_img_id)

        edge_results = yolo(img, conf=0.01, iou=0.7, imgsz=640, verbose=False, device="npu:0")
        detections = []
        if edge_results[0].boxes is not None:
            boxes = edge_results[0].boxes.xyxy.cpu().numpy()
            scores = edge_results[0].boxes.conf.cpu().numpy()
            labels = edge_results[0].boxes.cls.cpu().numpy().astype(int)
            for bbox, score, label in zip(boxes, scores, labels):
                detections.append(Detection(
                    bbox=tuple(bbox.tolist()), label=int(label), score=float(score)
                ))
        edge_out = EdgeOutput(image_id=fname, image_shape=(H, W), detections=detections)
        edge_high = [d for d in detections if d.score >= CONF_SPLIT]

        support_regions = generate_support_regions(edge_out, alpha=EXPAND_RATIO)
        support_mask = get_support_mask(support_regions, (H, W), PATCH_SIZE)

        exp_patches = compute_exploration_patches(
            img, edge_out, support_regions, support_mask,
            budget_patches=EXP_BUDGET,
            patch_size=PATCH_SIZE,
            w_distance=W_DISTANCE,
            w_texture=W_TEXTURE,
            w_lowconf=W_LOWCONF,
        )

        canvas = build_lowres_canvas(img, exp_patches)
        cloud_dets = send_cloud(canvas, CLOUD_CONF)
        fused = nms_fuse(edge_high, cloud_dets)

        payload = len(exp_patches) * (PATCH_SIZE * PATCH_SIZE * 3 + 4) + 500
        total_payload += payload

        for d in fused:
            x1, y1, x2, y2 = d.bbox
            results_coco.append({
                "image_id": coco_img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(d.score),
            })

    print("\n[4/4] 评估...")
    metrics = evaluate(results_coco, "exploration_only", img_ids_used)
    metrics["avg_payload_bytes"] = total_payload / len(img_ids_used)
    metrics["avg_payload_kb"] = total_payload / len(img_ids_used) / 1024
    metrics["num_images"] = len(img_ids_used)

    print("\n=== Exploration-only (正确算法 + 低分辨率背景) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out = "exploration_only_v2_results.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果保存: {out}")

if __name__ == "__main__":
    main()
