#!/usr/bin/env python3
"""
云边协同 Pipeline 可视化 V2 — 每张图分4张输出
================================================
对每张图片分别输出:
  1. original_XX.jpg     原图（无框）
  2. roi_crop_XX.jpg      50% 裁剪区域内容（无框）
  3. yolo_detect_XX.jpg   原图 + YOLO全检测(蓝) + GT(绿)
  4. fused_result_XX.jpg  原图 + 融合结果(红) + 高置信YOLO(蓝) + 裁剪区域(黄)

用法:
  python visualize_pipeline_v2.py
  python visualize_pipeline_v2.py --num 5    # 只处理5张
"""

import json, os, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from collections import defaultdict

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_SERVER_URL = "http://localhost:8000"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/vis_pipeline_v2/"

SPLIT_THRESHOLD = 0.7
YOLO_CONF = 0.01
CLOUD_CONF = 0.05
SCALE_FACTOR = 0.50
NMS_IOU = 0.5

CAT_NAMES = {1:"swimmer", 2:"boat", 3:"jetski", 4:"life_saving", 5:"buoy"}
CAT_COLORS = {1:(255,80,80), 2:(80,80,255), 3:(255,200,0), 4:(255,80,255), 5:(80,255,255)}

def load_gt(path):
    with open(path) as f:
        coco = json.load(f)
    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)
    return coco, img_map, ann_map

def xyxy_to_xywh(x1,y1,x2,y2):
    return [x1,y1,x2-x1,y2-y1]

def coco_to_xyxy(b):
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def iou_xyxy(a, b):
    xa,ya = max(a[0],b[0]), max(a[1],b[1])
    xb,yb = min(a[2],b[2]), min(a[3],b[3])
    if xb<=xa or yb<=ya: return 0
    inter = (xb-xa)*(yb-ya)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def compute_crop(w, h, dets, sf=SCALE_FACTOR):
    xs1 = [d[0] for d in dets]; ys1 = [d[1] for d in dets]
    xs2 = [d[2] for d in dets]; ys2 = [d[3] for d in dets]
    cx = (min(xs1)+max(xs2))/2; cy = (min(ys1)+max(ys2))/2
    ss = np.sqrt(sf); cw, ch = int(w*ss), int(h*ss)
    x1, y1 = int(cx-cw/2), int(cy-ch/2)
    if x1<0: x1=0
    if y1<0: y1=0
    x2, y2 = x1+cw, y1+ch
    if x2>w: x2=w; x1=max(0,x2-cw)
    if y2>h: y2=h; y1=max(0,y2-ch)
    return x1,y1,x2,y2

def run_yolo(model, path, conf=YOLO_CONF):
    r = model(path, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    return [{"bbox":[float(xyxy[i][0]),float(xyxy[i][1]),float(xyxy[i][2]),float(xyxy[i][3])],
             "score":float(confs[i]), "class_id":int(clses[i])} for i in range(len(xyxy))]

def send_cloud(img, conf=CLOUD_CONF):
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    r = requests.post(f"{CLOUD_SERVER_URL}/detect",
        files={"file":("img.jpg", buf.getvalue(), "image/jpeg")},
        data={"conf_threshold": str(conf)}, timeout=60)
    if r.status_code != 200: return []
    d = r.json()
    if "results" in d:
        return [det for item in d["results"] for det in item.get("detections", [])]
    return d.get("detections", d if isinstance(d, list) else [])

def parse_bbox(b):
    if isinstance(b, list): return b[0],b[1],b[2],b[3]
    if isinstance(b, dict): return b["x1"],b["y1"],b["x2"],b["y2"]
    raise ValueError(str(type(b)))

def apply_nms(dets, iou=NMS_IOU):
    if not dets: return []
    by_cls = defaultdict(list)
    for d in dets: by_cls[d["category_id"]].append(d)
    res = []
    for cls, ds in by_cls.items():
        ds.sort(key=lambda x:-x["score"])
        kept = []
        for d in ds:
            if not any(iou_xyxy(coco_to_xyxy(d["bbox"]), coco_to_xyxy(k["bbox"])) > iou for k in kept):
                kept.append(d)
        res.extend(kept)
    return res

def get_font(size=14):
    for path in ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()

def draw_box(draw, bbox_xyxy, color, label, font):
    x1,y1,x2,y2 = bbox_xyxy
    draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
    # 标签背景
    tw = font.getlength(label) if hasattr(font, 'getlength') else len(label)*7
    draw.rectangle([x1, max(0,y1-16), x1+tw+4, y1], fill=color)
    draw.text((x1+2, max(0,y1-15)), label, fill=(255,255,255), font=font)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="处理图片数量")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    coco, img_map, ann_map = load_gt(GT_PATH)

    image_ids = sorted(ann_map.keys())[:args.num]
    print(f"[可视化] 处理 {len(image_ids)} 张图片 -> {OUTPUT_DIR}")
    print(f"每张图输出4个文件: original / roi_crop / yolo_detect / fused_result\n")

    font = get_font(14)

    for idx, img_id in enumerate(image_ids):
        img_info = img_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"  [{idx+1}] 跳过 {img_id}: 文件不存在")
            continue

        img_array = np.array(Image.open(img_path).convert("RGB"))
        actual_h, actual_w = img_array.shape[:2]
        gt_w, gt_h = img_info["width"], img_info["height"]
        gt_anns = ann_map[img_id]

        # ===== 1. 原图（无框）=====
        img_original = Image.fromarray(img_array.copy())
        # 缩小到合理大小
        scale = 1280 / max(actual_w, actual_h)
        if scale < 1:
            img_original = img_original.resize((int(actual_w*scale), int(actual_h*scale)), Image.LANCZOS)

        # ===== YOLO 推理 =====
        yolo_dets = run_yolo(model, img_path)
        if not yolo_dets:
            print(f"  [{idx+1}] img_id={img_id} 无检测, 跳过")
            continue

        high_conf = [d for d in yolo_dets if d["score"] >= SPLIT_THRESHOLD]
        low_conf = [d for d in yolo_dets if d["score"] < SPLIT_THRESHOLD and d["score"] >= 0.05]

        # ===== 2. ROI 裁剪图 =====
        crop_region = None
        cloud_dets_actual = []
        img_roi_crop = None

        if low_conf:
            crop_region = compute_crop(actual_w, actual_h, [d["bbox"] for d in low_conf])
            cx1,cy1,cx2,cy2 = crop_region
            crop_array = img_array[cy1:cy2, cx1:cx2].copy()
            img_roi_crop = Image.fromarray(crop_array)
            scale2 = 1280 / max(crop_array.shape[1], crop_array.shape[0])
            if scale2 < 1:
                img_roi_crop = img_roi_crop.resize(
                    (int(crop_array.shape[1]*scale2), int(crop_array.shape[0]*scale2)), Image.LANCZOS)

            # 云端推理
            cloud_raw = send_cloud(crop_array)
            for det in cloud_raw:
                bx1,by1,bx2,by2 = parse_bbox(det["bbox"])
                score = det.get("score", det.get("confidence", 0))
                cls_id = det.get("class_id", det.get("category_id", 0))
                if score < CLOUD_CONF: continue
                cloud_dets_actual.append({
                    "bbox": [cx1+bx1, cy1+by1, cx1+bx2, cy1+by2],
                    "score": float(score),
                    "class_id": cls_id,
                })
        else:
            print(f"  [{idx+1}] img_id={img_id} 无低置信检测, 无ROI裁剪")

        # ===== 3. YOLO 检测结果图 =====
        img_yolo = Image.fromarray(img_array.copy())
        draw_yolo = ImageDraw.Draw(img_yolo)

        # GT 绿框
        for ann in gt_anns:
            sx, sy = actual_w/gt_w, actual_h/gt_h
            x, y = ann["bbox"][0]*sx, ann["bbox"][1]*sy
            w, h = ann["bbox"][2]*sx, ann["bbox"][3]*sy
            draw_yolo.rectangle([x, y, x+w, y+h], outline=(0,200,0), width=2)

        # YOLO 蓝框
        for d in yolo_dets:
            cls = d["class_id"] + 1
            score = d["score"]
            color = (50,50,255) if score >= SPLIT_THRESHOLD else (100,100,180)
            label = f"{CAT_NAMES.get(cls,'?')} {score:.2f}"
            draw_box(draw_yolo, d["bbox"], color, label, font)

        # 画裁剪区域黄框
        if crop_region:
            cx1,cy1,cx2,cy2 = crop_region
            draw_yolo.rectangle([cx1,cy1,cx2,cy2], outline=(255,255,0), width=3)
            draw_yolo.text((cx1+5, cy1+5), "50% Crop ROI", fill=(255,255,0), font=font)

        scale3 = 1280 / max(actual_w, actual_h)
        if scale3 < 1:
            img_yolo = img_yolo.resize((int(actual_w*scale3), int(actual_h*scale3)), Image.LANCZOS)

        # ===== 4. 融合结果图 =====
        img_fused = Image.fromarray(img_array.copy())
        draw_fused = ImageDraw.Draw(img_fused)

        # GT 绿框（淡）
        for ann in gt_anns:
            sx, sy = actual_w/gt_w, actual_h/gt_h
            x, y = ann["bbox"][0]*sx, ann["bbox"][1]*sy
            w, h = ann["bbox"][2]*sx, ann["bbox"][3]*sy
            draw_fused.rectangle([x, y, x+w, y+h], outline=(0,180,0), width=2)

        # NMS 融合
        fused_all = []
        for d in high_conf:
            xywh = xyxy_to_xywh(*d["bbox"])
            fused_all.append({"category_id": d["class_id"]+1, "bbox": xywh, "score": d["score"]})
        for d in cloud_dets_actual:
            xywh = xyxy_to_xywh(*d["bbox"])
            fused_all.append({"category_id": d["class_id"], "bbox": xywh, "score": d["score"]})
        fused_nms = apply_nms(fused_all)

        # 高置信YOLO 蓝框
        high_gt_ids = set()
        for d in high_conf:
            cls = d["class_id"] + 1
            color = CAT_COLORS.get(cls, (50,50,255))
            label = f"{CAT_NAMES.get(cls,'?')} {d['score']:.2f} [edge]"
            draw_box(draw_fused, d["bbox"], (50,50,255), label, font)

        # 云端检测 红框
        for d in cloud_dets_actual:
            cls = d["class_id"]
            color = CAT_COLORS.get(cls, (255,50,50))
            label = f"{CAT_NAMES.get(cls,'?')} {d['score']:.2f} [cloud]"
            draw_box(draw_fused, d["bbox"], color, label, font)

        # 画裁剪区域黄框
        if crop_region:
            cx1,cy1,cx2,cy2 = crop_region
            draw_fused.rectangle([cx1,cy1,cx2,cy2], outline=(255,255,0), width=3)
            draw_fused.text((cx1+5, cy1+5), "50% Crop ROI", fill=(255,255,0), font=font)

        scale4 = 1280 / max(actual_w, actual_h)
        if scale4 < 1:
            img_fused = img_fused.resize((int(actual_w*scale4), int(actual_h*scale4)), Image.LANCZOS)

        # ===== 保存 =====
        prefix = f"{idx+1:02d}_img{img_id}"
        img_original.save(os.path.join(OUTPUT_DIR, f"original_{prefix}.jpg"), quality=90)
        if img_roi_crop:
            img_roi_crop.save(os.path.join(OUTPUT_DIR, f"roi_crop_{prefix}.jpg"), quality=90)
        img_yolo.save(os.path.join(OUTPUT_DIR, f"yolo_detect_{prefix}.jpg"), quality=90)
        img_fused.save(os.path.join(OUTPUT_DIR, f"fused_result_{prefix}.jpg"), quality=90)

        n_high = len(high_conf)
        n_low = len(low_conf)
        n_cloud = len(cloud_dets_actual)
        n_fused = len(fused_nms)
        print(f"  [{idx+1}/{len(image_ids)}] img_id={img_id} | high={n_high} low={n_low} cloud={n_cloud} fused={n_fused}")

    print(f"\n[完成] {len(image_ids)} 张图片 -> {OUTPUT_DIR}")
    print(f"每张图4个文件: original / roi_crop / yolo_detect / fused_result")

if __name__ == "__main__":
    main()
