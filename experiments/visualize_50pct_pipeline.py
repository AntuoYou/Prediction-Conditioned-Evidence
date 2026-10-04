#!/usr/bin/env python3
"""
50% 云边协同 Pipeline 可视化
============================
选若干张图片，生成对比图：
  左: 原图 + GT(绿) + YOLO全检测(蓝)
  右: 原图 + 融合结果(红) + 裁剪区域(黄框)
保存到 /app/Salience-DETR/experiments/vis_50pct/
"""

import json, os, time, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from collections import defaultdict

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_SERVER_URL = "http://localhost:8000"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/vis_50pct/"

SPLIT_THRESHOLD = 0.7
YOLO_CONF = 0.01
CLOUD_CONF = 0.05
SCALE_FACTOR = 0.50
NMS_IOU = 0.5
MATCH_IOU = 0.5

CAT_NAMES = {1:"swimmer", 2:"boat", 3:"jetski", 4:"life_saving", 5:"buoy"}
CAT_COLORS = {1:(255,100,100), 2:(100,100,255), 3:(255,255,100), 4:(255,100,255), 5:(100,255,255)}

def load_gt(path):
    with open(path) as f:
        coco = json.load(f)
    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)
    return coco, img_map, ann_map

def scale_to_gt(bbox, aw, ah, gw, gh):
    if aw == gw and ah == gh:
        return list(bbox)
    sx, sy = gw/aw, gh/ah
    return [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]

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

def draw_boxes(img_pil, dets, color_fn, gt_anns=None):
    """在图片上画框, dets=[{bbox:[x1,y1,x2,y2], score, class_id}] (实际图片空间)"""
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # GT 绿框
    if gt_anns:
        for ann in gt_anns:
            x,y,w,h = ann["bbox"]
            # GT 空间 → 实际图片空间
            draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=2)
            draw.text((x, y-12), f"GT:{CAT_NAMES.get(ann['category_id'],'?')}", fill=(0,255,0), font=font)

    for d in dets:
        x1,y1,x2,y2 = d["bbox"] if len(d["bbox"])==4 and d["bbox"][2]>d["bbox"][0] else coco_to_xyxy(d["bbox"])
        cls = d.get("class_id", d.get("category_id", 0))
        score = d.get("score", 0)
        color = color_fn(cls) if callable(color_fn) else color_fn
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{CAT_NAMES.get(cls,'?')} {score:.2f}"
        draw.text((x1, y1-14), label, fill=color, font=font)

    return img_pil

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    coco, img_map, ann_map = load_gt(GT_PATH)

    # 选有标注的图片，挑前10张（可改数量）
    image_ids = sorted(ann_map.keys())[:10]
    print(f"[可视化] 处理 {len(image_ids)} 张图片 -> {OUTPUT_DIR}")

    for idx, img_id in enumerate(image_ids):
        img_info = img_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        img_array = np.array(Image.open(img_path).convert("RGB"))
        actual_h, actual_w = img_array.shape[:2]
        gt_w, gt_h = img_info["width"], img_info["height"]
        gt_anns = ann_map[img_id]

        # GT bbox → 实际图片空间
        gt_actual = []
        for ann in gt_anns:
            sx, sy = actual_w/gt_w, actual_h/gt_h
            gt_actual.append({
                "bbox": [ann["bbox"][0]*sx, ann["bbox"][1]*sy,
                         ann["bbox"][2]*sx, ann["bbox"][3]*sy],
                "category_id": ann["category_id"],
            })

        # YOLO 推理
        yolo_dets = run_yolo(model, img_path)
        if not yolo_dets:
            print(f"  [{idx+1}] img_id={img_id} 无检测, 跳过")
            continue

        # 分流
        high_conf = [d for d in yolo_dets if d["score"] >= SPLIT_THRESHOLD]
        low_conf = [d for d in yolo_dets if d["score"] < SPLIT_THRESHOLD and d["score"] >= 0.05]

        # 裁剪 + 云端
        cloud_dets_actual = []
        crop_region = None
        if low_conf:
            crop_region = compute_crop(actual_w, actual_h, [d["bbox"] for d in low_conf])
            cx1,cy1,cx2,cy2 = crop_region
            crop = img_array[cy1:cy2, cx1:cx2].copy()
            cloud_raw = send_cloud(crop)

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

        # NMS 融合 (都在实际图片空间, 转成 COCO xywh)
        fused_all = []
        for d in high_conf:
            xywh = xyxy_to_xywh(*d["bbox"])
            fused_all.append({"category_id": d["class_id"]+1, "bbox": xywh, "score": d["score"]})
        for d in cloud_dets_actual:
            xywh = xyxy_to_xywh(*d["bbox"])
            fused_all.append({"category_id": d["class_id"], "bbox": xywh, "score": d["score"]})
        fused_nms = apply_nms(fused_all)
        # 转回 xyxy 实际空间
        fused_xyxy = [{"bbox": coco_to_xyxy(d["bbox"]), "score": d["score"], "class_id": d["category_id"]} for d in fused_nms]

        # ===== 画图 =====
        # 左图: 原图 + GT(绿) + YOLO全检测(蓝)
        img_left = Image.fromarray(img_array.copy())
        yolo_xyxy = [{"bbox": d["bbox"], "score": d["score"], "class_id": d["class_id"]+1} for d in yolo_dets]
        img_left = draw_boxes(img_left, yolo_xyxy, lambda c: (100,100,255))

        # 画GT框 (手动画, 因为GT是COCO格式)
        draw = ImageDraw.Draw(img_left)
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        for ann in gt_anns:
            sx, sy = actual_w/gt_w, actual_h/gt_h
            x, y = ann["bbox"][0]*sx, ann["bbox"][1]*sy
            w, h = ann["bbox"][2]*sx, ann["bbox"][3]*sy
            draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=2)

        # 右图: 原图 + 融合结果(红) + 裁剪区域(黄框)
        img_right = Image.fromarray(img_array.copy())
        img_right = draw_boxes(img_right, fused_xyxy, lambda c: (255,50,50))

        # 画裁剪区域
        if crop_region:
            cx1,cy1,cx2,cy2 = crop_region
            draw2 = ImageDraw.Draw(img_right)
            draw2.rectangle([cx1,cy1,cx2,cy2], outline=(255,255,0), width=4)
            draw2.text((cx1+5, cy1+5), "50% Crop ROI", fill=(255,255,0), font=font)

        # 高置信 YOLO(蓝) 也画在右图上
        for d in high_conf:
            x1,y1,x2,y2 = d["bbox"]
            draw2 = ImageDraw.Draw(img_right)
            draw2.rectangle([x1,y1,x2,y2], outline=(100,100,255), width=2)

        # 拼接左右
        gap = 20
        total_w = img_left.width + gap + img_right.width
        total_h = max(img_left.height, img_right.height) + 40
        canvas = Image.new("RGB", (total_w, total_h), (40,40,40))

        # 标题
        draw_c = ImageDraw.Draw(canvas)
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font_title = font
        draw_c.text((10, 8), "Before: GT(green) + YOLO(blue)", fill=(200,200,200), font=font_title)
        draw_c.text((img_left.width + gap + 10, 8), "After Fused: YOLO high(blue) + Cloud(red) + Crop(yellow)", fill=(200,200,200), font=font_title)

        canvas.paste(img_left, (0, 40))
        canvas.paste(img_right, (img_left.width + gap, 40))

        # 缩小保存
        scale = 1600 / canvas.width
        if scale < 1:
            canvas = canvas.resize((int(canvas.width*scale), int(canvas.height*scale)), Image.LANCZOS)

        fname = f"vis_{idx+1:02d}_img{img_id}.jpg"
        canvas.save(os.path.join(OUTPUT_DIR, fname), quality=90)
        print(f"  [{idx+1}/{len(image_ids)}] {fname} | high={len(high_conf)} low={len(low_conf)} cloud={len(cloud_dets_actual)} fused={len(fused_xyxy)}")

    print(f"\n[完成] {len(image_ids)} 张可视化图片 -> {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
