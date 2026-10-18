#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 逐图导出：A3/A4/A5/A6 逐图检测结果 + 边缘检测结果。"""
import sys, os, io, json, time, argparse, importlib
import numpy as np
import cv2
from PIL import Image
import requests

EXP = "/app/Salience-DETR/experiments"
sys.path.insert(0, EXP)
OUT = os.path.join(EXP, "p11_dump")
os.makedirs(OUT, exist_ok=True)
GT_PATH = "/app/datasets/afo_clean/annotations/instances_test.json"
IMG_DIR = "/app/datasets/afo_clean/images/test"
YOLO_W  = "/app/datasets/afo_clean/weights/yolov8s_afo_clean/weights/best.pt"
CLOUD   = "http://localhost:8000/detect"

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="all", choices=["all", "edge", "jpeg", "a6", "a5"])
ap.add_argument("--conf", type=float, default=0.05)
a = ap.parse_args()


def jclean(o):
    if isinstance(o, dict):  return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [jclean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return [jclean(v) for v in o.tolist()]
    return o


def send_cloud(rgb, conf=None, retries=3):
    conf = a.conf if conf is None else conf
    last = None
    for k in range(retries):
        try:
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
            buf.seek(0)
            r = requests.post(CLOUD, files={"file": ("img.jpg", buf, "image/jpeg")},
                              data={"conf_threshold": str(conf)}, timeout=120)
            return r.json().get("detections", [])
        except Exception as e:
            last = e; time.sleep(2)
    print("    [send_cloud 连续失败]", last, flush=True)
    return []


def to_coco(dets, img_id):
    out = []
    for d in dets:
        b = d["bbox"]
        out.append({"image_id": int(img_id),
                    "category_id": int(d.get("class_id", 0)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0))})
    return out


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(jclean(obj), open(p, "w"))
    print("  -> 已保存 %s" % p, flush=True)


def run_edge():
    from ultralytics import YOLO
    imgs = json.load(open(GT_PATH))["images"]
    print("边缘检测导出：%d 张（conf=0.01）" % len(imgs))
    model = YOLO(YOLO_W)
    rec = {}
    for i, im in enumerate(imgs):
        p = os.path.join(IMG_DIR, im["file_name"])
        if not os.path.exists(p): continue
        r = model(p, conf=0.01, verbose=False)[0]
        ds = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                               r.boxes.conf.cpu().numpy().tolist(),
                               r.boxes.cls.cpu().numpy().astype(int).tolist()):
                ds.append({"category_id": int(c),
                           "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": float(s)})
        rec[str(im["id"])] = ds
        if (i + 1) % 100 == 0: print("  %d/%d" % (i+1, len(imgs)), flush=True)
    save("edge.json", rec)


def run_jpeg():
    imgs = json.load(open(GT_PATH))["images"]
    for tag, s, q in [("A3", 0.40, 50), ("A4", 0.50, 95)]:
        print("== %s  s=%.2f q=%d ==" % (tag, s, q), flush=True)
        all_dets, kb, n = [], 0.0, 0
        for i, im in enumerate(imgs):
            p = os.path.join(IMG_DIR, im["file_name"])
            if not os.path.exists(p): continue
            bgr = cv2.imread(p); H, W = bgr.shape[:2]
            small = cv2.resize(bgr, (max(1, int(W*s)), max(1, int(H*s))), interpolation=cv2.INTER_AREA)
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=q)
            kb += len(buf.getvalue())/1024.0; n += 1
            buf.seek(0)
            back = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)
            cv_img = cv2.resize(back, (W, H), interpolation=cv2.INTER_LINEAR)
            all_dets.extend(to_coco(send_cloud(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)), im["id"]))
            if (i + 1) % 100 == 0: print("  %d/%d" % (i+1, len(imgs)), flush=True)
        save("%s_dets.json" % tag, all_dets)
        print("  %s: %d dets | %.1f KB/图 | %d 图" % (tag, len(all_dets), kb/max(1,n), n))


def run_via_module(modname, tag, extra_argv, dump_name):
    print("== %s：导入 %s 并挂钩 evaluate ==" % (tag, modname), flush=True)
    mod = importlib.import_module(modname)
    fn = getattr(mod, "evaluate", None)
    if fn is None or not callable(fn):
        print("!! %s 中找不到可调用的 evaluate" % modname); return
    box = []
    def patched(dets, t, img_ids, *args, **kw):
        try:
            box.append({"tag": t, "dets": list(dets),
                        "img_ids": sorted(int(x) for x in img_ids)})
            print("  [captured] tag=%s dets=%d imgs=%d" % (t, len(dets), len(img_ids)), flush=True)
        except Exception as e:
            print("  [capture warn]", e)
        return fn(dets, t, img_ids, *args, **kw)
    mod.evaluate = patched
    sys.argv = [modname + ".py"] + list(extra_argv)
    print("  调用 %s.main()，argv=%s" % (modname, sys.argv[1:]), flush=True)
    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        print("!! %s 没有 main()" % modname); return
    t0 = time.time(); main_fn()
    print("  完成，用时 %.1f 分钟，捕获 %d 次 evaluate" % ((time.time()-t0)/60.0, len(box)))
    if not box:
        print("!! 未捕获到 evaluate，导出失败"); return
    save(dump_name, box)


if True:
    print("输出目录：%s" % OUT)
    if a.which in ("all", "edge"): run_edge()
    if a.which in ("all", "jpeg"): run_jpeg()
    if a.which in ("all", "a6"):
        run_via_module("expE_evidence_aware", "A6", ["--max-images", "1000"], "A6_evaluates.json")
    if a.which in ("all", "a5"):
        run_via_module("expB_support_plus_exploration", "A5",
                       ["--max-images", "1000"], "A5_evaluates.json")
    print("\n全部完成。下一步：python -u p11_analysis.py --reps 30")
