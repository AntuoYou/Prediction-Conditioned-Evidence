import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

YOLO_TO_COCO_CAT = {
    0: 1,  # swimmer
    1: 2,  # boat
    2: 3,  # jetski
    3: 4,  # life_saving_appliances
    4: 5,  # buoy
}

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [
        float(x1),
        float(y1),
        float(max(0.0, x2 - x1)),
        float(max(0.0, y2 - y1)),
    ]

def area_xyxy(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def iou_xyxy(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0

def intersection_ratio(candidate_box, roi_box):
    ix1 = max(candidate_box[0], roi_box[0])
    iy1 = max(candidate_box[1], roi_box[1])
    ix2 = min(candidate_box[2], roi_box[2])
    iy2 = min(candidate_box[3], roi_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = area_xyxy(candidate_box)
    return inter / a if a > 0 else 0.0

def center_inside(box, roi):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

def adaptive_context_ratio(box, img_w, img_h, cat_id):
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    area_ratio = (w * h) / max(1.0, img_w * img_h)

    if cat_id in (1, 5):
        if area_ratio < 0.0005:
            return 4.0
        if area_ratio < 0.002:
            return 3.0
        return 2.5

    if area_ratio < 0.001:
        return 2.5
    if area_ratio < 0.005:
        return 2.0
    return 1.5

def expand_roi(box, img_w, img_h, cat_id):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ratio = adaptive_context_ratio(box, img_w, img_h, cat_id)

    nw = w * ratio
    nh = h * ratio

    rx1 = max(0.0, cx - nw / 2.0)
    ry1 = max(0.0, cy - nh / 2.0)
    rx2 = min(float(img_w), cx + nw / 2.0)
    ry2 = min(float(img_h), cy + nh / 2.0)

    return [rx1, ry1, rx2, ry2]

def nms_per_class(dets, iou_thr=0.6):
    by_class = defaultdict(list)
    for d in dets:
        by_class[d["category_id"]].append(d)

    kept = []
    for _, items in by_class.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        used = [False] * len(items)

        for i, d in enumerate(items):
            if used[i]:
                continue
            kept.append(d)
            box_i = xywh_to_xyxy(d["bbox"])

            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                box_j = xywh_to_xyxy(items[j]["bbox"])
                if iou_xyxy(box_i, box_j) >= iou_thr:
                    used[j] = True

    return kept

def load_coco_gt(path):
    with open(path, "r") as f:
        gt = json.load(f)
    return gt["images"]

def build_file_to_image_id(images):
    mapping = {}
    for img in images:
        file_name = str(img["file_name"])
        image_id = img["id"]
        mapping[file_name] = image_id
        mapping[Path(file_name).name] = image_id
    return mapping

def load_cloud_results(path, file_to_image_id):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        elif "annotations" in data:
            data = data["annotations"]
        elif "detections" in data:
            data = data["detections"]
        else:
            raise ValueError(f"Unknown cloud json format keys: {list(data.keys())[:20]}")

    by_img = defaultdict(list)
    missing_cloud_ids = 0
    total_raw_dets = 0

    for item in data:
        # 格式 A：每张图一个 item，内部有 detections 列表
        if isinstance(item, dict) and "detections" in item and "image_id" in item:
            raw_image_id = item["image_id"]
            raw_key = str(raw_image_id)
            base_key = Path(raw_key).name

            if raw_image_id in file_to_image_id:
                image_id = file_to_image_id[raw_image_id]
            elif raw_key in file_to_image_id:
                image_id = file_to_image_id[raw_key]
            elif base_key in file_to_image_id:
                image_id = file_to_image_id[base_key]
            else:
                missing_cloud_ids += len(item.get("detections", []))
                continue

            for det in item.get("detections", []):
                total_raw_dets += 1

                if "bbox" not in det:
                    continue

                bbox = det["bbox"]
                if isinstance(bbox, dict):
                    x1 = float(bbox["x1"])
                    y1 = float(bbox["y1"])
                    x2 = float(bbox["x2"])
                    y2 = float(bbox["y2"])
                elif isinstance(bbox, list) and len(bbox) == 4:
                    # 如果是 list，这里按 xyxy 兜底处理
                    x1, y1, x2, y2 = [float(x) for x in bbox]
                else:
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                if "category_id" in det:
                    category_id = int(det["category_id"])
                elif "class_id" in det:
                    # 云端原始 class_id 是 0-based，COCO category_id 是 1-based
                    category_id = int(det["class_id"]) + 1
                else:
                    continue

                score = float(det.get("score", det.get("confidence", 1.0)))

                by_img[int(image_id)].append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                        "score": score,
                        "source": "cloud_full",
                    }
                )

            continue

        # 格式 B：扁平 COCO results，每条一个 bbox
        if isinstance(item, dict) and "image_id" in item and "bbox" in item:
            total_raw_dets += 1

            raw_image_id = item["image_id"]
            raw_key = str(raw_image_id)
            base_key = Path(raw_key).name

            if raw_image_id in file_to_image_id:
                image_id = file_to_image_id[raw_image_id]
            elif raw_key in file_to_image_id:
                image_id = file_to_image_id[raw_key]
            elif base_key in file_to_image_id:
                image_id = file_to_image_id[base_key]
            else:
                missing_cloud_ids += 1
                continue

            bbox = item["bbox"]

            if isinstance(bbox, dict):
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                coco_bbox = xyxy_to_xywh([x1, y1, x2, y2])
            elif isinstance(bbox, list) and len(bbox) == 4:
                # 扁平 COCO result 通常已经是 xywh
                coco_bbox = [float(x) for x in bbox]
            else:
                continue

            if "category_id" in item:
                category_id = int(item["category_id"])
            elif "class_id" in item:
                category_id = int(item["class_id"]) + 1
            else:
                continue

            score = float(item.get("score", item.get("confidence", 1.0)))

            by_img[int(image_id)].append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "score": score,
                    "source": "cloud_full",
                }
            )

    print("Loaded cloud raw detections:", total_raw_dets)
    print("Loaded cloud detections:", sum(len(v) for v in by_img.values()))
    print("Cloud images matched:", len(by_img))
    print("Cloud detections with unmatched image_id:", missing_cloud_ids)

    return by_img

def find_image_path(img_root, file_name):
    p = Path(img_root) / file_name
    if p.exists():
        return p

    p2 = Path(img_root) / Path(file_name).name
    if p2.exists():
        return p2

    matches = list(Path(img_root).rglob(Path(file_name).name))
    if matches:
        return matches[0]

    return None

def choose_cloud_candidate(low_det, roi_box, cloud_dets, min_cloud_score=0.05):
    yolo_box = xywh_to_xyxy(low_det["bbox"])
    cat_id = low_det["category_id"]

    candidates = []
    for cd in cloud_dets:
        if cd["category_id"] != cat_id:
            continue
        if cd["score"] < min_cloud_score:
            continue

        cbox = xywh_to_xyxy(cd["bbox"])
        c_iou = iou_xyxy(cbox, yolo_box)
        c_inside = center_inside(cbox, roi_box)
        c_inter = intersection_ratio(cbox, roi_box)

        if c_inside or c_inter >= 0.30 or c_iou >= 0.05:
            spatial = max(c_iou, c_inter)
            rank = cd["score"] + 0.25 * spatial
            candidates.append((rank, cd))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="datasets/SeaDronesSee/annotations/instances_val.json")
    parser.add_argument("--img-root", default="datasets/SeaDronesSee/images/val")
    parser.add_argument(
        "--yolo",
        default="/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt",
    )
    parser.add_argument("--cloud", default="experiments/cloud_full_1547_conf005.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--split-thr", type=float, default=0.70)
    parser.add_argument("--cloud-min-score", type=float, default=0.05)
    parser.add_argument("--out-prefix", default="experiments/full_cloud_guided_refinement_300")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        raise FileNotFoundError(f"GT not found: {args.gt}")
    if not Path(args.img_root).exists():
        raise FileNotFoundError(f"Image root not found: {args.img_root}")
    if not Path(args.cloud).exists():
        raise FileNotFoundError(f"Cloud full result not found: {args.cloud}")
    if not Path(args.yolo).exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo}")

    all_images = load_coco_gt(args.gt)
    file_to_image_id = build_file_to_image_id(all_images)
    images = all_images[: args.limit]
    cloud_by_img = load_cloud_results(args.cloud, file_to_image_id)
    model = YOLO(args.yolo)

    replace_results = []
    fuse_results = []

    stat = {
        "images": 0,
        "yolo_total": 0,
        "high_conf": 0,
        "low_conf": 0,
        "cloud_available": 0,
        "cloud_used_replace": 0,
        "missing_image": 0,
    }

    for idx, img in enumerate(images, 1):
        image_id = img["id"]
        file_name = img["file_name"]
        img_path = find_image_path(args.img_root, file_name)

        if img_path is None:
            stat["missing_image"] += 1
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            stat["missing_image"] += 1
            continue

        img_h, img_w = im.shape[:2]
        stat["images"] += 1

        pred = model(
            str(img_path),
            imgsz=640,
            conf=args.yolo_conf,
            iou=0.7,
            verbose=False,
        )[0]

        high_dets = []
        low_dets = []

        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, clss):
                cls_id = int(cls_id)
                if cls_id not in YOLO_TO_COCO_CAT:
                    continue

                cat_id = YOLO_TO_COCO_CAT[cls_id]
                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = max(0.0, min(x1, img_w))
                x2 = max(0.0, min(x2, img_w))
                y1 = max(0.0, min(y1, img_h))
                y2 = max(0.0, min(y2, img_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                det = {
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                    "score": float(score),
                    "source": "yolo",
                }

                stat["yolo_total"] += 1
                if score >= args.split_thr:
                    high_dets.append(det)
                    stat["high_conf"] += 1
                else:
                    low_dets.append(det)
                    stat["low_conf"] += 1

        cloud_dets = cloud_by_img.get(image_id, [])

        replace_img_dets = list(high_dets)
        fuse_img_dets = list(high_dets)

        for ld in low_dets:
            low_box = xywh_to_xyxy(ld["bbox"])
            roi_box = expand_roi(low_box, img_w, img_h, ld["category_id"])
            cd = choose_cloud_candidate(
                ld,
                roi_box,
                cloud_dets,
                min_cloud_score=args.cloud_min_score,
            )

            if cd is not None:
                stat["cloud_available"] += 1
                stat["cloud_used_replace"] += 1

                new_det = {
                    "image_id": image_id,
                    "category_id": cd["category_id"],
                    "bbox": cd["bbox"],
                    "score": max(float(cd["score"]), float(ld["score"]) + 1e-4),
                    "source": "cloud_full_guided",
                }

                replace_img_dets.append(new_det)
                fuse_img_dets.append(ld)
                fuse_img_dets.append(new_det)
            else:
                replace_img_dets.append(ld)
                fuse_img_dets.append(ld)

        replace_results.extend(nms_per_class(replace_img_dets, iou_thr=0.60))
        fuse_results.extend(nms_per_class(fuse_img_dets, iou_thr=0.60))

        if idx % 50 == 0:
            print(f"[{idx}/{len(images)}] processed")

    replace_out = args.out_prefix + "_replace.json"
    fuse_out = args.out_prefix + "_fuse.json"
    stat_out = args.out_prefix + "_stats.json"

    for d in replace_results:
        d.pop("source", None)
    for d in fuse_results:
        d.pop("source", None)

    with open(replace_out, "w") as f:
        json.dump(replace_results, f)

    with open(fuse_out, "w") as f:
        json.dump(fuse_results, f)

    with open(stat_out, "w") as f:
        json.dump(stat, f, indent=2)

    print("Saved:", replace_out, len(replace_results))
    print("Saved:", fuse_out, len(fuse_results))
    print("Saved:", stat_out)
    print(json.dumps(stat, indent=2))

if __name__ == "__main__":
    main()
