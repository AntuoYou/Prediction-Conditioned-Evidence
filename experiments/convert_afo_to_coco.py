"""
Convert AFO YOLO format to COCO JSON format.
Images are spread across PART_1/2/3, labels all in PART_1/6categories.
Usage: python convert_afo_to_coco.py
"""
import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict

SRC = "/app/datasets/afo/afo_raw"
DST = "/app/datasets/afo/coco_format"
LABEL_DIR = Path(SRC) / "PART_1" / "PART_1" / "6categories"
IMG_DIRS = [
    Path(SRC) / "PART_1" / "PART_1" / "images",
    Path(SRC) / "PART_2" / "PART_2" / "images",
    Path(SRC) / "PART_3" / "PART_3" / "images",
]

CLASSES = ["human", "wind/sup-board", "kayak", "boat", "bouy", "sailboat"]
TRAIN_PCT, VAL_PCT, TEST_PCT = 70, 15, 15
SEED = 42

def find_image(img_stem):
    """Find image file across all PART directories."""
    for img_dir in IMG_DIRS:
        for ext in [".jpg", ".png", ".JPG", ".PNG"]:
            p = img_dir / (img_stem + ext)
            if p.exists():
                return p
    return None

def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    bw = w * img_w
    bh = h * img_h
    return [max(0,x1), max(0,y1), min(bw, img_w-max(0,x1)), min(bh, img_h-max(0,y1))]

def main():
    random.seed(SEED)
    print("Converting AFO YOLO → COCO...")

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    print(f"  Found {len(label_files)} label files")

    # Build image lookup: stem -> full path
    print("  Building image index...")
    img_lookup = {}
    for img_dir in IMG_DIRS:
        for f in img_dir.iterdir():
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                stem = f.stem
                if stem not in img_lookup:
                    img_lookup[stem] = f
    print(f"  Found {len(img_lookup)} total images")

    images_info = []
    img_id = 1
    not_found = 0

    for lf in label_files:
        stem = lf.stem
        img_path = img_lookup.get(stem)
        if img_path is None:
            not_found += 1
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        labels = []
        with open(lf) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                labels.append((cls_id, cx, cy, w, h))

        images_info.append({
            "id": img_id, "file_name": img_path.name,
            "width": img_w, "height": img_h,
            "labels": labels, "src": str(img_path),
        })
        img_id += 1

    print(f"  Matched {len(images_info)} images (not found: {not_found})")
    print(f"  Total annotations: {sum(len(i['labels']) for i in images_info)}")

    # Class distribution
    counts = defaultdict(int)
    for img in images_info:
        for l in img["labels"]:
            counts[l[0]] += 1
    print("  Class distribution:")
    for i, c in enumerate(CLASSES):
        print(f"    {c} (id={i}): {counts[i]}")

    # Split by video prefix to avoid data leakage
    print("  Splitting by video prefix...")
    prefix_groups = defaultdict(list)
    for img in images_info:
        # Extract prefix (everything before first number)
        name = img["file_name"]
        prefix = ""
        for ch in name:
            if ch.isdigit():
                break
            prefix += ch
        prefix = prefix.rstrip('_')
        prefix_groups[prefix].append(img)

    print(f"  Found {len(prefix_groups)} video prefixes")

    # Shuffle prefixes and split
    prefixes = list(prefix_groups.keys())
    random.shuffle(prefixes)

    train_imgs, val_imgs, test_imgs = [], [], []
    total = len(images_info)
    train_target = int(total * TRAIN_PCT / 100)
    val_target = int(total * VAL_PCT / 100)

    for p in prefixes:
        group = prefix_groups[p]
        if len(train_imgs) < train_target:
            train_imgs.extend(group)
        elif len(val_imgs) < val_target:
            val_imgs.extend(group)
        else:
            test_imgs.extend(group)

    print(f"  Split: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Create output dirs
    for split in ["train", "val", "test"]:
        os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/annotations", exist_ok=True)

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        print(f"  Processing {split_name}...")
        split_img_dir = Path(DST) / "images" / split_name

        # Copy images
        for img in split_imgs:
            dst_path = split_img_dir / img["file_name"]
            if not dst_path.exists():
                shutil.copy2(img["src"], dst_path)

        # Build COCO JSON
        coco = {
            "info": {"description": f"AFO {split_name}", "version": "1.0"},
            "licenses": [{"id": 1, "name": "CC BY-NC-SA 3.0"}],
            "images": [], "annotations": [], "categories": []
        }
        for i, name in enumerate(CLASSES):
            coco["categories"].append({"id": i, "name": name, "supercategory": "object"})

        ann_id = 1
        for img in split_imgs:
            coco["images"].append({
                "id": img["id"], "file_name": img["file_name"],
                "width": img["width"], "height": img["height"]
            })
            for cls_id, cx, cy, w, h in img["labels"]:
                bbox = yolo_to_coco_bbox(cx, cy, w, h, img["width"], img["height"])
                area = bbox[2] * bbox[3]
                coco["annotations"].append({
                    "id": ann_id, "image_id": img["id"],
                    "category_id": cls_id, "bbox": bbox,
                    "area": area, "iscrowd": 0,
                })
                ann_id += 1

        ann_path = f"{DST}/annotations/instances_{split_name}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)
        print(f"    {split_name}: {len(coco['images'])} imgs, {len(coco['annotations'])} anns")

    # Save dataset yaml for YOLO training
    yaml_path = f"{DST}/afo.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DST}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")
    print(f"\n  YOLO dataset YAML: {yaml_path}")
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
