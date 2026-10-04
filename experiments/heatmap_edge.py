import os, io, argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

YOLO_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/pce_results/visualizations"

def build_uncertainty_map(img, dets, sigma_scale=0.1):
    h, w = img.shape[:2]
    conf_map = np.zeros((h, w), dtype=np.float32)
    for d in dets:
        x1, y1, x2, y2 = d[:4]
        score = d[4]
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        bw, bh = x2-x1, y2-y1
        sigma_x = max(bw * sigma_scale, 8)
        sigma_y = max(bh * sigma_scale, 8)
        gx = np.arange(w) - cx
        gy = np.arange(h) - cy
        gx, gy = np.meshgrid(gx, gy)
        gauss = np.exp(-(gx**2/(2*sigma_x**2) + gy**2/(2*sigma_y**2)))
        conf_map = np.maximum(conf_map, gauss * score)
    return 1.0 - conf_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading YOLO...")
    model = YOLO(YOLO_PATH)

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])[:100]
    selected = []
    for fname in img_files:
        r = model(os.path.join(IMG_DIR, fname), conf=0.5, verbose=False)[0]
        n = len(r.boxes) if r.boxes is not None else 0
        if n >= 3:
            selected.append(fname)
        if len(selected) >= args.num_images:
            break

    for i, fname in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {fname}")
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        r = model(img_path, conf=0.01, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            for j in range(len(xyxy)):
                dets.append([xyxy[j][0], xyxy[j][1], xyxy[j][2], xyxy[j][3], confs[j]])

        uncertainty = build_uncertainty_map(img, dets)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        axes[0].imshow(img)
        for d in dets:
            if d[4] < 0.5: continue
            x1, y1, x2, y2 = d[:4]
            rect = Rectangle((x1,y1), x2-x1, y2-y1, lw=2, ec='lime', fc='none')
            axes[0].add_patch(rect)
        axes[0].set_title('Edge Detections', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(uncertainty, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Low-Confidence Heatmap', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Uncertainty (red=low confidence)', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(img, alpha=0.3)
        axes[2].imshow(uncertainty, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        for d in dets:
            if d[4] < 0.5: continue
            x1, y1, x2, y2 = d[:4]
            rect = Rectangle((x1,y1), x2-x1, y2-y1, lw=2, ec='white', fc='none')
            axes[2].add_patch(rect)
        axes[2].set_title('Heatmap Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"heatmap_{os.path.splitext(fname)[0]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\nDone! {args.output_dir}")

if __name__ == "__main__":
    main()
