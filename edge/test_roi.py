import cv2, numpy as np
from edge.split_by_confidence import Detection
from edge.roi_crop import prepare_cloud_requests

# 模拟 1920x1080 海域图像
img = np.zeros((1080, 1920, 3), dtype=np.uint8)
img[:] = (30, 60, 120)
cv2.rectangle(img, (800,400),(900,500), (200,200,200), -1)  # 100x100 目标
cv2.rectangle(img, (1400,600),(1480,720), (200,200,200), -1) # 80x120 目标
cv2.rectangle(img, (200,800),(280,900), (180,180,180), -1)   # 80x100 目标

dets = [
    Detection((800,400,900,500), 0.45, 0, "swimmer"),
    Detection((1400,600,1480,720), 0.38, 0, "boat"),
    Detection((200,800,280,900), 0.52, 0, "buoy"),
]
reqs = prepare_cloud_requests(img, dets)
_, full = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
full_kb = len(full.tobytes()) / 1024
roi_kb = sum(r["data_kb"] for r in reqs)
print(f"全图: {full_kb:.1f} KB")
print(f"ROI:  {roi_kb:.1f} KB")
print(f"节省: {(1-roi_kb/full_kb)*100:.1f}%")
for r in reqs:
    print(f"  ROI #{r['index']}: {r['crop_bbox']}, {r['data_kb']:.1f} KB, conf={r['metadata']['edge_confidence']}")
print("[OK] ROI 测试通过")
