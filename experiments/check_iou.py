import json, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 读你的结果 JSON
with open('expB_expansion_ablation.json') as f:
    data = json.load(f)

# 假设你保存了每个 expand 的 COCO results，如果没有需要改脚本保存
# 这里仅作示意：用 COCOeval 输出 AP@各 IoU
gt = COCO('/app/data/seadronessee/compressed_version/annotations/instances_val2017.json')
print("expand=0.1 AP75:", data['metrics']['0.1'].get('AP75', '未保存'))
print("expand=0.2 AP75:", data['metrics']['0.2'].get('AP75', '未保存'))
