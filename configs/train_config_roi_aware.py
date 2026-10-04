import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from datasets.roi_aware_coco import ROIAwareCocoDetection
from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# ========== ROI-aware Fine-tuning 配置 ==========
# 微调而非从头训练，epoch 少、学习率小
num_epochs = 10
batch_size = 1
num_workers = 8
pin_memory = True
print_freq = 50
starting_epoch = 0
max_norm = 0.1

output_dir = None  # auto: checkpoints/salience_detr_swin_l_800_1333/train/{timestamp}
find_unused_parameters = False

# ========== 数据集 ==========
coco_path = '/app/data/seadronessee/compressed_version'

# 用 detr preset（不用 strong_detr_mosaic，避免 mosaic + ROI 分布冲突）
train_transform = presets.detr

# ROI 比例配置: 40%全图 + 42%ROI50 + 9%ROI40 + 9%ROI60
roi_config = {
    "full": 0.40,
    "roi50": 0.42,
    "roi40": 0.09,
    "roi60": 0.09,
}

# 训练集: ROI-aware Dataset
train_dataset = ROIAwareCocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
    roi_config=roi_config,
)

# 验证集: 原始全图评估（监控全图性能是否退化）
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,
)

# ========== 模型 ==========
model_path = "configs/salience_detr/salience_detr_swin_l_800_1333.py"

# 从当前最佳 checkpoint 微调
resume_from_checkpoint = "checkpoints/salience_detr_swin_l_800_1333/train/2026-02-17-11_33_16/best_ap.pth"

# ========== 优化器 ==========
learning_rate = 1e-5  # 微调用较小学习率
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[7], gamma=0.5)

param_dicts = param_dict.finetune_backbone_and_linear_projection(
    lr=learning_rate
)
