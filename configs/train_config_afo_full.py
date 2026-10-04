import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# AFO full-image training config
num_epochs = 36
batch_size = 1
num_workers = 4
pin_memory = True
print_freq = 50
starting_epoch = 0
max_norm = 0.1

output_dir = "checkpoints/afo_fullimage"
find_unused_parameters = False

# AFO dataset path
coco_path = '/app/datasets/afo/coco_format'

train_transform = presets.strong_detr_mosaic

train_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/train",
    ann_file=f"{coco_path}/annotations/instances_train.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/val",
    ann_file=f"{coco_path}/annotations/instances_val.json",
    transforms=None,
)

# AFO model config (6 classes)
model_path = "configs/salience_detr/salience_detr_swin_l_800_1333_afo.py"

# Finetune from SeaDronesSee checkpoint
resume_from_checkpoint = "checkpoints/salience_detr_swin_l_800_1333/train/2026-02-17-11_33_16/best_ap.pth"

learning_rate = 2e-5
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[24, 30], gamma=0.5)

param_dicts = param_dict.finetune_backbone_and_linear_projection(
    lr=learning_rate
)
