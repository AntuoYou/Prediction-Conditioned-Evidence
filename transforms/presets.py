import albumentations as A
import cv2
import torch

from transforms import v2 as T
from transforms.album_transform import RandomShortestSize
from transforms.albumentations_warpper import AlbumentationsWrapper
from transforms.crop import RandomSizeCrop as CropRandomSizeCrop  # 重命名以避免冲突
from transforms.mix_transform import CachedMixUp, CachedMosaic, MixUp, Mosaic

basic = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# train transform
hflip = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

lsj = T.Compose([
    T.ScaleJitter(target_size=(1024, 1024), antialias=True),
    T.RandomCrop(size=(1024, 1024), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

lsj_1536 = T.Compose([
    T.ScaleJitter(target_size=(1536, 1536), antialias=True),
    T.RandomCrop(size=(1536, 1536), pad_if_needed=True, fill=(123.0, 117.0, 104.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

multiscale = T.Compose([
    T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

detr = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            CropRandomSizeCrop(384, 600),  # 使用重命名后的导入
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

ssd = T.Compose([
    T.RandomPhotometricDistort(),
    T.RandomZoomOut(fill=[123.0, 117.0, 104.0]),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

ssdlite = T.Compose([
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

strong_album = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomChoice([
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            CropRandomSizeCrop(384, 600),  # 使用重命名后的导入
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),
    AlbumentationsWrapper(
        A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(0.1, 0.3),
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.RGBShift(
                            r_shift_limit=10,
                            g_shift_limit=10,
                            b_shift_limit=10,
                            p=1.0,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=1.0,
                ),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
                A.ChannelShuffle(p=0.1),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.1,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
        )
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

rtdetr_transform = T.Compose([
    T.RandomPhotometricDistort(p=0.8),
    T.RandomZoomOut(p=0.5, fill=0, side_range=(1.0, 4.0)),
    T.RandomIoUCrop(),
    T.RandomHorizontalFlip(p=0.5),
    T.Resize(size=[640, 640], antialias=True),
    T.ToImageTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

# some transform examples related to mosaic, mixup, cached_mosaic and cached_mixup
# you may want to add flip, crop, resize transforms into them for better performance
mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    Mosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

cached_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

mixup = T.Compose([
    T.RandomHorizontalFlip(),
    MixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

cached_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

mixup_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    MixUp(),
    Mosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

cached_mixup_mosaic = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMixUp(),
    CachedMosaic(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

mosaic_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    Mosaic(),
    MixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])

cached_mosaic_mixup = T.Compose([
    T.RandomHorizontalFlip(),
    CachedMosaic(),
    CachedMixUp(),
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==================== 推荐首选：Strong-DETR-Mosaic 增强方案 ====================
# 此方案融合了 DETR 的多尺度策略、Albumentations 强色彩增强和 Mosaic 数据混合
# 专为牺牲计算量换取 mAP 提升的场景设计，特别适合 SeaDronesSee 等小物体检测任务
strong_detr_mosaic = T.Compose([
    # 1. 基础几何增强：水平翻转
    T.RandomHorizontalFlip(p=0.5),

    # 2. DETR 风格多尺度与随机裁剪策略
    T.RandomChoice([
        # 选项A：直接多尺度缩放
        T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        # 选项B：先缩放，再随机裁剪，最后再缩放（增加多样性）
        T.Compose([
            T.RandomShortestSize([400, 500, 600], antialias=True),
            CropRandomSizeCrop(384, 600),  # 随机裁剪到 384-600 像素范围
            T.RandomShortestSize(min_size=scales, max_size=1333, antialias=True),
        ]),
    ]),

    # 3. Albumentations 强光度增强（在 Mosaic 前对单张图像进行）
    AlbumentationsWrapper(
        A.Compose(
            [
                # 亮度对比度扰动
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,  # 亮度扰动范围 ±25%
                    contrast_limit=0.25,  # 对比度扰动范围 ±25%
                    p=0.5
                ),
                # 色调饱和度明度扰动
                A.HueSaturationValue(
                    hue_shift_limit=15,  # 色调偏移限制
                    sat_shift_limit=30,  # 饱和度偏移限制
                    val_shift_limit=20,  # 明度偏移限制
                    p=0.5
                ),
                # 噪声增强（二选一）
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0)),  # 高斯噪声
                    A.ISONoise(color_shift=(0.01, 0.05),  # ISO 噪声
                               intensity=(0.1, 0.3)),
                ], p=0.2),
                # 图像压缩模拟（模拟 JPEG 压缩伪影）
                A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  # 边界框格式
                label_fields=["labels"],  # 标签字段
                min_visibility=0.1  # 过滤可见度低于10%的边界框
            ),
        )
    ),

    # 4. Mosaic 数据混合（四图拼接，极大增加上下文多样性）
    Mosaic(),  # 使用标准 Mosaic，如需加速可替换为 CachedMosaic()

    # 5. 张量转换与归一化
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    # 6. 边界框清洗（关键步骤：清除增强后无效的边界框）
    T.SanitizeBoundingBox(labels_getter=lambda x: x[-1]),
])