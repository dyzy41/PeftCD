import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_best_model_checkpoint(ckpt_dir):
    best_ckpt = max(
        (os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and 'best' in f),
        key=lambda p: float(
            os.path.splitext(os.path.basename(p))[0]       # 去掉 ".ckpt"
              .split('-')[-1]                               # 取最后一段 "val_iou=0.7491"
              .split('=')[-1]                               # 取"="后面的数字部分
        )
    )
    print(f"Best checkpoint: {best_ckpt}")
    return best_ckpt


import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def define_transforms(crop_size, resize_size):
    train_transform_dict = {}
    test_transform_dict = {}

    # 1. 定义公共的 base list
    base_list = [
        # A.AdvancedBlur(p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        # A.Illumination(intensity_range=(0.01, 0.1), p=0.5),
        # A.Sharpen(alpha=(0.05, 0.2), p=0.5),
        A.GridDistortion(distort_limit=(-0.15, 0.15), p=0.5),
        A.D4(),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ]

    # 2. base pipeline
    base_transform = A.Compose(
        base_list,
        additional_targets={"image1": "image", "mask": "mask"},
        strict=True
    )
    train_transform_dict['base'] = base_transform

    # 3. crop pipeline：在 Normalize 和 ToTensor 之前插入 RandomCrop
    crop_transform = copy.deepcopy(base_transform)
    crop_transform.transforms.insert(-2, A.Compose(
    [
        A.RandomCrop(height=crop_size, width=crop_size)
    ],
        additional_targets={"image1": "image", "mask": "mask"},
        strict=True
        )
    )
    train_transform_dict['crop'] = crop_transform

    # 4. resize pipeline：最前面加上 Resize
    resize_transform = copy.deepcopy(base_transform)
    resize_transform.transforms.insert(0, A.Compose(
    [
        A.Resize(height=resize_size,
                 width=resize_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST, 
                 p=1),
    ],
        additional_targets={"image1": "image", "mask": "mask"},
        strict=True
        )
    )
    train_transform_dict['resize'] = resize_transform

    # 5. 测试集：只 normalize + to tensor
    test_list = [
        A.Normalize(),
        ToTensorV2()
    ]
    test_transform = A.Compose(
        test_list,
        additional_targets={"image1": "image", "mask": "mask"},
        strict=True
    )
    test_transform_dict['base'] = test_transform

    # 6. 测试集 resize
    test_resize = copy.deepcopy(test_transform)
    test_resize.transforms.insert(0, A.Compose(
    [
        A.Resize(height=resize_size,
                 width=resize_size,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST, 
                 p=1),
    ],
        additional_targets={"image1": "image", "mask": "mask"},
        strict=True
        )
    )
    test_transform_dict['resize'] = test_resize

    return train_transform_dict, test_transform_dict
