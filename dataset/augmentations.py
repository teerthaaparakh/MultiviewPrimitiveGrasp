import sys, os
sys.path.append(os.environ["KGN_DIR"])
from detectron2.projects.point_rend import ColorAugSSDTransform
import detectron2.data.transforms as T
import albumentations as A

def build_grasp_augmentations(cfg):
    # augs = []
    additional_augs = None
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            )
        )
    augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomApply(T.RandomRotation([-90.0, 90.0], 
                        expand=True, center=[[0.3, 0.3], [0.7, 0.7]], 
                        sample_style="range"), prob=0.3))

    albumention_lst = [
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=0.5),
        A.FancyPCA (alpha=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.5, interpolation=2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=1.0, p=0.50),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.MotionBlur(blur_limit=9, allow_shifted=True, always_apply=False, p=0.5),
        A.GaussNoise(var_limit=(10.0, 80.0), mean=0, per_channel=True, p=0.5),
        A.RandomBrightness(limit=0.2, always_apply=False, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1.0),
        A.RandomContrast(limit=0.2, p=1.0),
    ]

    t1 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
    t2 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
    t3 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
    t4 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
    t5 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)

    additional_augs = A.Compose([A.OneOf([t1, t2, t3, t4, t5], p=0.5)])
    return augs, additional_augs
