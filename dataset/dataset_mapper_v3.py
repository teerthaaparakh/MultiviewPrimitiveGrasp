# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
import cv2

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from utils.mapper_utils import apply_augmentations, get_sample_grasp_props_dict
from utils.other_configs import NORMALIZATION_CONST, OBJECT_DICTS
from detectron2.data import transforms as T

from dataset.dataset_function_util import visualize_mapper_dict, visualize_datapoint

def mapper(original_dataset_dict, draw=False, is_test=False):
    dataset_dict = copy.deepcopy(original_dataset_dict)
    depth = np.load(dataset_dict["depth_file_name"])
    image = cv2.imread(dataset_dict["file_name"])[:, :, ::-1]  #RGB
    seg = cv2.imread(dataset_dict["seg_file_name"], cv2.IMREAD_UNCHANGED)
    seg = seg.astype(np.uint8)

    assert len(seg.shape) == 2, f"segmentation shape is {seg.shape}"
    if len(depth.shape) == 2:
        depth = depth[..., None]
    if len(image.shape) == 2:
        image = image[..., None]

    h, w = image.shape[:2]
    annotations = dataset_dict.pop("annotations")  # number of objects in the image

    # apply augmentations
    # final dict has the same keys as in mapper dict
    # augs = T.AugmentationList(
    #     [
    #         # T.RandomFlip(prob=1.0),
    #         T.RandomRotation([-90.0, 90.0], 
    #                             expand=True, center=[[0.3, 0.3], [0.7, 0.7]], 
    #                             sample_style="range")
    #     ]
    # )
    
    augs = T.AugmentationList(
        [
            T.RandomFlip(prob=0.5),
            T.RandomApply(
                T.RandomRotation([-90.0, 90.0], 
                            expand=True, center=[[0.3, 0.3], [0.7, 0.7]], 
                            sample_style="range"),
                            prob=0.7)
        ]
    )

    trans_image, trans_depth, trans_annotations = apply_augmentations(
        augs, image, depth, seg, annotations
    )

    # select random grasp
    sampled_grasp_dict = get_sample_grasp_props_dict(trans_annotations)

    if draw:
        visualize_datapoint(
            original_dataset_dict, grasp_indices=sampled_grasp_dict["grasp_indices"]
        )

    image_input = np.concatenate(
        (trans_image / 255.0, trans_depth[None, ...]), axis=0
    )

    new_dict = {
        "image": torch.from_numpy(image_input).float(),
        "rgb": trans_image,
        "depth": trans_depth,
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=sampled_grasp_dict["category_ids"],
            gt_boxes=sampled_grasp_dict["bboxes"],
            gt_keypoints=sampled_grasp_dict["kpts"],
            gt_centerpoints=sampled_grasp_dict["center_kpts"],
            gt_orientations=sampled_grasp_dict["orientations"],  # NCx1
            gt_widths=sampled_grasp_dict["grasp_widths"],
            gt_scales=sampled_grasp_dict["scales"],
        ),
    }

    if draw and (not is_test):
        visualize_mapper_dict(new_dict, name="aug" + str(dataset_dict["image_id"]))

    return new_dict