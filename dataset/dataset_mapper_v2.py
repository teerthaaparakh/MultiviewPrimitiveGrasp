# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
import cv2

from detectron2.structures.keypoints import Keypoints
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from fvcore.transforms.transform import Transform, TransformList

from detectron2.data import transforms as T
from typing import List
from utils.other_configs import NORMALIZATION_CONST, OBJECT_DICTS
from utils.util import get_orientation_class

from dataset.dataset_function_util import visualize_mapper_dict, visualize_datapoint


def transform_img_centerpoints(
    transforms: TransformList, all_instance_centerpoints: torch.Tensor
) -> torch.Tensor:
    """
    all_instance_centerpoints: Nx1x3
    """
    # transformed_coords = transforms.apply_coords(all_instance_centerpoints[:, 0, :2])
    transformed_coords = transforms.apply_coords(all_instance_centerpoints[:, :2])
    if isinstance(transformed_coords, np.ndarray):
        transformed_coords = torch.from_numpy(transformed_coords)
    keypoints = torch.cat(
        (transformed_coords, all_instance_centerpoints[:, 2:3]),
        axis=1
        # (transformed_coords, all_instance_centerpoints[:, 0, 2:3]), axis=1
    )
    # returned keypoints has shape Nx1x3
    return keypoints.unsqueeze(1)


def transform_img_kpts(
    transforms: TransformList,
    all_instance_kpts: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    all_instance_kpts: Nx4x3, N: number of objects
    centers: Nx3,
    scales: N
    """
    # all_instance_kpts = np.stack(all_instance_kpts)

    # import pdb;  pdb.set_trace()
    n, m, k = all_instance_kpts.shape

    assert k == 3
    assert m == 4, f"expected m = 4 but found m = {m}"

    # raw_kpts = all_instance_kpts[:, :, :2] * torch.tensor([width, height])
    raw_kpts = all_instance_kpts[:, :, :2] * NORMALIZATION_CONST
    raw_kpts = raw_kpts / scales.reshape((-1, 1, 1)) + centers[:, :2].unsqueeze(dim=1)
    flattened_kpts = raw_kpts.reshape((n * m, 2))

    transformed_coords = transforms.apply_coords(flattened_kpts)
    transformed_coords = transformed_coords.reshape((n, m, 2))

    if isinstance(transformed_coords, np.ndarray):
        transformed_coords = torch.from_numpy(transformed_coords)

    transformed_centers = transform_img_centerpoints(transforms, copy.deepcopy(centers))

    # obtaining offsets again
    transformed_offsets = transformed_coords - transformed_centers[:, :, :2]
    # transformed_offsets = transformed_offsets * scales.reshape((-1, 1, 1)) / torch.tensor([width, height])
    transformed_offsets = (
        transformed_offsets * scales.reshape((-1, 1, 1)) / NORMALIZATION_CONST
    )

    new_keypoints = torch.cat(
        (transformed_offsets, all_instance_kpts[:, :, 2:]), axis=2
    )

    return new_keypoints, transformed_centers[:, 0, :]


def get_sample_grasp_props_dict(all_grasps_dict):
    new_dict = {k: [] for k in all_grasps_dict.keys()}

    num_instances = len(all_grasps_dict["kpts"])
    for i in range(num_instances):
        num_grasps = len(all_grasps_dict["kpts"][i])
        if num_grasps > 0:
            sampled_idx = np.random.randint(0, num_grasps)
            new_dict["object_indices"].append(i)
            new_dict["grasp_indices"].append(sampled_idx)
            new_dict["kpts"].append(all_grasps_dict["kpts"][i][sampled_idx])
            new_dict["center_kpts"].append(
                all_grasps_dict["center_kpts"][i][sampled_idx]
            )
            new_dict["orientations"].append(
                all_grasps_dict["orientations"][i][sampled_idx]
            )
            new_dict["scales"].append(all_grasps_dict["scales"][i][sampled_idx])
            new_dict["grasp_widths"].append(
                all_grasps_dict["grasp_widths"][i][sampled_idx]
            )
            new_dict["category_ids"].append(all_grasps_dict["category_ids"][i])
            new_dict["bboxes"].append(all_grasps_dict["bboxes"].tensor[i])

    # converting to torch arrays
    new_dict["kpts"] = torch.stack(new_dict["kpts"])
    new_dict["center_kpts"] = torch.stack(new_dict["center_kpts"])
    new_dict["orientations"] = torch.tensor(new_dict["orientations"])
    new_dict["scales"] = torch.tensor(new_dict["scales"])
    new_dict["grasp_widths"] = torch.tensor(new_dict["grasp_widths"])
    new_dict["category_ids"] = torch.tensor(new_dict["category_ids"], dtype=torch.long)
    new_dict["bboxes"] = Boxes(torch.stack(new_dict["bboxes"]))

    return new_dict


def print_dict_info(all_object_dict):
    print("All instance tensor shapes:")
    print(f"    Keypoints:    {all_object_dict['kpts'].shape}")
    print(f"    Centerpoints: {all_object_dict['center_kpts'].shape}")
    print(f"    Orientations: {all_object_dict['orientations'].shape}")
    print(f"    Scales:       {all_object_dict['scales'].shape}")
    print(f"    Widths:       {all_object_dict['grasp_widths'].shape}")


def apply_augmentations(image: np.ndarray, depth: np.ndarray, all_object_grasp_dict):
    # ///////////////////////////////////////////////////////////////
    # augmentations
    # ///////////////////////////////////////////////////////////////
    h, w = image.shape[:2]
    augs = T.AugmentationList(
        [
            T.RandomFlip(prob=1.0),
        ]
    )

    auginput = T.AugInput(image, boxes=all_object_grasp_dict["bboxes"].tensor.numpy())
    transform = augs(auginput)
    transformed_image = auginput.image.transpose(2, 0, 1)
    transformed_boxes = auginput.boxes
    # TODO Check these functions

    # get raw keypoints from offsets, centers, scales and height and width
    transformed_kpts, transformed_center_kpts = transform_img_kpts(
        transforms=transform,
        all_instance_kpts=all_object_grasp_dict["kpts"],
        centers=all_object_grasp_dict["center_kpts"],
        scales=all_object_grasp_dict["scales"],
        height=h,
        width=w,
    )
    if len(transformed_boxes) != len(transformed_kpts):
        print(transformed_boxes.shape, transformed_kpts.shape, all_object_grasp_dict["kpts"].shape)
        assert False

    transformed_ori = get_orientation_class(transformed_kpts.numpy())
    transformed_ori = torch.from_numpy(transformed_ori)

    # TODO Check for float depths
    transformed_depth = transform.apply_image(depth[:, :, 0])
    augmented_grasps_dict = {
        "kpts": transformed_kpts,
        "center_kpts": transformed_center_kpts,
        "orientations": transformed_ori,
        "scales": all_object_grasp_dict["scales"],
        "grasp_widths": all_object_grasp_dict["grasp_widths"],
        "bboxes": Boxes(torch.from_numpy(transformed_boxes)),
        "category_ids": all_object_grasp_dict["category_ids"]
    }


    return transformed_image, transformed_depth, augmented_grasps_dict


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(original_dataset_dict, draw=False, is_test=False):
    dataset_dict = copy.deepcopy(original_dataset_dict)

    depth = np.load(dataset_dict["depth_file_name"])
    image = cv2.imread(dataset_dict["file_name"])[:, :, ::-1]
    
    if len(depth.shape) == 2:
        depth = depth[..., None]
    if len(image.shape) == 2:
        image = image[..., None]

    h, w = image.shape[:2]
    
    # logging.info("Number of channels in image:", image.shape[2])

    # num_instances == number of objects
    annotations = dataset_dict.pop("annotations")  # number of objects in the image
    num_instances = len(annotations)

    data_dict_keys = [
        "kpts",
        "center_kpts",
        "orientations",
        "grasp_widths",
        "scales",
        "grasp_indices",
        "category_ids",
        "bboxes",
        "object_indices",
    ]
    mapper_dict = {k: [] for k in data_dict_keys}

    for i in range(num_instances):
        object_dict = annotations[i]
        mapper_dict["kpts"].append(torch.from_numpy(object_dict["kpts"]))
        mapper_dict["center_kpts"].append(torch.from_numpy(object_dict["center_kpts"]))
        mapper_dict["orientations"].append(
            torch.from_numpy(object_dict["orientations"])
        )
        mapper_dict["grasp_widths"].append(
            torch.from_numpy(object_dict["grasp_widths"])
        )
        mapper_dict["scales"].append(torch.from_numpy(object_dict["scales"]))
        mapper_dict["category_ids"].append(OBJECT_DICTS[object_dict["obj_type"]])
        mapper_dict["bboxes"].append(torch.from_numpy(object_dict["bbox"]))
        mapper_dict["object_indices"].append(i)

    mapper_dict["category_ids"] = torch.tensor(
        mapper_dict["category_ids"], dtype=torch.long
    )
    
    mapper_dict["bboxes"] = Boxes(torch.stack(mapper_dict["bboxes"]))

    
    if is_test:
        
        image_input = np.concatenate(
            (image / 255.0, depth), axis=2
        ).transpose(2, 0, 1)
        final_dict = copy.deepcopy(mapper_dict)
        transformed_image = image.transpose(2, 0, 1)
        transformed_depth = depth

    else:
        train_dict = get_sample_grasp_props_dict(mapper_dict)
        # print("Before Augmentations")
        # print_dict_info(train_dict)
        if draw:
            visualize_datapoint(
                original_dataset_dict, grasp_indices=train_dict["grasp_indices"]
            )

        transformed_image, transformed_depth, final_dict = apply_augmentations(
            image, depth, all_object_grasp_dict=train_dict
        )

        image_input = np.concatenate(
            (transformed_image / 255.0, transformed_depth[None , ...]), axis=0
        )

    new_dict = {
        "image": torch.from_numpy(image_input).float(),
        "rgb": transformed_image,
        "depth": transformed_depth,
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=final_dict["category_ids"],
            gt_boxes=final_dict["bboxes"],
            gt_keypoints=final_dict["kpts"],
            gt_centerpoints=final_dict["center_kpts"],
            gt_orientations=final_dict["orientations"],  # NCx1
            gt_widths=final_dict["grasp_widths"],
            gt_scales=final_dict["scales"],
        ),
    }

    if draw and (not is_test):
        visualize_mapper_dict(new_dict, name="aug" + str(dataset_dict["image_id"]))

    return new_dict
