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
    transformed_coords = transforms.apply_coords(all_instance_centerpoints[:, 0, :2])
    # import pdb; pdb.set_trace()
    if isinstance(transformed_coords, np.ndarray):
        transformed_coords = torch.from_numpy(transformed_coords)
    keypoints = torch.cat(
        (transformed_coords, all_instance_centerpoints[:, 0, 2:3]), axis=1
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
    raw_kpts = raw_kpts / scales.reshape((-1, 1, 1)) + centers[:, :, :2]
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

    return new_keypoints, transformed_centers


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(original_dataset_dict, scaled_rgb=True, draw=False):
    dataset_dict = copy.deepcopy(original_dataset_dict)

    depth = np.load(dataset_dict["depth_file_name"])
    image = cv2.imread(dataset_dict["file_name"])[:, :, ::-1]
    if len(depth.shape) == 2:
        depth = depth[..., None]
    if len(image.shape) == 2:
        image = image[..., None]

    h, w = image.shape[:2]
    print("Image shape:", h, w)
    # logging.info("Number of channels in image:", image.shape[2])

    # num_instances == number of objects
    annotations = dataset_dict.pop("annotations")  # number of objects in the image

    num_instances = len(annotations)
    all_instance_keypoints = []
    all_instance_widths = []
    all_instance_centerpoints = []
    all_instance_orientations = []
    all_instance_scales = []
    all_instance_grasp_indices = []
    category_ids = []
    obj_box = []
    # print("num_instances", num_instances)
    for i in range(num_instances):
        object_dict = annotations[i]
        # it can have multiple keypoint sets
        keypoints = object_dict["keypoints"]
        width = object_dict["grasp_width"]  # 5 length vector
        ori = object_dict["ori_clss"]  # 5 length vector
        center = object_dict["centers"]  # 5x3
        center = center[:, None, :]  # 5x1x2
        scales = object_dict["scales"]

        if len(keypoints) > 0:
            sampled_idx = np.random.randint(0, len(keypoints))
            all_instance_grasp_indices.append(sampled_idx)

            all_instance_keypoints.append(torch.tensor(keypoints)[sampled_idx])
            all_instance_orientations.append(ori[sampled_idx])
            all_instance_widths.append(width[sampled_idx])
            all_instance_centerpoints.append(torch.tensor(center)[sampled_idx])
            all_instance_scales.append(scales[sampled_idx])

            category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])
            obj_box.append(object_dict["bbox"])

    if draw:
        visualize_datapoint(
            original_dataset_dict, grasp_indices=all_instance_grasp_indices
        )

    all_instance_keypoints = torch.stack(all_instance_keypoints)
    all_instance_centerpoints = torch.stack(all_instance_centerpoints)
    all_instance_orientations = torch.tensor(all_instance_orientations)
    all_instance_scales = torch.tensor(all_instance_scales)
    all_instance_widths = torch.tensor(all_instance_widths)

    # print("All instance tensor shapes:")
    # print(f"    Keypoints:    {all_instance_keypoints.shape}")
    # print(f"    Centerpoints: {all_instance_centerpoints.shape}")
    # # print(f"    Centerpoints (Value): {all_instance_centerpoints}")
    # print(f"    Orientations: {all_instance_orientations.shape}")
    # print(f"    Scales:       {all_instance_scales.shape}")
    # print(f"    Widths:       {all_instance_widths.shape}")
    # ///////////////////////////////////////////////////////////////
    # augmentations
    # ///////////////////////////////////////////////////////////////
    augs = T.AugmentationList(
        [
            # T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
            # T.RandomFlip(prob=1.0),
            # T.RandomCrop("absolute", (200, 200)),
            #     T.RandomApply(
            T.RandomRotation(
                [-90.0, -89.0],
                expand=True,
                # center=[[0.3, 0.3], [0.7, 0.7]],
                center=[[0.49, 0.49], [0.51, 0.51]],
                sample_style="range",
            ),
            #     prob=0.3,
            # )
        ]
    )

    auginput = T.AugInput(image, boxes=np.array(obj_box))
    transform = augs(auginput)
    transformed_image = auginput.image.transpose(2, 0, 1)
    transformed_boxes = auginput.boxes
    # TODO Check these functions

    # get raw keypoints from offsets, centers, scales and height and width
    transformed_kpts, transformed_center_kpts = transform_img_kpts(
        transforms=transform,
        all_instance_kpts=all_instance_keypoints,
        centers=all_instance_centerpoints,
        scales=all_instance_scales,
        height=h,
        width=w,
    )

    transformed_ori = get_orientation_class(transformed_kpts.numpy())
    transformed_ori = torch.from_numpy(transformed_ori)

    # print("Transformed informations for an object:")
    # print(f"    Keypoints:")
    # print(f"        new shape: {transformed_kpts.shape}")
    # print(f"        new: {transformed_kpts[0]}")
    # print(f"        old: {all_instance_keypoints[0]}")

    # print(f"    Centerpoints:")
    # print(f"        new shape: {transformed_center_kpts.shape}")
    # print(f"        new: {transformed_center_kpts[0]}")
    # print(f"        old: {all_instance_centerpoints[0]}")

    # print(f"    Orientations:")
    # print(f"        new shape: {transformed_ori.shape}")
    # print(f"        new: {transformed_ori[0]}")
    # print(f"        old: {all_instance_orientations[0]}")

    # TODO Check for float depths
    transformed_depth = transform.apply_image(depth[:, :, 0])
    # ///////////////////////////////////////////////////////////////

    image_input = np.concatenate(
        (transformed_image / 255.0, transformed_depth[None, ...]), axis=0
    )

    new_dict = {
        "input": torch.from_numpy(image_input).float(),
        "rgb": transformed_image,
        "depth": transformed_depth,
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_boxes=Boxes(np.array(transformed_boxes)),
            gt_keypoints=transformed_kpts,
            gt_centerpoints=transformed_center_kpts,
            gt_orientations=transformed_ori,  # NCx1
            gt_widths=all_instance_widths,
            gt_scales=all_instance_scales,
        ),
    }

    if draw:
        visualize_mapper_dict(new_dict, name="flip" + str(dataset_dict["image_id"]))

    return new_dict


# from dataset_function import load_dataset
# from utils.path_util import get_data_dir, get_pickled_aug_data_dir, get_pickled_normal_data_dir
# import pickle
# import os
# import glob

# def load_dataset_wrapper():
#     data_save_dir = get_pickled_normal_data_dir()
#     os.makedirs(data_save_dir, exist_ok=True)
#     data_file_path = os.path.join(data_save_dir, "data.pkl")
#     if os.path.exists(data_file_path):
#         with open(data_file_path, 'rb') as f:
#             dataset_dicts = pickle.load(f)
#     else:
#         data_path = get_data_dir()
#         dataset_dicts = load_dataset(data_path)
#         with open(data_file_path, 'wb') as f:
#             pickle.dump(dataset_dicts, f)

#     return dataset_dicts


# def generate_pickled_aug_data(dataset_dicts):
#     folder_name = get_pickled_aug_data_dir()
#     os.makedirs(folder_name, exist_ok=True)
#     for i, ddict in enumerate(dataset_dicts):
#         augmented_dict = generate_pickled_data(ddict)
#         file_path = os.path.join(folder_name, str(i)+".pkl" )
#         with open(file_path, "wb") as f:
#             pickle.dump(augmented_dict, f)


# def load_pickled_aug_dataset():
#     files_path = os.path.join(get_pickled_aug_data_dir(), "*.pkl")
#     files_path = glob.glob(files_path)
#     data_dicts = []
#     for file in files_path:
#         with open(file, "rb") as f:
#             data_dicts.append(pickle.load(f))
