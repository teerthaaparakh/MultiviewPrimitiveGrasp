from fvcore.transforms.transform import Transform, TransformList
import numpy as np
import torch
from typing import Tuple, List, Dict
import copy
from utils.other_configs import NORMALIZATION_CONST, OBJECT_DICTS
from detectron2.structures.boxes import Boxes

from utils.util import get_area, get_orientation_class
from detectron2.data import transforms as T

from dataset.dataset_function import generate_bbox_from_seg

def get_valid(kpt_coords: torch.Tensor, width: int, height: int):
    """
    kpt_coords: assuming of the form nxmx2
    where n is the number of grasps, m=4 (keypoints of the grasp)
    and 2 correspond to x and y coord
    0 <= x < width
    0 <= y < height

    Returned is a valid array of size (n,)
    """
    valid_x = (kpt_coords[:, :, 0] < width) & (0 < kpt_coords[:, :, 0])
    valid_y = (kpt_coords[:, :, 1] < height) & (0 < kpt_coords[:, :, 1])

    valid = valid_x & valid_y
    valid_grasps = torch.sum(valid, dim=1, dtype=torch.int16)
    result = (valid_grasps == 4) # all four keypoints must be valid
    return result


def transform_instance_grasps(
    transforms: TransformList,
    kpts: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    height: int,
    width: int,
    filter=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    all_instance_kpts: Nx4x3, N: number of objects
    centers: Nx3,
    scales: N
    """
    # all_instance_kpts = np.stack(all_instance_kpts)

    # import pdb;  pdb.set_trace()
    n, m, k = kpts.shape

    assert k == 3
    assert m == 4, f"expected m = 4 but found m = {m}"

    # raw_kpts = all_instance_kpts[:, :, :2] * torch.tensor([width, height])
    raw_kpts = kpts[:, :, :2] * NORMALIZATION_CONST
    raw_kpts = raw_kpts / scales.reshape((-1, 1, 1)) + centers[:, :2].unsqueeze(dim=1)
    flattened_kpts = raw_kpts.reshape((n * m, 2))

    transformed_coords = transforms.apply_coords(flattened_kpts)
    transformed_coords = transformed_coords.reshape((n, m, 2))

    if isinstance(transformed_coords, np.ndarray):
        transformed_coords = torch.from_numpy(transformed_coords)

    if filter:
        valid = get_valid(transformed_coords, width=width, height=height)
    else:
        valid = torch.ones(n, dtype=torch.bool)

    filtered_coords = transformed_coords[valid]
    transformed_centers = (filtered_coords[:, 0] + filtered_coords[:, 3]) / 2
    transformed_centers = torch.cat((transformed_centers, centers[valid][:, 2:3]), axis=1).unsqueeze(1)


    # transformed_centers = transform_instance_centerpoints(transforms, copy.deepcopy(centers))

    # obtaining offsets again
    transformed_offsets = filtered_coords - transformed_centers[:, :, :2]
    transformed_offsets = (
        transformed_offsets * scales[valid].reshape((-1, 1, 1)) / NORMALIZATION_CONST
    )

    new_keypoints = torch.cat(
        (transformed_offsets, kpts[valid][:, :, 2:]), axis=2
    )
    transformed_ori = torch.from_numpy(get_orientation_class(new_keypoints.numpy()))

    return valid, (new_keypoints, transformed_centers[:, 0, :], transformed_ori)


def apply_augmentations(augs: T.AugmentationList, image: np.ndarray, 
                        depth: np.ndarray, seg: np.ndarray, annotations):
    # ///////////////////////////////////////////////////////////////
    # augmentations
    # ///////////////////////////////////////////////////////////////
    h, w = image.shape[:2]
    auginput = T.AugInput(image, 
                          sem_seg=seg)
    transform = augs(auginput)
    transformed_image = auginput.image.transpose(2, 0, 1)
    transformed_depth = transform.apply_image(depth[:, :, 0])
    transformed_seg = auginput.sem_seg

    transformed_boxes = generate_bbox_from_seg(transformed_seg)
    
    # get raw keypoints from offsets, centers, scales and height and width
    num_instances = len(annotations)

    all_instance_dict = {
        "kpts": [],
        "center_kpts": [],
        "orientations": [],
        "grasp_widths": [],
        "scales": [],
        "grasp_indices": [],
        "category_ids": [],
        "bboxes": [],
        "object_indices": [],
    }
    
    for idx in range(num_instances):

        valid, (trans_kpts, trans_center_kpts, trans_ori) = transform_instance_grasps(
            transform,
            kpts=torch.from_numpy(annotations[idx]["kpts"]),
            centers=torch.from_numpy(annotations[idx]["center_kpts"]),
            scales=torch.from_numpy(annotations[idx]["scales"]),
            height=h,
            width=w,
            filter=True
        )
        # the segmentation masks start from 1, so adding one to the index
        instance_box = transformed_boxes[annotations[idx]["obj_index"] + 1]
        if get_area(instance_box) > 100:

            # TODO (MP): check whether indexing using `valid` is happening properly
            # or not
            grasp_widths = torch.from_numpy(annotations[idx]["grasp_widths"])[valid]
            scales = torch.from_numpy(annotations[idx]["scales"])[valid]
            all_instance_dict["kpts"].append(trans_kpts)
            all_instance_dict["center_kpts"].append(trans_center_kpts)
            all_instance_dict["orientations"].append(trans_ori)
            all_instance_dict["scales"].append(scales)
            all_instance_dict["grasp_widths"].append(grasp_widths)
            all_instance_dict["bboxes"].append(torch.tensor(instance_box))
            all_instance_dict["category_ids"].append(OBJECT_DICTS[annotations[idx]["obj_type"]])
            all_instance_dict["object_indices"].append(annotations[idx]["obj_index"])

    cat_list = all_instance_dict["category_ids"]
    all_instance_dict["category_ids"] = torch.tensor(cat_list, dtype=torch.long)
    bboxes = all_instance_dict["bboxes"]
    all_instance_dict["bboxes"] = Boxes(torch.stack(bboxes))

    return transformed_image, transformed_depth, all_instance_dict


def print_dict_info(all_object_dict):
    print("All instance tensor shapes:")
    print(f"    Keypoints:    {all_object_dict['kpts'].shape}")
    print(f"    Centerpoints: {all_object_dict['center_kpts'].shape}")
    print(f"    Orientations: {all_object_dict['orientations'].shape}")
    print(f"    Scales:       {all_object_dict['scales'].shape}")
    print(f"    Widths:       {all_object_dict['grasp_widths'].shape}")


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

def accumulate_grasp_properties(annotations_dict_list):
    num_instances = len(annotations_dict_list)

    mapper_dict = {
        "kpts": [],
        "center_kpts": [],
        "orientations": [],
        "grasp_widths": [],
        "scales": [],
        "grasp_indices": [],
        "category_ids": [],
        "bboxes": [],
        "object_indices": [],
    }

    for i in range(num_instances):
        object_dict = annotations_dict_list[i]
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
    return mapper_dict

