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
from utils.other_configs import OBJECT_DICTS
from utils.util import get_all_objects_ori_class

from dataset.dataset_function_util import visualize_mapper_dict, visualize_datapoint





def transform_img_kpts(
    transforms: TransformList, all_instance_kpts: List[np.ndarray]
) -> List[np.ndarray]:
    
    all_instance_kpts = np.stack(all_instance_kpts)
    
    
    n, m, k = all_instance_kpts.shape

    assert k == 3
    assert m == 4 or m == 1, f"expected m = 4 but found m = {m}"
    flattened_kpts = all_instance_kpts.reshape((n*m, 3))
    flattened_coords = flattened_kpts[:, :2]

    transformed_coords = transforms.apply_coords(flattened_coords)
    keypoints = np.concatenate((transformed_coords, flattened_kpts[:, 2:3]), axis=1)

    new_keypoints = keypoints.reshape((n, m, 3))
    return new_keypoints

    # for kpts_instance in all_instance_kpts:
    #     flattened_coords = kpts_instance[:, :2]

    #     transformed_coords = transforms.apply_coords(flattened_coords)
    #     keypoints = np.concatenate((transformed_coords, flattened_kpts[:, 2:3]), axis=1)

    #     new_keypoints = keypoints.reshape((n, m, 3))
    #     result.append(Keypoints(new_keypoints))

    return result


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict, scaled_rgb=True, draw=False):
    if draw:
        visualize_datapoint(dataset_dict)
        
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
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
    all_instance_keypoints = []
    all_instance_widths = []
    all_instance_centerpoints = []
    all_instance_orientation = []
    all_instance_scales = []
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

        if len(keypoints)>0:
            sampled_idx = np.random.randint(0, len(keypoints))
            all_instance_keypoints.append(np.array(keypoints)[sampled_idx]) 
            all_instance_orientation.append(torch.tensor(ori)[sampled_idx])
            all_instance_widths.append(torch.tensor(width)[sampled_idx])
            all_instance_centerpoints.append(np.array(center)[sampled_idx])
            all_instance_scales.append(torch.tensor(scales)[sampled_idx])
            
            # all_instance_keypoints.append(Keypoints(np.array(keypoints)))
            # all_instance_orientation.append(torch.tensor(ori))
            # all_instance_widths.append(torch.tensor(width))
            # all_instance_centerpoints.append(Keypoints(np.array(center)))
            # all_instance_scales.append(torch.tensor(scales))

            category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])
            obj_box.append(object_dict["bbox"])

    # ///////////////////////////////////////////////////////////////
    # augmentations
    # ///////////////////////////////////////////////////////////////
    augs = T.AugmentationList(
        [
            # T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute", (200, 200)),
            T.RandomApply(
            T.RandomRotation(
                [-90.0, 90.0],
                expand=True,
                center=[[0.3, 0.3], [0.7, 0.7]],
                sample_style="range",
            ),
            prob=0.3,
        )
        ]
    )

    auginput = T.AugInput(image, boxes=np.array(obj_box))
    transform = augs(auginput)
    transformed_image = auginput.image.transpose(2, 0, 1)
    transformed_boxes = auginput.boxes
    # TODO Check these functions
    transformed_kpts = transform_img_kpts(
        transforms=transform, all_instance_kpts=all_instance_keypoints
    )
    transformed_center_kpts = transform_img_kpts(
        transforms=transform, all_instance_kpts=all_instance_centerpoints
    )
    
    transformed_ori = get_all_objects_ori_class(transformed_kpts)

    
    # TODO Check for float depths
    transformed_depth = transform.apply_image(depth[:, :, 0])
    # ///////////////////////////////////////////////////////////////

    image_input = np.concatenate(
        (transformed_image/255.0, transformed_depth[..., None]), axis=2
    )

    new_dict = {
        "input": torch.from_numpy(image_input).permute(2, 0, 1).float(),
        "rgb": transformed_image,
        "depth": transformed_depth,
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_boxes=Boxes(np.array(transformed_boxes)),
            gt_keypoints=torch.from_numpy(transformed_kpts),
            gt_centerpoints=torch.from_numpy(transformed_center_kpts),
            gt_orientations=torch.tensor(transformed_ori),  # NCx1
            gt_widths=torch.tensor(all_instance_widths),
            gt_scales=torch.tensor(all_instance_scales),
        ),
    }
    
    if draw:
        
        visualize_mapper_dict(new_dict, name=dataset_dict["image_id"])

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
            


