import torch
import copy
import cv2
import numpy as np
from detectron2.structures import Keypoints, Instances
import torch
from utils.other_configs import *


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    depth = np.load(dataset_dict["depth_file_name"])
    image = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED) / 255.0

    # assert len(depth.shape) == 3
    # logging.info("Depth image shape:", depth.shape)
    if len(depth.shape) == 2:
        depth = depth[..., None]
    if len(image.shape) == 2:
        image = image[..., None]

    h, w = image.shape[:2]
    # logging.info("Number of channels in image:", image.shape[2])

    image_input = np.concatenate((image, depth), axis=2)

    annotations = dataset_dict.pop("annotations")
    num_instances = len(annotations)
    all_instance_keypoints = []
    category_ids = []
    for i in range(num_instances):
        keypts_lst = []
        object_dict = annotations[i]
        keypoints = object_dict["ret"]
        num_keypoints = len(keypoints) // 3
        for keypt_idx in range(num_keypoints):
            keypts_lst.append(keypoints[3 * keypt_idx : 3 * (keypt_idx + 1)])
        all_instance_keypoints.append(keypts_lst)
        category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])

    return {
        "image": torch.from_numpy(image_input).permute(2, 0, 1).float(),
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_keypoints=Keypoints(all_instance_keypoints),
        ),
    }
