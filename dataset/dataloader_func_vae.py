import torch
import copy
import cv2
import numpy as np
from detectron2.structures import Keypoints, Boxes, Instances
import sys, os

sys.path.append(os.environ["KGN_DIR"])
import torch
from utils.other_configs import *


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper_vae(dataset_dict):
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
        scale = object_dict["scales"]
        # 5x1x2

        all_instance_keypoints.append(keypoints.reshape(4, -1))
        all_instance_orientation.append(torch.tensor(ori))
        all_instance_widths.append(torch.tensor(width))
        all_instance_centerpoints.append(center[None, :])
        all_instance_scales.append(torch.tensor(scale))

        category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])
        obj_box.append(object_dict["bbox"])

    return {
        "image": torch.from_numpy(image_input).permute(2, 0, 1).float(),
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_boxes=Boxes(np.array(obj_box)),
            gt_keypoints=Keypoints(np.array(all_instance_keypoints)),
            gt_centerpoints=Keypoints(np.array(all_instance_centerpoints)),
            gt_orientations=torch.tensor(all_instance_orientation),  # NCx1
            gt_widths=torch.tensor(all_instance_widths),
            gt_scales=torch.tensor(all_instance_scales),
        ),
    }


if __name__ == "__main__":
    from dataset_func_vae import dataset_function_vae

    ll = dataset_function_vae(2)

    dd = mapper_vae(ll[0])

    print(dd)
