import torch
import copy
import cv2
import numpy as np
from detectron2.structures import Keypoints, Instances, Boxes
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

    annotations = dataset_dict.pop("annotations") # number of objects in the image 
    num_instances = len(annotations)
    all_instance_keypoints = []
    all_instance_widths = []
    all_instance_centerpoints = []
    all_instance_orientation = []
    category_ids = []
    obj_box = []
    # print("num_instances", num_instances)
    total_grasps = 0
    for i in range(num_instances): 
        keypts_lst = []
        object_dict = annotations[i]
        keypoints = object_dict["keypoints"]
        width = object_dict["grasp_width"]
        ori = object_dict["ori_clss"]
        center = object_dict["centers"]       
         
        total_grasps += len(keypoints) 
        # num_keypoints = len(keypoints)
        # for keypt_idx in range(num_keypoints):
        #     keypts_lst.append(keypoints[3 * keypt_idx : 3 * (keypt_idx + 1)])
        all_instance_keypoints.append(keypoints) 
        all_instance_orientation.append(ori)
        all_instance_widths.append(width)
        all_instance_centerpoints.append(center)
        
        category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])
        obj_box.append(object_dict["bbox"])
        
    # all_instance_keypoints = np.concatenate(category_ids, axis = 0)
    # category_ids = np.concatenate(category_ids, axis = 0)
    # assert all_instance_keypoints.shape == (total_grasps, 4, 3)
    # assert category_ids.shape == (total_grasps,)
   
    return {
        "image": torch.from_numpy(image_input).permute(2, 0, 1).float(),
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_boxes=Boxes(np.array(obj_box)),
            gt_keypoints=Keypoints(np.array(all_instance_keypoints)),
            gt_centerpoints=torch.tensor(all_instance_centerpoints),
            gt_orientations=torch.tensor(all_instance_orientation),      # NCx1
            gt_widths=torch.tensor(all_instance_widths)   
        ),
    }
