# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
import cv2

from detectron2.structures.keypoints import Keypoints
from detectron2.structures.instances import Instances
from fvcore.transforms.transform import Transform, TransformList

from detectron2.data import transforms as T
from typing import List
from utils.other_configs import OBJECT_DICTS


def transform_img_kpts(transforms: TransformList, all_instance_kpts: List[Keypoints]) -> List[Keypoints]:

    result = []
    for kpts_instance in all_instance_kpts:
        keypoint_tensor = kpts_instance.tensor
        n, m, _ = keypoint_tensor.shape
        assert keypoint_tensor.shape[-1] == 3
        flattened_kpts = keypoint_tensor.reshape((n*m, 3))
        flattened_coords = flattened_kpts[:, :2]

        transformed_coords = transforms.apply_coords(flattened_coords)
        keypoints = np.concatenate((transformed_coords, flattened_kpts[:, 2:3]), axis=1)

        new_keypoints = keypoints.reshape((n, m, 3))
        result.append(Keypoints(new_keypoints))

    return result


# Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    depth = np.load(dataset_dict["depth_file_name"])
    image = cv2.imread(dataset_dict["file_name"])[:,:, ::-1] / 255.0

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

        all_instance_keypoints.append(Keypoints(np.array(keypoints)))
        all_instance_orientation.append(torch.tensor(ori))
        all_instance_widths.append(torch.tensor(width))
        all_instance_centerpoints.append(Keypoints(np.array(center)))
        all_instance_scales.append(torch.tensor(scales))

        category_ids.append(OBJECT_DICTS[object_dict["obj_type"]])
        obj_box.append(object_dict["bbox"])


    # ///////////////////////////////////////////////////////////////
    # augmentations 
    # ///////////////////////////////////////////////////////////////
    augs = T.AugmentationList([
                # T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
                T.RandomFlip(prob=0.5),
                T.RandomCrop("absolute", (200, 200)),
            ]) 
    
    auginput = T.AugInput(image, boxes=np.array(obj_box))
    transform = augs(auginput)
    transformed_image = auginput.image.transpose(2, 0, 1)
    transformed_boxes = auginput.boxes
    new_all_keypoints = []
    transformed_kpts = transform_img_kpts(transforms=transform, 
                                          all_instance_kpts=all_instance_keypoints)
    center_kpts = transform_img_kpts(transforms=transform, 
                                          all_instance_kpts=all_instance_centerpoints)
    transformed_depth = transform.apply_image(depth[:, :, 0])
    # ///////////////////////////////////////////////////////////////

    image_input = np.concatenate((transformed_image, transformed_depth[..., None]), 
                                 axis=2)

    new_dict = {
        "image": torch.from_numpy(image_input).permute(2, 0, 1).float(),
        "height": h,
        "width": w,
        "instances": Instances(
            (h, w),
            gt_classes=torch.tensor(category_ids, dtype=torch.long),
            gt_boxes=Boxes(np.array(obj_box)),
            gt_keypoints=all_instance_keypoints,
            gt_centerpoints=all_instance_centerpoints,
            gt_orientations=all_instance_orientation,  # NCx1
            gt_widths=all_instance_widths,
            gt_scales=all_instance_scales
        ),
    }

    return new_dict

from detectron2.data import detection_utils as utils
 # Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage

    augs = T.AugmentationList([
                T.RandomApply(T.RandomBrightness(0.9, 1.1), prob=0.5),
                T.RandomFlip(prob=0.5),
                T.RandomCrop("absolute", (200, 200)),
            ]) 
    
    auginput = T.AugInput(image)
    transform = augs(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }
dataloader = build_detection_train_loader(cfg, mapper=mapper)