import numpy as np
import sys
import os, os.path as osp

from data.bboxes import generate_bbox
from data.data_utils import get_scene_data_item
sys.path.append(os.environ["KGN_DIR"])

import typing as T
import pickle
from utils.path_util import get_pickled_data_dir, get_data_dir

# from dataset.dataset_function import load_dataset
import random
import logging
from glob import glob
import numpy as np


from utils.util import get_area
from utils.other_configs import *
from glob import glob
import json
from detectron2.structures import BoxMode
import cv2
from utils.seg import get_bb
from dataset.dataset_function_util import (
    get_per_obj_processed_grasps,
    visualize_datapoint,
)
import random

def load_dataset_wrapper(t="train"):
    name = "VAE"

    path = osp.join(get_pickled_data_dir(), f"{name}_{t}.pkl")
    if osp.exists(path):
        with open(path, "rb") as f:
            result = pickle.load(f)
    else:
        logging.warn(f"Dataset {name} pickle not found. Generating ...")
        os.makedirs(get_pickled_data_dir(), exist_ok=True)
        
        initial_data_dir = get_data_dir()
        _scene_dirs = glob(initial_data_dir + "/*/color_images")
        scene_dirs = [s.split(os.sep)[-2] for s in _scene_dirs]

        print(f"Example scenes found (last 5): {scene_dirs[-5:]}")
        # randomly choose 1% percent of the total scenes
        k = max(1, int(len(scene_dirs) * 0.01))
        chosen_for_validation = random.sample(scene_dirs, k=k)

        train_dataset_lst = load_dataset(
            get_data_dir(), scenes_to_skip=chosen_for_validation, num_samples=9000000
        )

        val_dataset_lst = load_dataset(
            get_data_dir(), exclusively=chosen_for_validation, num_samples=100000
        )

        train_path = osp.join(get_pickled_data_dir(), f"{name}_train.pkl")
        val_path = osp.join(get_pickled_data_dir(), f"{name}_val.pkl")

        with open(train_path, "wb") as f:
            pickle.dump(train_dataset_lst, f)

        with open(val_path, "wb") as f:
            pickle.dump(val_dataset_lst, f)

        if t == "train":
            result = train_dataset_lst
        else:
            result = val_dataset_lst

    logging.info(f"Dataset {name}_{t} loaded. Length of dataset: {len(result)}")
    return result


def load_dataset(
    data_dir,
    num_samples=10,
    scenes_to_skip: T.List[str] = [],
    exclusively: T.Optional[T.List] = None,
) -> T.List[T.Dict]:
    """
    Requires the data_dir to have the following structure:
    data_dir
        - {scene_id_1}
            - color_images
                - {1}.png
                - {2}.png
                - ...
            - depth_img
                - {1}.png
                - ...
            - seg_labels
                - {1}.png
                - ...
            - scene_info.json
                TODO: specify what scene json is expected to contain
        - {scene_id_2}
            - ...
        ...
    """
    # total_num_data = NUM_TRAINING_DATA
    list_dict = []
    color_files_lst = sorted(
        glob(os.path.join(data_dir, "*/color_images/*.png"), recursive=True)
    )
    k = min(num_samples, len(color_files_lst))
    new_color_files_lst = sorted(random.sample(color_files_lst, k=k))
    for idx, color_image_path in enumerate(new_color_files_lst):
        # print(f"Processing datapoint: {idx}")
        scene_id, img_id = get_scene_and_image_id(color_image_path)

        if exclusively is not None:
            if not (str(scene_id) in exclusively):
                continue

        if str(scene_id) in scenes_to_skip:
            print(f"Scene {scene_id} is skipped.")
            continue

        print(f"Processing scene and image: {scene_id}, {img_id}")
        scene_path = osp.join(data_dir, scene_id)

        json_path = os.path.join(scene_path, "scene_info.json")
        with open(json_path, "r") as json_file:
            scene_data = json.load(json_file)

        # image level information
        current_dict = {
            "file_name": os.path.join(
                scene_path, f"color_images/color_image_{img_id}.png"
            ),
            "depth_file_name": os.path.join(
                scene_path, f"depth_raw/depth_raw_{img_id}.npy"
            ),
            "image_id": idx,
            "seg_file_name": os.path.join(
                scene_path, f"seg_labels/segmask_label_{img_id}.jpg"
            ),
            "scene_id": scene_id,
            "cam_ext": scene_data["camera_poses"][img_id],
            "cam_int": scene_data["intrinsic"],
        }

        rgb = cv2.imread(current_dict["file_name"])
        depth = np.load(current_dict["depth_file_name"])
        
        rgb_shape = rgb.shape
        
        current_dict["height"] = rgb_shape[0]
        current_dict["width"] = rgb_shape[1]

        # grasp level information in the annotations list of dicts
        annotations = []
        num_grasps = []

        bboxes = generate_bbox(current_dict["seg_file_name"])
        num_objs = len(scene_data["grasp_poses"])
        # bowl, small cylinder,  cuboid, big cylinder, stick
        for j in range(num_objs):
            # TODO (TP): good to have a comment here for what these
            # two conditions are checking
            if (j + 1 in bboxes) and (get_area(bboxes[j + 1]) > 100):
                obj_dict = get_obj_scene_dict(
                    scene_data,
                    bboxes,
                    depth,
                    current_dict["cam_ext"],
                    current_dict["cam_int"],
                    index=j,
                )
                if obj_dict['center_kpts'] > 0:
                    num_grasps.append(len(obj_dict["center_kpts"]))
                    annotations.append(obj_dict)

        current_dict["annotations"] = annotations
        current_dict["num_grasps"] = num_grasps
        list_dict.append(current_dict)

    return list_dict
