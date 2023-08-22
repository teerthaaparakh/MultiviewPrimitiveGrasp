import numpy as np
import sys
import os, os.path as osp
import typing as T

sys.path.append(os.environ["KGN_DIR"])
from utils.util import get_area
from utils.other_configs import *
from glob import glob
import json
from detectron2.structures import BoxMode
import cv2
from utils.seg import get_bb
from dataset.dataset_function_util import get_per_obj_processed_grasps, visualize_datapoint

def get_scene_data_item(scene_data, bboxes, rgb, depth, cam_extr, cam_intr, index):
    obj_pose = np.array(scene_data["obj_poses"][index])  # 4x4
    obj_dim = np.array(scene_data["obj_dims"][index])  # single element array
    obj_type = scene_data["obj_types"][index]
    per_obj_grasp_poses = np.array(scene_data["grasp_poses"][index])
    per_obj_grasp_collisions = np.array(scene_data["grasp_collision"][index])
    per_obj_grasp_widths = np.array(scene_data["grasp_widths"][index])

    # returns a dictionary with keys: offset_kpts, center_2d, scale, valid
    # and orientation_bin
    processed_grasps_dict = get_per_obj_processed_grasps(
        per_obj_grasp_poses, per_obj_grasp_widths, cam_extr, cam_intr, depth
    )

    # get overall valid grasps: combination of
    # 1. when keypoints lie outside the image
    # 2. grasp is in collision with other objects
    valid = np.logical_and(processed_grasps_dict["valid"], (~per_obj_grasp_collisions))

    obj_dict = {
        "obj_pose": obj_pose,
        "obj_dim": obj_dim,
        "obj_type": obj_type,
        "ori_clss": processed_grasps_dict["orientation_bin"][valid],
        "centers": processed_grasps_dict["center_2d"][valid],
        "keypoints": processed_grasps_dict["offset_kpts"][valid],
        "scales":processed_grasps_dict["scale"][valid],
        "bbox": list(bboxes[index + 1]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    return obj_dict


def generate_bbox(seg_img_path):
    """
    for the particular image, returns a dictionary
    with keys being the object ids, and the values being the bbs for
    the corresponding mask
    """
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
    indices = np.unique(seg_img)[1:]

    ll = {}
    for idx in indices:
        kernel = np.ones((3, 3), np.uint8)
        new_seg = seg_img == idx
        new_seg = cv2.erode(new_seg.astype(np.uint8), kernel)
        bb = get_bb(new_seg)
        if bb:
            ll[idx] = bb

    return ll


def get_scene_and_image_id(color_image_path):
    path_elements = color_image_path.split(os.sep)
    assert path_elements[-1].startswith(
        "color_image_"
    ), "the image file does not start with color_image_"
    scene_id = path_elements[-3]
    img_id = int(path_elements[-1].replace("color_image_", "")[:-4])
    return scene_id, img_id


def load_dataset(data_dir, draw=False) -> T.List[T.Dict]:
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
    for idx, color_image_path in enumerate(color_files_lst):

        print(f"Processing datapoint: {idx}")
        scene_id, img_id = get_scene_and_image_id(color_image_path)
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
            "height": 480,
            "width": 640,
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

        # grasp level information in the annotations list of dicts
        annotations = []
        num_grasps = []

        bboxes = generate_bbox(current_dict["seg_file_name"])
        num_objs = len(scene_data["grasp_poses"])
        # bowl, small cylinder,  cuboid, big cylinder, stick
        for j in range(num_objs):
            # TODO (TP): good to have a comment here for what these
            # two conditions are checking
            if (j + 1 in bboxes) and (get_area(bboxes[j + 1]) > 50):
                obj_dict = get_scene_data_item(scene_data, bboxes, rgb, depth, 
                                               current_dict["cam_ext"],
                                               current_dict["cam_int"],
                                               index=j)
                
                num_grasps.append(len(obj_dict["centers"]))
                annotations.append(obj_dict)

        current_dict["annotations"] = annotations
        current_dict["num_grasps"] = num_grasps
        list_dict.append(current_dict)

        if draw:
            visualize_datapoint(current_dict)

    return list_dict




