import numpy as np
import logging
from copy import copy
import cv2
from utils.other_configs import *
import random
import os, os.path as osp

import sys

# sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_debug_img_dir
from utils.util import get_orientation_class

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def get_kpts_3d(pose, width, cam_extr, world=False):
    """
    pose: 4x4 single grasp pose
    width: float
    Returns
        kpts_3d: 3x4 (xyz for each of the four keypoints)
    """
    # width = CANONICAL_LEN
    length = STICK_LEN / 2
    kpts_local_vertex = [
        [0, 0, width / 2],
        [-length, 0, width / 2],
        [-length, 0, -width / 2],
        [0, 0, -width / 2],
    ]
    kpts_3d = pose @ np.concatenate((kpts_local_vertex, np.ones((4, 1))), axis=1).T
    if world:
        return kpts_3d[:3, :].T
    else:
        X_WC = cam_extr
        X_CW = np.linalg.inv(X_WC)
        kpts_3d_cam = X_CW @ kpts_3d
        return kpts_3d_cam[:3, :].T


def get_kpts_2d(kpts_3d_cam, cam_intr):
    """
    kpts_3d: 4x3
    cam_extr: 4x4
    cam_intr: 3x3

    Returns
        4x2 (x, y projection of the 4 keypoints on the image)
    """
    cam_intr = np.array(cam_intr)
    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    px = (kpts_3d_cam[:, 0] * fx / kpts_3d_cam[:, 2]) + cx
    py = (kpts_3d_cam[:, 1] * fy / kpts_3d_cam[:, 2]) + cy
    return np.stack((px, py), axis=-1)


def get_kpts_2d_validity(kpts_2d, img_height, img_width):
    """
    kpts_2d: 4x2
    Returns:
        valid: scalar
    """
    px, py = kpts_2d[:, 0], kpts_2d[:, 1]
    if (px < 0).all() or (px >= img_width).all():
        # logging.warn("Projected keypoint is outside the image [x].")
        return False
    elif (py < 0).all() or (py >= img_height).all():
        logging.warn("Projected keypoint is outside the image [y].")
        return False
    return True


def get_kpts_2d_detectron(
    kpts_2d: np.ndarray, kpts_3d_cam: np.ndarray, depth: np.ndarray
):
    """
    obtains scale, center and keypoints offset (with visibility vals) for detectron

    kpts_2d: 4x2
    kpts_3d_cam: 4x3
    img_width, img_height: int, int
    depth: array of shape (img_height, img_width)

    Returns: dictionary
        offset: 4x3 (x, y, v) for each of the four points (normalized and scaled by
                                                           center depth)
        scale: float
        center: 3  (x, y, v) for the center
    """

    assert kpts_2d.shape == (4, 2)
    assert kpts_3d_cam.shape == (4, 3)

    h, w = depth.shape

    valid = get_kpts_2d_validity(kpts_2d, h, w)
    v = np.ones(4)
    center_2d = (kpts_2d[0] + kpts_2d[3]) / 2

    if valid:
        center_3d_cam = (kpts_3d_cam[0] + kpts_3d_cam[3]) / 2
        scale = center_3d_cam[2]

        px, py = kpts_2d[:, 0], kpts_2d[:, 1]
        px = np.clip(np.int32(px), 0, w - 1)
        py = np.clip(np.int32(py), 0, h - 1)

        clipped_kpts_2d = np.stack((px, py), axis=-1)
        offsets = scale * (clipped_kpts_2d - center_2d) / np.array([w, h])
        
        assert offsets.shape == (4, 2)

        # offsets = clipped_kpts_2d

        depth_val = depth[clipped_kpts_2d[:, 1], clipped_kpts_2d[:, 0]]
        kpts_depth = kpts_3d_cam[:, 2]
        v[depth_val > kpts_depth] = 2

    else:
        # putting in dummy values, matching the dimensions
        offsets = np.zeros_like(kpts_2d)
        scale = 1.0

    return {
        "offset_kpts": np.concatenate((offsets, v.reshape((4, 1))), axis=1),
        "center_2d": np.array([*(center_2d), 2]),
        "scale": scale,
        "valid": valid,
    }


def process_grasp(grasp_pose, grasp_width, cam_extr, cam_intr, depth):
    """
    returns a dictionary with keys: offset_kpts, center_2d, scale, valid
    """
    kpts_3d_cam = get_kpts_3d(grasp_pose, grasp_width, cam_extr=cam_extr, world=False)
    kpts_2d = get_kpts_2d(kpts_3d_cam, cam_intr=cam_intr)
    grasp_projection_dict = get_kpts_2d_detectron(kpts_2d, kpts_3d_cam, depth)
    
    return grasp_projection_dict


def get_per_obj_processed_grasps(poses, widths, cam_extr, cam_intr, depth):
    dict_keys = ["offset_kpts", "center_2d", "scale", "valid"]
    combined_dicts = {k: [] for k in dict_keys}

    num_grasps = len(poses)
    for idx in range(num_grasps):
        grasp_dict = process_grasp(poses[idx], widths[idx], cam_extr, cam_intr, depth)
        
        for k in dict_keys:
            combined_dicts[k].append(grasp_dict[k])

    # converting all to numpy arrays
    for k, v in combined_dicts.items():
        combined_dicts[k] = np.array(v)

    # add the orientation bin index
    combined_dicts["orientation_bin"] = get_orientation_class(
        combined_dicts["offset_kpts"]
    )
    
    return combined_dicts


def draw_grasp_on_image(image, grasp_dict, name=None):
    h, w = image.shape[:2]
    v =  grasp_dict["offset_kpts"][:, 2]
    offsets_scaled = np.array([[w, h]]) * grasp_dict["offset_kpts"][:, :2] / grasp_dict["scale"]
    kpts_2d = grasp_dict["center_2d"][:2].reshape((1, 2)) + offsets_scaled
    # kpts_2d = grasp_dict["offset_kpts"][:, :2]
    assert kpts_2d.shape == (4, 2)
    px, py = kpts_2d.T.astype(np.int64)
    # print("inside_draw_on_image", px, py, v)
    colors = [(255, 0, 0), (0, 255, 240), (0, 255, 0), (0, 0, 255), (240, 240, 0)]
    yellow = (255, 255, 240)

    image = copy(image).astype(np.uint8)
    for i in range(len(px)):
        if np.abs(v[i] - 2) < 1e-3:
            image = cv2.circle(
                image, (px[i], py[i]), radius=2, color=colors[i], thickness=-1
            )
        else:
            image = cv2.circle(
                image, (px[i], py[i]), radius=3, color=yellow, thickness=2
            )

    cx, cy = grasp_dict["center_2d"][:2].astype(np.uint16)
    orientation_bin = str(grasp_dict["orientation_bin"])

    cv2.putText(image, orientation_bin, (cx, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 
                2, cv2.LINE_AA)

    for i in range(len(px) - 1):
        image = cv2.line(
            image, (px[i], py[i]), (px[i + 1], py[i + 1]), (255, 255, 255), thickness=1
        )
    if name is not None:
        cv2.imwrite(name, image)
    return image

def visualize_datapoint(datapoint):
    """
    datapoint represents an elements from the dataset list used by detectron
    TODO: check the orientation bin
    """

    os.makedirs(get_debug_img_dir(), exist_ok=True)
    rgb = copy(cv2.imread(datapoint["file_name"]))

    for idx, obj_dict in enumerate(datapoint["annotations"]):
        #  bbox is (min_col, min_row, max_col, max_row)
        bbox = obj_dict["bbox"]
        img_with_obj_bb = cv2.rectangle(rgb, bbox[:2], bbox[2:], color=(127, 0, 255), thickness=1)

        # img_with_obj_bb = copy(img_with_obj_bb)
        # drawing any 5 random chosen grasps for the object
        num_grasps = len(obj_dict["centers"])
        print(f"    object id {idx}, num_grasps {num_grasps}")
        k = min(5, num_grasps)
        indices = random.sample(range(num_grasps), k=k)

        for i in indices:
            grasp_dict = {
                "offset_kpts": obj_dict["keypoints"][i],
                "center_2d": obj_dict["centers"][i],
                "scale": obj_dict["scales"][i],
                "orientation_bin": obj_dict["ori_clss"][i],
            }
            
            # print("grasp dict", grasp_dict)
            rgb = draw_grasp_on_image(img_with_obj_bb, grasp_dict)

    scene_id = datapoint["scene_id"]
    img_id = datapoint["image_id"]
    cv2.imwrite(osp.join(get_debug_img_dir(), f"{scene_id}_{img_id}.png"), rgb)
