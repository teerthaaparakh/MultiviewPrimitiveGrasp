import cv2
import numpy as np
import os

from utils.global_utils import NUM_BINS 


def get_scene_and_image_id(color_image_path):
    path_elements = color_image_path.split(os.sep)
    assert path_elements[-1].startswith(
        "color_image_"
    ), "the image file does not start with color_image_"
    scene_id = path_elements[-3]
    img_id = int(path_elements[-1].replace("color_image_", "")[:-4])
    return scene_id, img_id




def get_obj_scene_dict(scene_data, bboxes, depth, cam_extr, cam_intr, index):
    obj_pose = np.array(scene_data["obj_poses"][index])  # 4x4
    obj_dim = np.array(scene_data["obj_dims"][index])  # single element array
    obj_type = scene_data["obj_types"][index]
    per_obj_grasp_poses = np.array(scene_data["grasp_poses"][index])
    per_obj_grasp_collisions = np.array(scene_data["grasp_collision"][index])
    per_obj_grasp_widths = np.array(scene_data["grasp_widths"][index])


    num_grasps = len(per_obj_grasp_poses)
    for idx in range(num_grasps):
        kpts_3d_cam = get_kpts_3d(poses[idx], per_obj_grasp_widths, cam_extr=cam_extr, world=False)
        kpts_2d = get_kpts_2d(kpts_3d_cam, cam_intr=cam_intr)
        grasp_dict = get_kpts_2d_detectron(kpts_2d, kpts_3d_cam, depth) #returns "offset_kpts", "center_2d", "scale", "valid",
        grasp_dict["grasp_width"] = per_obj_grasp_widths

    # converting all to numpy arrays
    for k, v in grasp_dict.items():
        grasp_dict[k] = np.array(v)
    # dict_keys = ["offset_kpts", "center_2d", "scale", "valid", "grasp_width", "orientation_bin"]

    grasp_dict["orientation_bin"] = get_orientation_class(
        grasp_dict["offset_kpts"]
    )

    """ get overall valid grasps: combination of
    1. when keypoints lie outside the image
    2. grasp is in collision with other objects"""
    valid = np.logical_and(grasp_dict["valid"], (~per_obj_grasp_collisions))

    obj_dict = {
        "obj_pose": obj_pose,
        "obj_dim": obj_dim,
        "obj_type": obj_type,
        "obj_index": index,
        "orientations": grasp_dict["orientation_bin"][valid],
        "center_kpts": grasp_dict["center_2d"][valid],
        "kpts": grasp_dict["offset_kpts"][valid],
        "scales": grasp_dict["scale"][valid],
        "grasp_widths": grasp_dict["grasp_width"][valid],
        "bbox": np.array(bboxes[index + 1]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    return obj_dict


def get_orientation_class(kpts_2d, ori_range=[0, np.pi]) -> np.ndarray:
    # kpts_2d: (num_grasps, 4, 2)
    # if (kpts_2d.shape) == 2:
    #     kpts_2d = kpts_2d[None, ...]

    kpt_2 = kpts_2d[:, 1, :]
    kpt_3 = kpts_2d[:, 2, :]

    kpt_2x = kpt_2[:, 0]
    kpt_2y = kpt_2[:, 1]

    kpt_3x = kpt_3[:, 0]
    kpt_3y = kpt_3[:, 1]

    delta_x = kpt_3x - kpt_2x
    delta_y = kpt_3y - kpt_2y

    angle = np.arctan2(delta_y, delta_x)
    angle[angle < 0] += np.pi

    bin_size = (ori_range[1] - ori_range[0]) / NUM_BINS
    bin_index = np.floor(angle / bin_size).astype(int)
    return bin_index  # (num_grasps,)



