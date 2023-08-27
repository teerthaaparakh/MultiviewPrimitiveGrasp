import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.other_configs import *
from scipy.spatial.transform import Rotation as R

from dataloader.dataset_func import dataset_function
import cv2
from IPython import embed


def cam_pose(cam_int, points2d, points3d):
    retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
        points3d, points2d, cam_int, distCoeffs=np.array([])
    )
    return retval, rvecs, tvecs, reprojectionError


def get_grasp_pose(kpts_2d, grasp_width, cam_ext, cam_int, scale=1):
    # local_3d = np.array(
    #     [
    #         [0, 0, grasp_width / 2],
    #         [-STICK_LEN, 0, grasp_width / 2],
    #         [-STICK_LEN, 0, -grasp_width / 2],
    #         [0, 0, -grasp_width / 2],
    #     ]
    # )

    local_3d = (
        np.array(
            [
                [0, 0, CANONICAL_LEN / 2],
                [-CANONICAL_LEN, 0, CANONICAL_LEN / 2],
                [-CANONICAL_LEN, 0, -CANONICAL_LEN / 2],
                [0, 0, -CANONICAL_LEN / 2],
            ]
        )
        * scale
    )

    retval, rvecs, tvecs, reprojectionError = cam_pose(cam_int, kpts_2d, local_3d)
    r = R.from_rotvec(np.array(rvecs).reshape(-1, 3))
    rotation_matrix = r.as_matrix()[0]  # orientation of world in camera frame
    tvecs = np.array(tvecs).reshape(-1, 3)[0]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = tvecs
    grasp_pose_calc = cam_ext @ transform_matrix

    return grasp_pose_calc


if __name__ == "__main__":
    ll = dataset_function(1)
    data = ll[0]
    annotations = data["annotations"]
    obj_dict = annotations[0]
    kpts_3d = obj_dict["kpts_3d"][0]
    kpts_2d = obj_dict["kpts_2d"][0]
    grasp_pose = obj_dict["grasp_pose"][0]
    grasp_width = obj_dict["grasp_width"][0]
    scale = obj_dict["scales"][0]

    cam_int = np.array(data["cam_int"])
    cam_ext = np.array(data["cam_ext"])  # orientation of camera in world frame
    scene_id = data["scene_id"]

    grasp_pose_calc = get_grasp_pose(kpts_2d, grasp_width, cam_ext, cam_int)

    # import pdb; pdb.set_trace()
    print(grasp_pose, grasp_pose_calc)
