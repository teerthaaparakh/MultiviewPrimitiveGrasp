from scipy.spatial.transform import Rotation as R
import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.other_configs import *


def ViewGrasps(vis, poses=None, grip_widths=None, name=None, freq=100):
    """
    input
    poses: Nx4x4
    """
    stick_len = STICK_LEN
    stick_radius = STICK_RADIUS

    r_cl1 = R.from_euler("z", 90, degrees=True)
    r_cl3 = R.from_euler("x", 90, degrees=True)

    pose_cl1 = r_cl1.as_matrix()
    pose_cl3 = r_cl3.as_matrix()

    for i, pose in enumerate(poses):
        grip_width = grip_widths[i]
        X_pose_cl1 = np.eye(4)
        X_pose_cl1[:3, :3] = pose_cl1
        X_pose_cl1[:3, 3] = [-stick_len / 2, 0, grip_width / 2]

        X_pose_cl2 = np.eye(4)
        X_pose_cl2[:3, :3] = pose_cl1
        X_pose_cl2[:3, 3] = [-stick_len / 2, 0, -grip_width / 2]

        X_pose_cl3 = np.eye(4)
        X_pose_cl3[:3, :3] = pose_cl3
        X_pose_cl3[:3, 3] = [-stick_len, 0, 0]

        vis.view_cylinder(
            stick_len, stick_radius, X_pose=pose @ X_pose_cl1, name=f"scene/cl{i}1",
            color=0xFF0000
        )  # side stick
        vis.view_cylinder(
            stick_len, stick_radius, X_pose=pose @ X_pose_cl2, name=f"scene/cl{i}2",
            color=0x0000FF
        )  # side stick
        vis.view_cylinder(
            grip_width, stick_radius, X_pose=pose @ X_pose_cl3, name=f"scene/cl{i}3",
            color=0x00FF00
        )  # mid stick
