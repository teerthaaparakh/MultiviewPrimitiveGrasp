import numpy as np
import logging
from copy import copy
import cv2
from utils.other_configs import *

import sys, os

# sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_debug_img_dir, get_debug_img_dir_orig

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def generate_keypoints(
    grasp_pose, grasp_width, cam_intr, cam_extr, depth, draw=False, draw_info=None
):
    """
    cam_extr: 4x4
    cam_intr: 3x3
    draw_info (Optional): img_file, obj_id, img_id, colli
    """

    # TODO (TP): good to break functions into small functions and one final
    # function that uses all the other functions, otherwise it is hard to read
    # and make changes to.
    # For example, projcting keypoints can very well be another function
    # even getting the 3d points for a grasp can be another function
    # TODO (TP): use variable names to be descriptive and avoid unnecessary keywords
    # for example, "ret" in kpts_3d_ret

    ret = []
    kpts_3d_ret = []
    kpts_2d_ret = []
    valids = []
    center_ret = []
    scale_ret = []
    for j in range(len(grasp_pose)):
        width = grasp_width[j]
        pose = grasp_pose[j]
        kpts_local_vertex = [
            [0, 0, width / 2],
            [-STICK_LEN, 0, width / 2],
            [-STICK_LEN, 0, -width / 2],
            [0, 0, -width / 2],
        ]

        kpts_3d = pose @ np.concatenate((kpts_local_vertex, np.ones((4, 1))), axis=1).T
        kpts_3d_ret.append(kpts_3d[:3, :].T)

        X_WC = cam_extr
        X_CW = np.linalg.inv(X_WC)
        kpts_3d_cam = X_CW @ kpts_3d
        kpts_3d_cam = kpts_3d_cam[:3, :].T

        # iz = np.argsort(pcd_cam[:, -1])[::-1]
        # pcd_cam = pcd_cam[iz]
        cam_intr = np.array(cam_intr)
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        px = (kpts_3d_cam[:, 0] * fx / kpts_3d_cam[:, 2]) + cx
        py = (kpts_3d_cam[:, 1] * fy / kpts_3d_cam[:, 2]) + cy
        kpts_2d_ret.append(np.concatenate(([px], [py]), axis=0).T)

        center, scale = get_center(kpts_3d, X_CW, cam_intr)
        center_ret.append(center)
        scale_ret.append(scale)

        height_y, width_x = depth.shape
        if (px < 0).all() or (px >= width_x).all():
            logging.warn("Projected keypoint is outside the image [x].")
            projection_valid = False
        elif (py < 0).all() or (py >= height_y).all():
            logging.warn("Projected keypoint is outside the image [y].")
            projection_valid = False
        else:
            projection_valid = True

        px = np.clip(np.int32(px), 0, width_x - 1)
        py = np.clip(np.int32(py), 0, height_y - 1)
        v = np.ones(4)

        if draw:
            img_dir = get_debug_img_dir_orig()
            os.makedirs(img_dir, exist_ok=True)
            img_file, obj_id, img_id, colli = draw_info
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.info("Image saved for debugging.")
            draw_on_image(
                image,
                px,
                py,
                v,
                name=os.path.join(img_dir, f"{img_id}_{obj_id}_{j}.png"),
            )

        else:
            for i in range(4):
                depth_val = depth[py[i], px[i]]
                keypt_depth = kpts_3d_cam[i, 2]
                # embed()
                # print(depth_val, keypt_depth)
                if (depth_val) > keypt_depth:
                    v[i] = 2

            # if ~colli[j]:
            #     draw_on_image(image, px, py, v, name = os.path.join(DATA_DIR, f"{img_id}_{obj_id}_{j}.png"))
            # logging.info(f"{img_id}_{obj_id}_{j} done!--------------")

        offset_x = (px - center[0])/width_x
        offset_y = (py - center[1])/height_y
        ret.append(np.concatenate(([offset_x], [offset_y], [v]), axis=0).T)
        valids.append(projection_valid)

    # ret = np.array(ret)
    # assert ret.shape == (
    #     len(grasp_pose),
    #     4,
    #     3,
    # ), "please check the shape of ret in gen_kpts.py"
    return (
        np.array(valids),
        np.array(ret),
        np.array(kpts_3d_ret),
        np.array(kpts_2d_ret),
        np.array(center_ret),
        np.array(scale_ret)
    )


def gen_kpts_norm_cano(
    grasp_pose, grasp_width, cam_intr, cam_extr, v_kpts, depth, draw=False, draw_info=None
):
    """
    cam_extr: 4x4
    cam_intr: 3x3
    draw_info (Optional): img_file, obj_id, img_id, colli
    """

    # TODO (TP): good to break functions into small functions and one final
    # function that uses all the other functions, otherwise it is hard to read
    # and make changes to.
    # For example, projcting keypoints can very well be another function
    # even getting the 3d points for a grasp can be another function
    # TODO (TP): use variable names to be descriptive and avoid unnecessary keywords
    # for example, "ret" in kpts_3d_ret

    ret = []
    kpts_3d_ret = []
    kpts_2d_ret = []
    valids = []
    center_ret = []
    scale_ret = []
    for j in range(len(grasp_pose)):
        
        pose = grasp_pose[j]
        kpts_local_vertex = [
            [0, 0, CANONICAL_LEN/2],
            [-CANONICAL_LEN, 0, CANONICAL_LEN / 2],
            [-CANONICAL_LEN, 0, -CANONICAL_LEN / 2],
            [0, 0, -CANONICAL_LEN / 2],
        ]

        kpts_3d = pose @ np.concatenate((kpts_local_vertex, np.ones((4, 1))), axis=1).T
        

        X_WC = cam_extr
        X_CW = np.linalg.inv(X_WC)
        kpts_3d_cam = X_CW @ kpts_3d
        kpts_3d_cam = kpts_3d_cam[:3, :].T

        # iz = np.argsort(pcd_cam[:, -1])[::-1]
        # pcd_cam = pcd_cam[iz]
        cam_intr = np.array(cam_intr)
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        center, scale = get_center(kpts_3d, X_CW, cam_intr)
        _, obj_id, img_id, _= draw_info
        center_ret.append(center)
        scale_ret.append(scale)
        
        # SCALING ######################################
        kpts_local_vertex_canonical = [
            [0, 0, CANONICAL_LEN * scale/2],
            [-CANONICAL_LEN * scale, 0, CANONICAL_LEN * scale/ 2],
            [-CANONICAL_LEN * scale, 0, -CANONICAL_LEN * scale / 2],
            [0, 0, -CANONICAL_LEN * scale / 2],
        ]
        kpts_3d_canonical = pose @ np.concatenate((kpts_local_vertex_canonical, np.ones((4, 1))), axis=1).T
        kpts_3d_ret.append(kpts_3d_canonical[:3, :].T)
        kpts_3d_cam_canonical = X_CW @ kpts_3d_canonical
        kpts_3d_cam_canonical = kpts_3d_cam_canonical[:3, :].T
        
        px = (kpts_3d_cam_canonical[:, 0] * fx / kpts_3d_cam_canonical[:, 2]) + cx
        py = (kpts_3d_cam_canonical[:, 1] * fy / kpts_3d_cam_canonical[:, 2]) + cy
        kpts_2d_ret.append(np.concatenate(([px], [py]), axis=0).T)


        height_y, width_x = depth.shape
        if (px < 0).all() or (px >= width_x).all():
            logging.warn("Projected keypoint is outside the image [x].")
            projection_valid = False
        elif (py < 0).all() or (py >= height_y).all():
            logging.warn("Projected keypoint is outside the image [y].")
            projection_valid = False
        else:
            projection_valid = True

        px = np.clip(np.int32(px), 0, width_x - 1)
        py = np.clip(np.int32(py), 0, height_y - 1)
        v = np.ones(4)

        if draw:
            img_dir = get_debug_img_dir()
            os.makedirs(img_dir, exist_ok=True)
            img_file, obj_id, img_id, colli = draw_info
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.info("Image saved for debugging.")
            draw_on_image(
                image,
                px,
                py,
                v,
                name=os.path.join(img_dir, f"{img_id}_{obj_id}_{j}.png"),
            )

        else:
            v = v_kpts[j]

            # if ~colli[j]:
            #     draw_on_image(image, px, py, v, name = os.path.join(DATA_DIR, f"{img_id}_{obj_id}_{j}.png"))
            # logging.info(f"{img_id}_{obj_id}_{j} done!--------------")

        offset_x = (px - center[0])/width_x
        offset_y = (py - center[1])/height_y
        ret.append(np.concatenate(([offset_x], [offset_y], [v]), axis=0).T)
        valids.append(projection_valid)

    # ret = np.array(ret)
    # assert ret.shape == (
    #     len(grasp_pose),
    #     4,
    #     3,
    # ), "please check the shape of ret in gen_kpts.py"
    return (
        np.array(valids),
        np.array(ret),
        np.array(kpts_3d_ret),
        np.array(kpts_2d_ret),
        np.array(center_ret),
        np.array(scale_ret)
    )
    
    

def get_center(kpts_3d, X_CW, cam_intr):
    center = kpts_3d[:3, :].T
    center = (center[0] + center[3]) / 2

    center_cam = X_CW @ np.array([*center, 1])
    center_cam = center_cam[:3]
    
    scale = center_cam[2]

    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    px = (center_cam[0] * fx / center_cam[2]) + cx
    py = (center_cam[1] * fy / center_cam[2]) + cy
    return [px, py, 2], scale


def draw_on_image(image, px, py, v, name=None):
    colors = [(255, 0, 0), (0, 255, 240), (0, 255, 0), (0, 0, 255), (240, 240, 0)]
    yellow = (255, 255, 240)

    image = copy(image)[:, :, ::-1].astype(np.uint8)
    for i in range(len(px)):
        if np.abs(v[i] - 2) < 1e-3:
            image = cv2.circle(
                image, (px[i], py[i]), radius=2, color=colors[i], thickness=-1
            )
        else:
            image = cv2.circle(
                image, (px[i], py[i]), radius=3, color=yellow, thickness=2
            )

    for i in range(len(px) - 1):
        image = cv2.line(
            image, (px[i], py[i]), (px[i + 1], py[i + 1]), (255, 255, 255), thickness=1
        )

    if name is not None:
        cv2.imwrite(name, image)

    return image


if __name__ == "__main__":
    import sys, os, json

    sys.path.append(os.environ["KGN_DIR"])
    from utils.path_util import get_dataset_dir

    scene_id = 0
    dataset_dir = get_dataset_dir()
    json_path = os.path.join(dataset_dir, str(scene_id), "scene_info.json")
    with open(json_path) as json_file:
        data = json.load(json_file)

    grasp_poses = data["grasp_poses"]
    # grasp_poses = np.array(
    #     [item for sublist in grasp_poses for item in sublist]
    # )  # sum(grasp_poses, [])

    grasp_collision = data["grasp_collision"]
    # grasp_collision = np.array(
    #     [item for sublist in grasp_collision for item in sublist]
    # )

    grasp_widths = data["grasp_widths"]
    # grasp_widths = np.array([item for sublist in grasp_widths for item in sublist])

    grasp_pose = grasp_poses[0][~grasp_collision[0]]
    grasp_width = grasp_widths[0][~grasp_collision[0]]

    ret, kpts_3d, kpts_2d = GenKpts(
        grasp_pose,
        grasp_width,
        cam_int,
        cam_poses[0],
        depth=np.load(current_dict["depth_file_name"]),
    )
