import numpy as np
import sys, os
sys.path.append(os.environ["KGN_DIR"])
from utils.path_util import get_dataset_dir
import cv2
import json
from IPython import embed

def get_pcd(
    in_world=True,
    filter_depth=False,
    depth_min=0.20,
    depth_max=1.50,
    cam_ext_mat=None,
    rgb_image=None,
    depth_image=None,
    seg_image=None,
    cam_intr_mat=None,
):
    """
    Get the point cloud from the entire depth image
    in the camera frame or in the world frame.
    Returns:
        2-element tuple containing

        - np.ndarray: point coordinates (shape: :math:`[N, 3]`).
        - np.ndarray: rgb values (shape: :math:`[N, 3]`).
        - np.ndarray: seg values (shape: :math:`[N, 1]`).
    """

    rgb_im = rgb_image
    depth_im = depth_image
    seg_im = seg_image
    img_shape = rgb_im.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    # pcd in camera from depth

    # assert (
    #     len(seg_im.shape) == 3
    # ), "the segmentation mask must be 3D with last dimension containing 1 channel"
    depth = depth_im.reshape(-1) 

    rgb = None
    if rgb_im is not None:
        rgb = rgb_im.reshape(-1, 3)
    # if seg_im is not None:
    #     seg = seg_im.reshape(-1, 1)
    depth_min = depth_min
    depth_max = depth_max
    if filter_depth:
        valid = depth > depth_min
        valid = np.logical_and(valid, depth < depth_max)
        depth = depth[valid]
        if rgb is not None:
            rgb = rgb[valid]
        # if seg is not None:
        #     seg = seg[valid]
        uv_one_in_cam = get_uv_one_in_cam(cam_intr_mat, img_height, img_width)[:, valid]
    else:
        uv_one_in_cam = get_uv_one_in_cam(cam_intr_mat, img_height, img_width)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    if not in_world:
        pcd_pts = pts_in_cam.T
        pcd_rgb = rgb
        # pcd_seg = seg
        return pcd_pts, pcd_rgb
    else:
        if cam_ext_mat is None:
            raise ValueError(
                "Please call set_cam_ext() first to set up"
                " the camera extrinsic matrix"
            )
        
        pts_in_cam = np.concatenate(
            (pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0
        )
        pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
        pcd_pts = pts_in_world[:3, :].T
        pcd_rgb = rgb
        # pcd_seg = seg

        return pcd_pts, pcd_rgb  
    
def get_combined_pcd(colors, depths, cams_extr, cam_intr, idx=None):
    pcd_pts = []
    pcd_rgb = []

    if idx is None:
        idx = list(range(len(colors)))

    count = 0
    for color, depth, cam_extr in zip(colors, depths, cams_extr):
        if count in idx:
            pts, rgb = get_pcd(
                cam_ext_mat=cam_extr, rgb_image=color, depth_image=depth, cam_intr_mat = cam_intr
            )
            pcd_pts.append(pts)
            pcd_rgb.append(rgb)
        count += 1

    return np.concatenate(pcd_pts, axis=0), np.concatenate(pcd_rgb, axis=0)


def get_uv_one_in_cam(cam_intr_mat, img_height, img_width):
    cam_int_mat_inv = np.linalg.inv(cam_intr_mat)

    img_pixs = np.mgrid[0: img_height,
                        0: img_width].reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    _uv_one = np.concatenate((img_pixs,np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)
    
    return uv_one_in_cam


def load_scene(scene_id):
    dataset_dir = get_dataset_dir()
    imgs_path = os.path.join(dataset_dir, str(scene_id), "color_images")
    # depths_path = os.path.join(dataset_dir, str(scene_id), "depth_img")
    depths_raw_path  = os.path.join(dataset_dir, str(scene_id), "depth_raw")
    scene_info_path = os.path.join(dataset_dir, str(scene_id), "scene_info.json")
    colors = []
    depths = []
    
    img_files = sorted(os.listdir(imgs_path))
    depth_raw_files = sorted(os.listdir(depths_raw_path))
    
    
    for i in img_files:
        color = cv2.imread(os.path.join(imgs_path,i))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        colors.append(color) 
    
    for i in depth_raw_files:
        depth = np.load(os.path.join(depths_raw_path,i))
        depths.append(depth) 
        
    with open(scene_info_path) as json_file:
        data = json.load(json_file)
        
    return colors, depths, data["camera_poses"], data["intrinsic"]


def meshcat_scene(vis, scene_id):
    colors, depths, camera_poses, intrinsic = load_scene(scene_id)
    combined_pcd, combined_color = get_combined_pcd(colors, depths, camera_poses, intrinsic)
    vis.view_pcd(combined_pcd, combined_color)


    
    
    