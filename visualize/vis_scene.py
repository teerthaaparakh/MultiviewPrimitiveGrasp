import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
import cv2
import json
from IPython import embed


# def load_scene(scene_id):
#     dataset_dir = get_dataset_dir()
#     imgs_path = os.path.join(dataset_dir, str(scene_id), "color_images")
#     # depths_path = os.path.join(dataset_dir, str(scene_id), "depth_img")
#     depths_raw_path = os.path.join(dataset_dir, str(scene_id), "depth_raw")
#     scene_info_path = os.path.join(dataset_dir, str(scene_id), "scene_info.json")
#     colors = []
#     depths = []

#     img_files = sorted(os.listdir(imgs_path))
#     depth_raw_files = sorted(os.listdir(depths_raw_path))

#     for i in img_files:
#         color = cv2.imread(os.path.join(imgs_path, i))
#         color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
#         colors.append(color)

#     for i in depth_raw_files:
#         depth = np.load(os.path.join(depths_raw_path, i))
#         depths.append(depth)

#     with open(scene_info_path) as json_file:
#         data = json.load(json_file)

#     return colors, depths, data["camera_poses"], data["intrinsic"]


# def meshcat_scene(vis, scene_id):
#     colors, depths, camera_poses, intrinsic = load_scene(scene_id)
#     combined_pcd, combined_color = get_combined_pcd(
#         colors, depths, camera_poses, intrinsic
#     )
#     vis.view_pcd(combined_pcd, combined_color)
