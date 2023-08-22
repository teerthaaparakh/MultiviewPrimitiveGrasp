import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import sys, os
import os.path as osp
from scipy.spatial.transform import Rotation as R
from dataset.dataset_function_util import get_kpts_3d

sys.path.append(os.environ["KGN_DIR"])
# from visualize.vis_grasp import ViewGrasps
# from visualize.vis_scene import meshcat_scene
from utils.other_configs import STICK_LEN, STICK_RADIUS
# from pyngrok import ngrok
import cv2, json
from utils.pcd_util import get_combined_pcd

class VizServer:
    def __init__(self, port_vis=6000) -> None:
        zmq_url = f"tcp://127.0.0.1:{port_vis}"
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)

        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

    def view_cylinder(self, ht, r, X_pose, name="scene/cylinder", color=None):
        if color is None:
            color = 0x22DD22
        self.mc_vis[name].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=color, opacity=1)
        )

        self.mc_vis[name].set_transform(X_pose)

    def view_pcd(self, pts, colors=None, name="scene/pcd", size=0.005):
        """
        inputs -
        pts: Nx3
        colors: Nx3, 3 for RGB channels

        """
        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis[name].set_object(
            g.PointCloud(pts.T, color=colors.T / 255.0, size=size)
        )

    def view_grasps(self, poses, grip_widths, name="scene/grasps", freq=1):
        # stick_len = STICK_LEN
        stick_len = STICK_LEN / 2
        stick_radius = STICK_RADIUS

        r_cl1 = R.from_euler("z", 90, degrees=True)
        r_cl3 = R.from_euler("x", 90, degrees=True)

        pose_cl1 = r_cl1.as_matrix()
        pose_cl3 = r_cl3.as_matrix()

        total_grasps = len(poses)
        
        for i in range(0, total_grasps, freq):
            pose = poses[i]
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

            self.view_cylinder(
                stick_len, stick_radius, X_pose=pose @ X_pose_cl1, name=f"{name}/cl{i}1",
                color=0xFF0000
            )  # side stick
            self.view_cylinder(
                stick_len, stick_radius, X_pose=pose @ X_pose_cl2, name=f"{name}/cl{i}2",
                color=0x0000FF
            )  # side stick
            self.view_cylinder(
                grip_width, stick_radius, X_pose=pose @ X_pose_cl3, name=f"{name}/cl{i}3",
                color=0x00FF00
            )  # mid stick


    def close(self):
        self.mc_vis.close()

class VizScene(VizServer):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def load_scene(self, scene_id):
        dataset_dir = self.data_dir
        imgs_path = os.path.join(dataset_dir, str(scene_id), "color_images")
        # depths_path = os.path.join(dataset_dir, str(scene_id), "depth_img")
        depths_raw_path = os.path.join(dataset_dir, str(scene_id), "depth_raw")
        scene_info_path = os.path.join(dataset_dir, str(scene_id), "scene_info.json")
        colors = []
        depths = []

        img_files = sorted(os.listdir(imgs_path))
        depth_raw_files = sorted(os.listdir(depths_raw_path))

        for i in img_files:
            color = cv2.imread(os.path.join(imgs_path, i))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            colors.append(color)

        for i in depth_raw_files:
            depth = np.load(os.path.join(depths_raw_path, i))
            depths.append(depth)

        with open(scene_info_path) as json_file:
            data = json.load(json_file)

        return colors, depths, data["camera_poses"], data["intrinsic"]

    def load_grasps(self, scene_id):
        
        scene_path = osp.join(self.data_dir, f"{scene_id}")

        json_path = os.path.join(scene_path, "scene_info.json")
        with open(json_path, "r") as json_file:
            scene_data = json.load(json_file)
        per_obj_grasp_poses = scene_data["grasp_poses"]
        per_obj_grasp_widths = scene_data["grasp_widths"]

        for i in range(len(per_obj_grasp_poses)):
            self.view_grasps(per_obj_grasp_poses[i], per_obj_grasp_widths[i], name=f"scene/grasps/{i}")
        

        colors, depths, camera_poses, intrinsic = self.load_scene(scene_id)
        combined_pcd, combined_color = get_combined_pcd(
            colors, depths, camera_poses, intrinsic
        )
        self.view_pcd(combined_pcd, combined_color)

        # print(per_obj_grasp_poses.shape)

    def overlay_grasp(self, scene_id):

        scene_path = osp.join(self.data_dir, f"{scene_id}")

        json_path = os.path.join(scene_path, "scene_info.json")
        with open(json_path, "r") as json_file:
            scene_data = json.load(json_file)

        per_obj_grasp_poses = scene_data["grasp_poses"]
        per_obj_grasp_widths = scene_data["grasp_widths"]

        colors, depths, camera_poses, intrinsic = self.load_scene(scene_id)

        kpts_3d = get_kpts_3d(per_obj_grasp_poses[0][0], per_obj_grasp_widths[0][0], 
                              None, world=True)
        
        combined_pcd, combined_color = get_combined_pcd(
            colors, depths, camera_poses, intrinsic
        )

        

        self.view_grasps(per_obj_grasp_poses[0][:1], per_obj_grasp_widths[0][:1], name=f"scene/grasps/")

        self.view_pcd(combined_pcd, combined_color)
        self.view_pcd(kpts_3d, colors=np.zeros((4, 3)), size=0.02, name="scene/keypoints")



import argparse
import numpy as np

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("-f", "--fname")

#     # args = parser.parse_args()
#     # fname = args.fname
#     # pcd = open3d.io.read_point_cloud(fname)

#     # pts = np.asarray(pcd.points)

#     pts = np.eye(4)

#     from scipy.spatial.transform import Rotation as R

#     r = R.from_euler("xyz", [45, 60, 20], degrees=True)

#     pts[:3, :3] = r.as_matrix()
#     pts[:3, 3] = [0.5, 0, 0]
#     pts = [pts]
#     widths = [0.05]
#     # print(pts)
#     vis = VizServer()
#     ViewGrasps(vis, pts, widths)
#     meshcat_scene(vis, 0)

#     # open3d.visualization.draw_geometries([pcd])
