import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import sys, os
sys.path.append(os.environ["KGN_DIR"])
from visualize.vis_grasp import ViewGrasps
from visualize.vis_scene import meshcat_scene

# from pyngrok import ngrok


class VizServer:
    def __init__(self, port_vis=6000) -> None:
        zmq_url = f"tcp://127.0.0.1:{port_vis}"
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)

        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

        # http_tunnel = ngrok.connect(port_vis, bind_tls=False)
        # web_url = http_tunnel.public_url

        # print(f'Meshcat is now available at {web_url}')

    def view_cylinder(self, ht, r, X_pose, name = "scene/cylinder"):
        self.mc_vis[name].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=0x22DD22, opacity=1)
        )

        
        self.mc_vis[name].set_transform(X_pose)

    def view_pcd(self, pts, colors=None, name="scene"): 
        """
        inputs - 
        pts: Nx3
        colors: Nx3, 3 for RGB channels
        
        """
        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(
            g.PointCloud(pts.T, color=colors.T / 255.0, size=0.005)
        )

    def close(self):
        self.mc_vis.close()


import argparse
import numpy as np

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--fname")

    # args = parser.parse_args()
    # fname = args.fname
    # pcd = open3d.io.read_point_cloud(fname)

    # pts = np.asarray(pcd.points)

    
    pts = np.eye(4)
    
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('xyz', [45, 60, 20], degrees= True)
    
    pts[:3,:3] = r.as_matrix()
    pts[:3, 3] = [0.5,0,0]
    pts = [pts]
    widths = [0.05]
    # print(pts)
    vis = VizServer()
    ViewGrasps(vis, pts, widths)
    meshcat_scene(vis, 0)

    # open3d.visualization.draw_geometries([pcd])
