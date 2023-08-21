import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.path_util import get_dataset_dir
from visualize.vis_server import VizServer
from visualize.vis_grasp import ViewGrasps
from visualize.vis_scene import meshcat_scene

import json


def main(scene_id=0):
    dataset_dir = get_dataset_dir()
    json_path = os.path.join(dataset_dir, str(scene_id), "scene_info.json")
    with open(json_path) as json_file:
        data = json.load(json_file)

    grasp_poses = data["grasp_poses"]
    grasp_poses = np.array(
        [item for sublist in grasp_poses for item in sublist]
    )  # sum(grasp_poses, [])

    grasp_collision = data["grasp_collision"]
    grasp_collision = np.array(
        [item for sublist in grasp_collision for item in sublist]
    )

    grasp_widths = data["grasp_widths"]
    grasp_widths = np.array([item for sublist in grasp_widths for item in sublist])

    grasp_poses = grasp_poses[~grasp_collision]
    grasp_widths = grasp_widths[~grasp_collision]

    vis = VizServer()
    ViewGrasps(vis, grasp_poses, grasp_widths)
    meshcat_scene(vis, scene_id)


if __name__ == "__main__":
    scene_id = 3
    main(scene_id)
