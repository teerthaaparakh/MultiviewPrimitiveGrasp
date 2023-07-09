import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.path_util import get_dataset_dir
from utils.util import get_area
from glob import glob
import json
from detectron2.data import DatasetCatalog
from dataloader.gen_kpts import generate_keypoints
from dataloader.seg import generate_bbox
from detectron2.structures import BoxMode

# dict_keys(['intrinsic', 'camera_poses', 'grasp_poses', 'grasp_widths', 'grasp_collision', 'obj_types', 'obj_dims', 'obj_poses'])

def dataset_function() -> list[dict]:
    dataset_dir = get_dataset_dir()
    list_dict = []
    folder_list = glob(os.path.join(dataset_dir, "*/"), recursive=False)

    for i, p in enumerate(folder_list):
        json_path = os.path.join(p, "scene_info.json")
        with open(json_path) as json_file:
            data = json.load(json_file)

        other_images = [
            os.path.join(p, "color_images/color_image_1.png"),
            os.path.join(p, "color_images/color_image_2.png"),
            os.path.join(p, "color_images/color_image_3.png"),
            os.path.join(p, "color_images/color_image_4.png"),
        ]

        other_depths = [
            os.path.join(p, "color_images/color_image_1.png"),
            os.path.join(p, "color_images/color_image_2.png"),
            os.path.join(p, "color_images/color_image_3.png"),
            os.path.join(p, "color_images/color_image_4.png"),
        ]

        current_dict = {
            "file_name": os.path.join(p, "color_images/color_image_0.png"),
            "depth_file_name": os.path.join(p, "depth_raw/depth_raw_0.npy"),
            "height": 480,
            "width": 640,
            "image_id": i,
            "seg_file_name": os.path.join(p, "seg_labels/segmask_label_0.jpg"),
            "other_images": other_images,
            "other_depths": other_depths,
        }

        bboxes = generate_bbox(current_dict["seg_file_name"])
        
        # TODO (TP): change it to get all 5000 data points
        
        annotations = []

        grasp_collisions = data["grasp_collision"]
        grasp_poses = data["grasp_poses"]
        grasp_widths = data["grasp_widths"]

        obj_dims = data["obj_dims"]
        obj_types = data["obj_types"]
        obj_poses = data["grasp_poses"]

        cam_int = data["intrinsic"]
        cam_poses = data["camera_poses"]

        total_obs = len(grasp_poses)

        for j in range(total_obs):
            if (j+1 in bboxes) and (get_area(bboxes[j+1]) > 50):
                
                obj_pose = np.array(obj_poses[j])
                obj_dim = np.array(obj_dims[j])
                obj_type = obj_types[j]
                grasp_pose = np.array(grasp_poses[j])
                grasp_width = np.array(grasp_widths[j])
                grasp_collision = np.array(grasp_collisions[j])

                result = generate_keypoints(
                    grasp_pose,
                    grasp_width,
                    cam_int,
                    cam_poses[0],
                    depth=np.load(current_dict["depth_file_name"]),
                )
                
                if result is None:
                    continue
                
                ret, kpts_3d, kpts_2d = result
                if len(ret[~grasp_collision])==0:
                    continue
                
                # TODO (TP): currently only first keypoint is taken
                obj_dict = {
                    "obj_pose": obj_pose,
                    "obj_dim": obj_dim,
                    "obj_type": obj_type,
                    "grasp_pose": grasp_pose[~grasp_collision],
                    "grasp_width": grasp_width[~grasp_collision],
                    "kpts_3d": kpts_3d[~grasp_collision],
                    "kpts_2d": kpts_2d[~grasp_collision],
                    "keypoints": ret[~grasp_collision][0], 
                    "bbox":list(bboxes[j+1]),
                    "bbox_mode": BoxMode.XYXY_ABS   
                }

                annotations.append(obj_dict)

        current_dict["annotations"] = annotations

        list_dict.append(current_dict)

    return list_dict


DatasetCatalog.register("KGN_dataset", dataset_function)


if __name__=="__main__":
    
# later, to access the data:
    l = dataset_function()