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
from utils.util import get_ori_clss

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

        # TODO (TP): fix the names below - variable name is depth but the files
        # are color images
        # TODO (TP): Try to write a for loop for this (both above and below)
        # TODO (TP): a suggestion: use `import os, os.path as osp` so that you 
        # can write os.path.join in short as osp.join
        other_depths = [
            os.path.join(p, "color_images/color_image_1.png"),
            os.path.join(p, "color_images/color_image_2.png"),
            os.path.join(p, "color_images/color_image_3.png"),
            os.path.join(p, "color_images/color_image_4.png"),
        ]

        # TODO (TP): change it to get all 5000 data points
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
                
        annotations = []

        grasp_collisions = data["grasp_collision"]
        grasp_poses = data["grasp_poses"]
        grasp_widths = data["grasp_widths"]

        obj_dims = data["obj_dims"]
        obj_types = data["obj_types"]
        obj_poses = data["obj_poses"]

        cam_int = data["intrinsic"]
        cam_poses = data["camera_poses"]

        total_grasps = len(grasp_poses)  
        
# bowl, small cylinder,  cuboid, big cylinder, stick    
        for j in range(total_grasps):

            # TODO (TP): good to have a comment here for what these 
            # two conditions are checking
            if (j+1 in bboxes) and (get_area(bboxes[j+1]) > 50):
                
                obj_pose = np.array(obj_poses[j])  #4x4
                obj_dim = np.array(obj_dims[j])   #single element array
                obj_type = obj_types[j]
                grasp_pose = np.array(grasp_poses[j])
                grasp_width = np.array(grasp_widths[j])
                grasp_collision = np.array(grasp_collisions[j])

                # TODO (TP): the optional arguments to use for draw_on_image
                # should have been left as comments for debugginf purposes.
                # Thanks to one of the commits - I found what the arguments were.
                # Thus, committing and pushing often - even for minor details - is helpful
                result = generate_keypoints(
                    grasp_pose,
                    grasp_width,
                    cam_int,
                    cam_poses[0],
                    depth=np.load(current_dict["depth_file_name"]),
                    draw=True,
                    draw_info=(current_dict["file_name"], obj_type, i, None)
                )
                
                projections_valid, ret, kpts_3d, kpts_2d, centers = result
                
                valid = np.logical_and(projections_valid, (~grasp_collision))

                if len(ret[valid]) == 0:
                    continue
                
                # kpts_3d: (num_grasps, 4, 3)
                # kpts_2d: (num_grasps, 4, 2)
                
                ori_clss = get_ori_clss(kpts_2d)
                
                # if len(ret[~grasp_collision])==0:
                #     continue
                
                
                obj_dict = {
                    "obj_pose": obj_pose,
                    "obj_dim": obj_dim,
                    "obj_type": obj_type,
                    "grasp_pose": grasp_pose[valid],
                    "grasp_width": grasp_width[valid],
                    "kpts_3d": kpts_3d[valid],
                    "kpts_2d": kpts_2d[valid],
                    "ori_clss": ori_clss[valid],
                    "centers": centers[valid],
                    "keypoints": ret[valid], 
                    # "grasp_pose": grasp_pose[~grasp_collision],
                    # "grasp_width": grasp_width[~grasp_collision],
                    # "kpts_3d": kpts_3d[~grasp_collision],
                    # "kpts_2d": kpts_2d[~grasp_collision],
                    # "keypoints": ret[~grasp_collision][0], 
                    "bbox":list(bboxes[j+1]),
                    "bbox_mode": BoxMode.XYXY_ABS   
                }

                annotations.append(obj_dict)
        # if(len(annotations) == 3):
        current_dict["annotations"] = annotations
        list_dict.append(current_dict)

    return list_dict


DatasetCatalog.register("KGN_dataset", dataset_function)


if __name__=="__main__":
    
# later, to access the data:
    l = dataset_function()
    datapoint = l[0]
    kpts_2d = datapoint["annotations"][0]["kpts_2d"]
    # print(kpts_2d.shape)
    