import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.path_util import get_dataset_dir
from utils.util import get_area
from utils.other_configs import *
from glob import glob
import json
from utils.gen_kpts import generate_keypoints, gen_kpts_norm_cano
from utils.seg import generate_bbox
from detectron2.structures import BoxMode
from utils.util import get_ori_clss
import random
from detectron2.structures import Keypoints, Boxes

# dict_keys(['intrinsic', 'camera_poses', 'grasp_poses', 'grasp_widths', 'grasp_collision', 'obj_types', 'obj_dims', 'obj_poses'])


def dataset_function(total_num_data) -> list[dict]:
    # total_num_data = NUM_TRAINING_DATA
    dataset_dir = get_dataset_dir()
    list_dict = []
    files_list = sorted(
        glob(os.path.join(dataset_dir, "*/color_images/*.png"), recursive=True)
    )
    assert total_num_data <= len(files_list)

    indexed_files = random.sample(files_list, total_num_data)

    for i, ppath in enumerate(indexed_files):
        split1 = ppath.split("/")
        scene_id = int(split1[-3])
        p = os.path.join(dataset_dir, split1[-3])
        imgid = int(split1[-1].split(".")[0].split("_")[-1])
        json_path = os.path.join(p, "scene_info.json")
        with open(json_path) as json_file:
            data = json.load(json_file)

        # TODO (TP): change it to get all 5000 data points
        current_dict = {
            "file_name": os.path.join(p, f"color_images/color_image_{imgid}.png"),
            "depth_file_name": os.path.join(p, f"depth_raw/depth_raw_{imgid}.npy"),
            "height": 480,
            "width": 640,
            "image_id": i,
            "seg_file_name": os.path.join(p, f"seg_labels/segmask_label_{imgid}.jpg"),
            "scene_id": scene_id,
        }

        bboxes = generate_bbox(current_dict["seg_file_name"])

        annotations = []
        num_grasps = []
        grasp_collisions = data["grasp_collision"]
        grasp_poses = data["grasp_poses"]
        grasp_widths = data["grasp_widths"]

        obj_dims = data["obj_dims"]
        obj_types = data["obj_types"]
        obj_poses = data["obj_poses"]

        cam_int = data["intrinsic"]
        cam_poses = data["camera_poses"]

        current_dict["cam_ext"] = cam_poses[imgid]
        current_dict["cam_int"] = cam_int

        total_grasps = len(grasp_poses)

        
        # bowl, small cylinder,  cuboid, big cylinder, stick
        for j in range(total_grasps):
            # TODO (TP): good to have a comment here for what these
            # two conditions are checking
            if (j + 1 in bboxes) and (get_area(bboxes[j + 1]) > 50):
                obj_pose = np.array(obj_poses[j])  # 4x4
                obj_dim = np.array(obj_dims[j])  # single element array
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
                    cam_poses[imgid],
                    depth=np.load(current_dict["depth_file_name"]),
                    draw=False,
                    draw_info=(current_dict["file_name"], obj_type, i, None),
                )

                projections_valid, ret, kpts_3d, kpts_2d, centers_, scales = result
                
                
                result = gen_kpts_norm_cano(
                    grasp_pose,
                    grasp_width,
                    cam_int,
                    cam_poses[imgid],
                    v_kpts=ret[:,:,2],
                    depth=np.load(current_dict["depth_file_name"]),
                    draw=False,
                    draw_info=(current_dict["file_name"], obj_type, i, None),
                )
                
                projections_valid, ret, kpts_3d, kpts_2d, centers, scales = result
                
                

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
                    "centers": centers[valid].reshape(1,-1)[0],
                    "keypoints": ret[valid].reshape(1,-1)[0],
                    "scales": scales[valid],
                    # "grasp_pose": grasp_pose[~grasp_collision],
                    # "grasp_width": grasp_width[~grasp_collision],
                    # "kpts_3d": kpts_3d[~grasp_collision],
                    # "kpts_2d": kpts_2d[~grasp_collision],
                    # "keypoints": ret[~grasp_collision][0],
                    "bbox": list(bboxes[j + 1]),
                    "bbox_mode": BoxMode.XYXY_ABS,
                }
                
                num_grasps.append(len(obj_dict["grasp_pose"]))
                annotations.append(obj_dict)
        # if(len(annotations) == 3):
        current_dict["annotations"] = annotations
        current_dict["num_grasps"] = num_grasps
        list_dict.append(current_dict)

    return list_dict


if __name__ == "__main__":
    # later, to access the data:
    l = dataset_function(2)
    print(len(l))
    # print(kpts_2d.shape)
