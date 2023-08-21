import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from dataloader.dataset_func import dataset_function
from utils.util import custom_random_generator

def dataset_function_vae(total_num_data):
    ll = dataset_function(total_num_data)
    ret_ll = []
    
    total_data_count = 0
    
    for data in ll:
        max_grasps = max(data["num_grasps"])
        grasp_indices = []
        total_obs = len(data["annotations"])
        for obs_no in range(total_obs):
            num_grasps_obs = data["num_grasps"][obs_no]
            grasp_indices.append(custom_random_generator(range(0,num_grasps_obs), max_grasps))
            
        for i in range(max_grasps):
            if total_data_count == total_num_data:
                break
            current_dict = {}
            current_dict["file_name"] = data["file_name"]
            current_dict["depth_file_name"] = data["depth_file_name"]
            current_dict["height"] = data
            current_dict["width"] = data["width"]
            current_dict["image_id"] = data["image_id"]
            current_dict["seg_file_name"] = data["seg_file_name"]
            current_dict["scene_id"] = data["scene_id"]
            current_dict["cam_ext"] = data["cam_ext"]
            current_dict["cam_int"] = data["cam_int"]
            orig_annos = data["annotations"]
            annotations = []
            for obs_no in range(total_obs):
                orig_obj_dict = orig_annos[obs_no]
                
                idid = grasp_indices[obs_no][i]
                obj_dict = {
                    "obj_pose": orig_obj_dict["obj_pose"],
                    "obj_dim": orig_obj_dict["obj_dim"],
                    "obj_type": orig_obj_dict["obj_type"],
                    "grasp_pose": orig_obj_dict["grasp_pose"][idid][None, :],
                    "grasp_width": np.array([orig_obj_dict["grasp_width"][idid]]),
                    "kpts_3d": orig_obj_dict["kpts_3d"][idid][None, :],
                    "kpts_2d": orig_obj_dict["kpts_2d"][idid][None, :],
                    "ori_clss": np.array([orig_obj_dict["ori_clss"][idid]]),
                    "centers": orig_obj_dict["centers"][idid*3:idid*3+3],
                    "keypoints": orig_obj_dict["keypoints"][idid*12:idid*12+12],
                    "scales": np.array([orig_obj_dict["scales"][idid]]),
                    "bbox": orig_obj_dict["bbox"],
                    "bbox_mode": orig_obj_dict["bbox_mode"],
                }
                annotations.append(obj_dict)
                
            current_dict["annotations"] = annotations
            ret_ll.append(current_dict)
            total_data_count += 1
    
        if total_data_count == total_num_data:
                break
            
    return ret_ll
            
            
    
    
if __name__=="__main__":
    ll = dataset_function_vae(2)    
    print(ll[0]["annotations"][0])
        