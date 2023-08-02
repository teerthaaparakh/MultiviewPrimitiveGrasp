import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.other_configs import *
import random
import math, torch

def get_area(bbox):
    min_col, min_row, max_col, max_row = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height * width


def get_ori_clss(kpts_2d, ori_range=[-np.pi / 2, np.pi / 2]):
    # kpts_2d: (num_grasps, 4, 2)
    kpt_2 = kpts_2d[:, 1, :]
    kpt_3 = kpts_2d[:, 2, :]

    kpt_2x = kpt_2[:, 0]
    kpt_2y = kpt_2[:, 1]

    kpt_3x = kpt_3[:, 0]
    kpt_3y = kpt_3[:, 1]

    delta_x = kpt_3x - kpt_2x
    delta_y = kpt_3y - kpt_2y

    angle = np.arctan(delta_y / delta_x) + np.pi / 2

    bin_index = np.floor(angle / ((ori_range[1] - ori_range[0]) / NUM_BINS)).astype(int)
    return bin_index  # (num_grasps,)


    
def get_grasp_features(instances):
    ret = []
    for inst in instances:
        # get the centers points location relative to bounding box
        instances[inst].gt_centerpoints.tensor[:,0][:,0] = instances[inst].gt_centerpoints.tensor[:,0][:,0] \
                                                            - instances[inst].gt_boxes.tensor[:,0]
        instances[inst].gt_centerpoints.tensor[:,0][:,1] = instances[inst].gt_centerpoints.tensor[:,0][:,1] \
                                                            - instances[inst].gt_boxes.tensor[:,1]
                                                    
        # # get the grasp keypoints location relative to bounding box
        # instances[inst].gt_keypoints.tensor[:, :, 0] = instances[inst].gt_keypoints.tensor[:, :, 0] \
        #                                             - instances[inst].gt_boxes.tensor[:,0][:, None]
                                                    
        # instances[inst].gt_keypoints.tensor[:, :, 1] = instances[inst].gt_keypoints.tensor[:, :, 1] \
        #                                             - instances[inst].gt_boxes.tensor[:,1][:, None]  
                                                    
        # scale centers and keypoints with heights and widths
        widths = instances[0].gt_boxes.tensor[:,2] - instances[0].gt_boxes.tensor[:,0]
        heights = instances[0].gt_boxes.tensor[:,3] - instances[0].gt_boxes.tensor[:,1]
        
        instances[inst].gt_centerpoints.tensor[:,0][:,0] = instances[inst].gt_centerpoints.tensor[:,0][:,0]/widths
        instances[inst].gt_centerpoints.tensor[:,0][:,1] = instances[inst].gt_centerpoints.tensor[:,0][:,1]/heights
        
        # instances[inst].gt_keypoints.tensor[:, :, 0] = instances[inst].gt_keypoints.tensor[:, :, 0]/widths[:,None]
        # instances[inst].gt_keypoints.tensor[:, :, 1] = instances[inst].gt_keypoints.tensor[:, :, 1]/heights[:,None]
        
        concat_features = torch.cat((instances[inst].gt_keypoints.tensor[:,:,0:2].flatten(start_dim = 1), 
                                            instances[inst].gt_centerpoints.tensor[:,0][:, 0:2]), axis = 1)
        ret.append(concat_features)
        
    return torch.stack(ret)
        
        
def custom_random_generator(array, max_items):
    array = list(array)
    if len(array) >= max_items:
        return random.sample(array, max_items) 
    else:
        ret = [0]*max_items
        repetition = math.floor(max_items/len(array))
        total = repetition*len(array)
        remaining = max_items - total
        ret[0: total] = array*repetition
        ret[total: ] = random.sample(array, remaining)
        return ret
    
    
if __name__=="__main__":
    arr = range(10)
    max_items = 13
    print(custom_random_generator(arr, max_items))
    
    
    
